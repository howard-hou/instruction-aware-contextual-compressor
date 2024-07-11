from pathlib import Path
import numpy as np
import time
from datetime import timedelta
import os
import json
from tqdm import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datasets import load_from_disk
from transformers import AutoTokenizer
from model.modeling_rankingprompter import RankingPrompterForPreTraining
from misc import count_parameters, recursive_round, estimate_remaining_time

torch.set_float32_matmul_precision("high")  # use the TensorFloat32

# I/O
log_interval = 100
model_output_dir = "saved_model/ranker"
# model config
model_name_or_path = "google/umt5-small"
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'ranking-prompter'
wandb_run_name = 'ranking-prompter-for-pretraining'
# data
dataset_path = "data/final_dataset/wiki_dureader_T2Ranking_for_ranker"
batch_size = 2  # if gradient_accumulation_steps > 1, this is the micro-batch size
test_batch_size = 4
gradient_accumulation_steps = 8  # accumulate gradients over n batches
num_workers = 4  # num of processes to use for data loading
# optimizer config
learning_rate = 1e-4
lr_end = 1e-7
# scheduler config
num_epochs = 1
num_warmup_steps = 1000
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
ddp_timeout = 600 # minutes

# system
# device examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
device = "cuda"
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
print("config:", json.dumps(config, indent=2, ensure_ascii=False))
# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend, timeout=timedelta(minutes=ddp_timeout))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    ddp_local_rank = 0
    ddp_world_size = 1
    seed_offset = 0
samples_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size
print(f"samples per iteration will be: {samples_per_iter:,}")

model_output_dir = Path(model_output_dir)
if master_process:
    model_output_dir.mkdir(exist_ok=True, parents=True)
torch.manual_seed(1337 + seed_offset)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

prompter_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = RankingPrompterForPreTraining.from_pretrained(model_name_or_path).to(device)
model.enable_amp_ctx(device_type=device_type, dtype=ptdtype)
trainable_params, all_param = count_parameters(model)
print(
    "prompter trainable params: {:.2f}B || all params: {:.2f}B || trainable%: {:.4f}".format(
        trainable_params / 1e9, all_param / 1e9, 100 * trainable_params / all_param
    )
)

## 2. data
### 2.1 load data
from torch.utils.data import DataLoader

remove_columns = ["question", "answer", "docid", 'retrieved_docids', 'retrieved_doc_scores']
dataset_dict = load_from_disk(dataset_path)
for name in dataset_dict:
    drop_columns = [c for c in dataset_dict[name].column_names if c in remove_columns]
    dataset_dict[name] = dataset_dict[name].remove_columns(drop_columns)

dataset_dict = dataset_dict.with_format("torch")
if master_process:
    print("train_set: ", dataset_dict["train"])
# Create a DataLoader with the desired batch size
train_subset = dataset_dict["train"].shard(ddp_world_size, ddp_local_rank)
train_dataloader = DataLoader(
    train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
print(f"train_dataloader-{ddp_local_rank}: {len(train_dataloader)} batches")
test_dataloader = {}
testset_names = [name for name in dataset_dict if "test" in name]
for i, name in enumerate(testset_names):
    if i % ddp_world_size == ddp_local_rank:
        test_dataloader[name] = DataLoader(
            dataset_dict[name], batch_size=test_batch_size, 
            num_workers=num_workers, shuffle=False
        )
print(f"test_dataloader-{ddp_local_rank}: {test_dataloader.keys()}")

# %% [markdown]
# ## 3. train
# ### 3.1 config optimizer and scheduler

# %%
import inspect
from transformers import get_polynomial_decay_schedule_with_warmup

# Create AdamW optimizer and use the fused version if it is available
fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device == "cuda"
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, **extra_args)
print(f"using fused AdamW: {use_fused}")
# scheduler config
num_update_steps = num_epochs * len(train_dataloader) // gradient_accumulation_steps
lr_scheduler = get_polynomial_decay_schedule_with_warmup(
    optimizer=optimizer,  # scheduler是针对optimizer的lr的
    lr_end=lr_end,
    power=1,  # 当power=1时（默认）等价于linear_schedule_with_warmup
    num_warmup_steps=num_warmup_steps // gradient_accumulation_steps,
    num_training_steps=num_update_steps,
)
print(f"num_update_steps: {num_update_steps} for optimizer")
# saving step
num_saves = 5
total_steps = num_epochs * len(train_dataloader)
saving_steps = [1+int(i* total_steps / num_saves) for i in range(1, num_saves)]
saving_steps = [1] + saving_steps + [total_steps]
print(f"saving_steps: {saving_steps}")

# %% [markdown]
# ### 3.2 traininig
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# generate labels for training and evaluation
def generate_labels(document_input_ids):
    # ranking labels, the first document is the positive one
    batch_size, num_doc = document_input_ids.shape[:2]
    labels = torch.zeros(batch_size, num_doc, device=device)
    labels[:, 0] = 1
    return labels


from metric import RetrievalMetrics
# helps estimate the model
@torch.no_grad()
def evaluate_prompter(model, dataloader):
    out = {}
    model.eval()
    losses = []
    retrieval_metrics = RetrievalMetrics()
    for batch in dataloader:
        document_input_ids = batch["document_input_ids"].to(device)
        document_attention_mask = batch["document_attention_mask"].to(device)
        prompter_question_input_ids = batch["prompter_question_input_ids"].to(device)
        prompter_question_attention_mask = batch["prompter_question_attention_mask"].to(
            device
        )
        labels = generate_labels(document_input_ids)
        prompter_output = model(
            document_input_ids=document_input_ids,
            document_attention_mask=document_attention_mask,
            question_input_ids=prompter_question_input_ids,
            question_attention_mask=prompter_question_attention_mask,
            labels=labels,
        )
        loss = prompter_output.loss.item()
        losses.append(loss)
        # ranking metric
        batch_size, num_doc = document_input_ids.shape[:2]
        rank_preds = prompter_output.logits.cpu().tolist()
        rank_targets = [[True] + [False] * (num_doc - 1) for _ in range(batch_size)]
        retrieval_metrics.update(rank_preds, rank_targets)

    out["ranker_val_loss"] = np.mean(losses)
    out["retrieval"] = retrieval_metrics.compute()
    model.train()
    return recursive_round(out)


train_loss = []
step = 0  # total steps = num_training_steps * gradient_accumulation_steps
start_time = time.time()
raw_model = model.module if ddp else model # unwrap DDP container if needed
for epoch in range(num_epochs):
    # Iterate through batches
    for batch in train_dataloader:
        document_input_ids = batch["document_input_ids"].to(device)
        document_attention_mask = batch["document_attention_mask"].to(device)
        prompter_question_input_ids = batch["prompter_question_input_ids"].to(device)
        prompter_question_attention_mask = batch["prompter_question_attention_mask"].to(device)
        labels = generate_labels(document_input_ids)
        prompter_output = model(
            document_input_ids=document_input_ids,
            document_attention_mask=document_attention_mask,
            question_input_ids=prompter_question_input_ids,
            question_attention_mask=prompter_question_attention_mask,
            labels=labels,
        )
        loss = prompter_output.loss / gradient_accumulation_steps
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_loss.append(prompter_output.loss.item())
        # log the loss on train set
        if step % log_interval == 0 and master_process:
            remaining_time = estimate_remaining_time(start_time, step, total_steps)
            print(f"step {step}/{total_steps}:",
                  f"train/loss {np.mean(train_loss):.4f}",
                  f"lr {lr_scheduler.get_lr()[0]:.7f}", 
                  f"remaining time {remaining_time}", sep="  ")
            if wandb_log:
                wandb.log({
                    "step": step,
                    "train/loss": np.mean(train_loss), 
                    "lr": lr_scheduler.get_lr()[0]})
            train_loss = []
        step += 1

        if step in saving_steps:
            eval_results = {}
            for name in tqdm(test_dataloader, desc=f"evaluating-{ddp_local_rank}"):
                eval_results[name] = evaluate_prompter(raw_model, test_dataloader[name])
            progress = f"{int(100*step / total_steps)}%"
            eval_output_path = model_output_dir / f"eval_results_{progress}-{ddp_local_rank}.json"
            json.dump(eval_results, open(eval_output_path, "w"), indent=2, ensure_ascii=False)
            if master_process:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": raw_model.config,
                    "iter_num": step,
                    "config": config,
                }
                datetime = time.strftime("%Y%m%d", time.localtime())
                model_output_path = model_output_dir / f"RankingPrompterForPreTraining_{progress}_{datetime}.pth"
                print(f"saving checkpoint to {model_output_path}")
                torch.save(checkpoint, model_output_path)

if ddp:
    destroy_process_group()