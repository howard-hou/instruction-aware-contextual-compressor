from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
from contextlib import nullcontext

import torch
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer
from model.modeling_rankingprompter import Ranker
from misc import recursive_round

torch.set_float32_matmul_precision("high")  # use the TensorFloat32

# I/O
ckpt_path = "saved_model/ranker/best_dureader-wikicn-T2Ranking_ckpt.pth"
eval_output_path = "saved_model/ranker/eval_results.json"
# model config
model_name_or_path = "google/umt5-small"
# data
dataset_path = "data/final_dataset/large_ranker_dataset"
batch_size = 2  # if gradient_accumulation_steps > 1, this is the micro-batch size
test_batch_size = 4
num_workers = 4  # num of processes to use for data loading
# optimizer config
learning_rate = 1e-4
lr_end = 1e-7
# scheduler config
num_epochs = 1
num_warmup_steps = 1000
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

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
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

prompter_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
prompter_config = AutoConfig.from_pretrained(model_name_or_path)
prompter_config.ctx = ctx

model = Ranker(prompter_config).to(device)
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
msg = model.load_state_dict(state_dict)
print("model loaded:", msg)

## 2. data
### 2.1 load data
from torch.utils.data import DataLoader

dataset_dict = load_from_disk(dataset_path).remove_columns(
    ["question", "docid", 'retrieved_docids', 'retrieved_doc_scores']).with_format("torch")
print(dataset_dict)
# filter data with different shape
test_dataloader = {}
for name in dataset_dict:
    if "test" in name:
        test_dataloader[name] = DataLoader(
            dataset_dict[name], batch_size=test_batch_size, 
            num_workers=num_workers, shuffle=False
        )

from metric import RetrievalMetrics

# helps estimate the model
@torch.no_grad()
def evaluate_prompter(model, dataloader, name):
    out = {}
    model.eval()
    losses = []
    retrieval_metrics = RetrievalMetrics()
    for batch in tqdm(dataloader, desc=f"evaluating {name}"):
        document_input_ids = batch["document_input_ids"].to(device)
        document_attention_mask = batch["document_attention_mask"].to(device)
        prompter_question_input_ids = batch["prompter_question_input_ids"].to(device)
        prompter_question_attention_mask = batch["prompter_question_attention_mask"].to(
            device
        )
        prompter_output = model(
            document_input_ids=document_input_ids,
            document_attention_mask=document_attention_mask,
            question_input_ids=prompter_question_input_ids,
            question_attention_mask=prompter_question_attention_mask,
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
    return recursive_round(out)


eval_results = {}
for name in test_dataloader:
    eval_result = evaluate_prompter(model, test_dataloader[name], name)
    print(f"{name}-eval reuslt:", eval_result)
    eval_results[name] = eval_result

json.dump(eval_results, open(eval_output_path, "w"), 
          indent=2, ensure_ascii=False)