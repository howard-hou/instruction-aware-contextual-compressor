'''
This script is used to build large ranker dataset. 
large dataset contains 4 parts:
  - wikicn dataset
  - dureader dataset
  - T2Ranking dataset
  - m3e dataset
'''

import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer


document_paths = [
    "data/wikicn/wikipedia-cn-20230720-documents_all.json",
    "data/dureader/dureader-documents_all.json",
    "data/T2Ranking/document_filter.json",
    ]
dataset_paths = [
    "data/wikicn/wikipedia-cn-20230720-dataset",
    "data/dureader/dureader_dataset",
    "data/T2Ranking/T2Ranking_train_dataset",
    ]
# add m3d dataset
m3e_dir = Path("data/M3E_ranking_dataset/")
document_paths += [d for d in m3e_dir.glob("*/documents.json")]
dataset_paths += [d for d in m3e_dir.glob("*")]
print("num of document", len(document_paths))
print("num of dataset", len(dataset_paths))

output_path = "data/final_dataset/large_ranker_dataset"
# input shape
num_doc = 20
doc_max_length = 512
ques_max_length = 32
# model config
model_name_or_path = "google/umt5-small"
# model_name_or_path = "/data/huggingface_model/umt5-small"
num_proc = 32 # num of processes to use for data preprocessing

# init tokenizer
prompter_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 1. load document

docid2doc = {}
for documnet_path in tqdm(document_paths, desc="load document"):
    docid2doc.update(
        {d["docid"]: d["document"] for d in json.load(open(documnet_path))}
    )

train_datasets = []
test_datasets = {}
for dataset_path in tqdm(dataset_paths, desc="load dataset"):
    dataset_name = Path(dataset_path).name
    dataset = load_from_disk(dataset_path)
    if "train" in dataset:
        train_datasets.append(dataset["train"])
    if "test" in dataset:
        test_datasets[dataset_name] = dataset["test"]
    if isinstance(dataset, Dataset):
        dataset_dict = dataset.train_test_split(
            test_size=0.01, shuffle=True, seed=22
        )
        train_datasets.append(dataset_dict["train"])
        test_datasets[dataset_name] = dataset_dict["test"]
# concatenate all dataset
train_dataset = concatenate_datasets(train_datasets)
print("train dataset", train_dataset)
for name in test_datasets:
    print("test dataset", name, test_datasets[name])

# ### 2.2 tokenize dataset

# %%
def preprocess_dataset(example):
    pos_docid = example["docid"]
    # put pos_docid in the first place
    docids = [pos_docid] + [
        docid for docid in example["retrieved_docids"] if docid != pos_docid
    ]
    docs = [docid2doc[docid] for docid in docids[:num_doc]]
    # padding docs to num_doc
    if len(docs) < num_doc:
        docs += [docs[-1]] * (num_doc - len(docs))
    # padding to specific length, make all example have the same shape
    prompter_tokenzied_docs = prompter_tokenizer(
        docs, padding="max_length", truncation=True, max_length=doc_max_length
    )
    prompter_tokenzied_question = prompter_tokenizer(
        example["question"],
        padding="max_length",
        truncation=True,
        max_length=ques_max_length,
    )
    return {
        "document_input_ids": prompter_tokenzied_docs.input_ids,
        "document_attention_mask": prompter_tokenzied_docs.attention_mask,
        "prompter_question_input_ids": prompter_tokenzied_question.input_ids,
        "prompter_question_attention_mask": prompter_tokenzied_question.attention_mask,
    }

all_dataset = DatasetDict()
tokenized_train_dataset = train_dataset.map(preprocess_dataset, num_proc=num_proc)
all_dataset["train"] = tokenized_train_dataset
for name in test_datasets:
    test_dataset = test_datasets[name]
    all_dataset["test_"+name] = test_dataset.map(preprocess_dataset, num_proc=num_proc)
print(all_dataset)
all_dataset.save_to_disk(output_path)