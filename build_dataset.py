'''
This script is used to build prompter dataset. 
medium dataset contains 2 parts:
  - wikicn dataset
  - dureader dataset
'''

import json
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer


document_paths = [
    "data/wikicn/wikipedia-cn-20230720-documents_all.json",
    "data/dureader/dureader-documents_all.json",
    ]
dataset_paths = [
    "data/wikicn/wikipedia-cn-20230720-dataset",
    "data/dureader/dureader_dataset",
    ]
output_path = "data/final_dataset/prompter_dataset"
# input shape
num_doc = 5
doc_max_length = 1024
ques_max_length = 32
ans_max_length = 128
# model config
# model_name_or_path = "google/umt5-small"
model_name_or_path = "/data/huggingface_model/umt5-small"
num_proc = 16 # num of processes to use for data preprocessing

# init tokenizer
prompter_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 1. load document

docid2doc = {}
for documnet_path in document_paths:
    docid2doc.update(
        {d["docid"]: d["document"] for d in json.load(open(documnet_path))}
    )

train_datasets = []
test_datasets = {}
for dataset_path in dataset_paths:
    dataset_name = Path(dataset_path).name
    dataset = load_from_disk(dataset_path)
    if "train" in dataset:
        train_datasets.append(dataset["train"])
    if "test" in dataset:
        test_datasets[dataset_name] = dataset["test"]
    if isinstance(dataset, Dataset):
        train_datasets.append(dataset)
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
    prompter_tokenzied_answer = prompter_tokenizer(
        example["answer"], 
        padding="max_length", 
        truncation=True,
        max_length=ans_max_length
    )
    return {
        "document_input_ids": prompter_tokenzied_docs.input_ids,
        "document_attention_mask": prompter_tokenzied_docs.attention_mask,
        "prompter_question_input_ids": prompter_tokenzied_question.input_ids,
        "prompter_question_attention_mask": prompter_tokenzied_question.attention_mask,
        "prompter_answer_input_ids": prompter_tokenzied_answer.input_ids,
        "prompter_answer_attention_mask": prompter_tokenzied_answer.attention_mask,
    }

all_dataset = DatasetDict()
tokenized_train_dataset = train_dataset.map(preprocess_dataset, num_proc=num_proc)
all_dataset["train"] = tokenized_train_dataset
for name in test_datasets:
    test_dataset = test_datasets[name]
    all_dataset["test_"+name] = test_dataset.map(preprocess_dataset, num_proc=num_proc)
print(all_dataset)
all_dataset.save_to_disk(output_path)
