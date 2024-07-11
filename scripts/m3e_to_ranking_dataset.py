from pathlib import Path
import json
import numpy as np
import hashlib
from random import randrange
from dataclasses import dataclass
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets, load_from_disk

from sentence_transformers import SentenceTransformer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="M3E dataset")
    parser.add_argument('--device', type=str, default='cuda', help='device name')
    parser.add_argument('--num_proc', type=int, default=32, help='device name')
    return parser.parse_args()

args = parse_args()

model = SentenceTransformer('moka-ai/m3e-base').to(args.device)
print(f"model in {model.device}")

@dataclass
class M3EHfDatsetWithInfo:
    hf_dataset: HfDataset
    name: str
    instruction: str = ''


def load_all_datasets(m3e_datasets_dir: Path) -> list[M3EHfDatsetWithInfo]:
    m3e_datasets = []
    for data_dir in m3e_datasets_dir.glob('*.dataset'):
        dataset_name = data_dir.stem
        dataset_dict = load_from_disk(str(data_dir))
        if isinstance(dataset_dict, dict):
            dataset: HfDataset = concatenate_datasets(list(dataset_dict.values()))
        else:
            dataset = dataset_dict
        m3e_datasets.append(
            M3EHfDatsetWithInfo(
                hf_dataset=dataset,
                name=dataset_name,
            )
        )
        print(f'load {dataset_name}')
    return m3e_datasets


def hash_to_12_length(input_string):
    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Convert the input string to bytes and update the hash object
    md5_hash.update(input_string.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    hashed_string = md5_hash.hexdigest()

    # Truncate the hash to 12 characters
    truncated_hash = hashed_string[:12]

    return truncated_hash

def generate_docid(example):
    return {"docid": hash_to_12_length(example["document"])}

def save_documents(hf_dataset, name, document_path):
    documents = []
    docid_set = set()
    for docid, doc in zip(hf_dataset["docid"], hf_dataset["document"]):
        docid2doc = dict(docid=docid, document=doc, source=name)
        if docid not in docid_set:
            documents.append(docid2doc)
            docid_set.add(docid)
    json.dump(documents, open(document_path, "w"), ensure_ascii=False, indent=2)
    return documents

def build_documents_index(document_dataset):
    # document_dataset = HfDataset.from_list(documents)
    # 
    document_dataset_with_emb = document_dataset.map(
        lambda example: {'doc_embedding': model.encode(example["document"])}, 
        batched=True)
    document_dataset_with_emb.add_faiss_index(column='doc_embedding')
    return document_dataset_with_emb


def encode_question(questions):
    question_dataset = HfDataset.from_list(questions)
    # 
    question_dataset_with_emb = question_dataset.map(
        lambda example: {'question_embedding': model.encode(example["question"])}, 
        batched=True
    )
    return question_dataset_with_emb


def run_retriever(document_dataset_with_emb, question_dataset_with_emb, topk=100, num_proc=32):
    def retrieve_topk_documents(example):
        ques_embedding = np.array(example["question_embedding"], dtype=np.float32)
        scores, retrieved_examples = document_dataset_with_emb.get_nearest_examples('doc_embedding', ques_embedding, k=topk)
        example["retrieved_docids"] = retrieved_examples["docid"]
        example["retrieved_doc_scores"] = scores.tolist()
        return example
    # 根据硬件，调整num_proc
    question_dataset_with_retrieval = question_dataset_with_emb.map(retrieve_topk_documents, 
                                                                    num_proc=num_proc,
                                                                    remove_columns=["question_embedding"])
    # keep columns
    keep_columns = ['question', 'answer', 'docid', 'retrieved_docids', 'retrieved_doc_scores']
    question_dataset_with_retrieval = question_dataset_with_retrieval.remove_columns(
        [col for col in question_dataset_with_retrieval.column_names if col not in keep_columns])
    return question_dataset_with_retrieval

def compute_topk_accuracy(predictions, true_labels, topk=5):
    assert len(predictions) == len(true_labels), "预测结果和真实标签的数量必须相同"
    
    num_correct = 0
    for pred, true_label in zip(predictions, true_labels):
        if true_label in pred[:topk]:
            num_correct += 1
    
    top1_accuracy = num_correct / len(predictions)
    return top1_accuracy

def check_badcase(questions, predictions, true_labels, topk=5):
    assert len(predictions) == len(true_labels), "预测结果和真实标签的数量必须相同"
    
    bad_cases = []
    for ques, pred, true_label in zip(questions, predictions, true_labels):
        if true_label not in pred[:topk]:
            bad_cases.append(ques)
    return bad_cases

def eval_retriever(question_dataset_with_retrieval):
    questions, predictions, targets = [], [], []
    for s in question_dataset_with_retrieval:
        questions.append(s["question"])
        predictions.append(s["retrieved_docids"])
        targets.append(s["docid"])

    results = {}
    for k in [1, 3, 5, 10, 20, 50, 100]:
        topk_acc = compute_topk_accuracy(predictions, targets, topk=k)
        results[f"top{k}"] = round(topk_acc, 3)
    print(results)
    return results

def get_shard_of_dataset(dataset, sample_per_shard=100000):
    num_shards = len(dataset) // sample_per_shard + 1
    for i in range(num_shards):
        yield i, dataset.shard(num_shards=num_shards, index=i)
    

def process_one_dataset(dataset, out_dir):
    print(f"process {dataset.name}, total size {len(dataset.hf_dataset)}")
    dataset_dir = out_dir / dataset.name
    dataset_dir.mkdir(exist_ok=True)
    hf_dataset = dataset.hf_dataset.rename_columns({"text_pos": "document", "text": "question"})
    hf_dataset = hf_dataset.filter(lambda x: x["document"] is not None and len(x["document"]) > 0 and len(x["question"]) > 0)
    hf_dataset = hf_dataset.map(generate_docid, num_proc=args.num_proc)
    save_documents(hf_dataset, dataset.name, dataset_dir / "documents.json")
    eval_results = []
    output_datasets = []
    for i, shard in get_shard_of_dataset(hf_dataset):
        print(f"process {dataset.name} - shard {i}")
        document_dataset_with_emb = build_documents_index(shard)
        question_dataset_with_emb = encode_question(shard)
        question_dataset_with_retrieval = run_retriever(document_dataset_with_emb, 
                                                        question_dataset_with_emb, 
                                                        topk=100, 
                                                        num_proc=args.num_proc)
        eval_results.append(eval_retriever(question_dataset_with_retrieval))
        output_datasets.append(question_dataset_with_retrieval)
        if i == 1:
            break
    output_dataset = concatenate_datasets(output_datasets)
    output_dataset.save_to_disk(dataset_dir)
    json.dump(eval_results, open(dataset_dir /  "eval.json", "w"), ensure_ascii=False, indent=2)

m3e_datasets = load_all_datasets(Path("M3E_dataset/"))
out_dir = Path("M3E_ranking_dataset")
out_dir.mkdir(exist_ok=True)
for dataset in m3e_datasets:
    process_one_dataset(dataset, out_dir)
