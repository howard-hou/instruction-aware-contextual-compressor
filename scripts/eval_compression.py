import json
import os
import jieba
from tqdm import tqdm
from rouge_chinese import Rouge
import argparse
from pathlib import Path
from datasets import load_from_disk
import openai
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_LEN = 4000
openai.api_key = "YOUR_API_KEY"

encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
system_prompt = "You are a helpful assistant."


# setup chatgpt api and model, fix to gpt-3.5-turbo-0613
def get_chat_response(user_prompt, top_p=0.1):
    response_data = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        top_p=top_p,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response_data["choices"][0]["message"]["content"]

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def cutoff_string_to_max_tokens(string: str, max_len: int = 4096):
    tokens = encoding.encode(string)[:max_len]
    return encoding.decode(tokens)


# design a prompt here
def get_prompt(question, document):
    prompt = "根据以下文本，回答问题：\n"
    prompt += f"问题：{question}\n"
    prompt += f"文本：{document}\n"
    prompt += "答案："
    return prompt


def run_compression_response(dataset_path, response_file, eval_output_file):
    def get_chatgpt_response(example):
        doc = example['compressed_context']
        question = example["question"]
        prompt = get_prompt(question, doc)
        prompt = cutoff_string_to_max_tokens(prompt, max_len=MAX_LEN)
        response = get_chat_response(prompt)
        return {"response": response}
        
    ds = load_from_disk(dataset_path)
    
    docid_set = set()
    if os.path.exists(response_file):
        for l in open(response_file):
            e = json.loads(l)
            docid_set.add(e["docid"])
    # version 2, with retry
    pbar = tqdm(total=len(ds))
    w = open(response_file, "a")
    with ThreadPoolExecutor(max_workers=10) as exe:
        future_to_example = {}
        for example in ds:
            docid = example["docid"]
            if docid in docid_set:
                continue
            future = exe.submit(get_chatgpt_response, example, 60)
            future_to_example[future] = example
        
        for future in as_completed(future_to_example):
            example = future_to_example[future]
            try:
                response = future.result()
            except:
                print("fail: ", example["question"])
                continue
            example.update(response)
            w.write(json.dumps(example, ensure_ascii=False) + "\n")
            pbar.update(1)

    hyps, refs = [], []
    for l in open(response_file):
        e = json.loads(l)
        hyp = ' '.join(jieba.cut(e["response"], HMM=False))
        ref = ' '.join(jieba.cut(e["answer"], HMM=False))
        hyps.append(hyp)
        refs.append(ref)
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    json.dump(scores, open(eval_output_file, "w"), ensure_ascii=False, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="rerank dataset")
    parser.add_argument('input_dir', type=str, help='dataset name')
    parser.add_argument('output_dir', type=str, help='output path')

    return parser.parse_args()


def main():
    args = parse_args()
    eval_output_dir = Path(args.output_dir)
    input_datasets = Path(args.input_dir).glob("*")
    for dataset_path in input_datasets:
        dataset_path = Path(dataset_path)
        response_file = eval_output_dir / f"{dataset_path.stem}_response.jsonl"
        eval_output_file = eval_output_dir / f"{dataset_path.stem}_eval.json"
        run_compression_response(dataset_path, response_file, eval_output_file)

if __name__ == "__main__":
    main()
