from transformers import AutoTokenizer, AutoModel
import argparse
from pathlib import Path


def push_to_hub(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, 
                                      trust_remote_code=True)
    print("pushing to hub...")
    repo_name = Path(args.model_path).name
    tokenizer.push_to_hub(repo_name)
    commit_info = model.push_to_hub(repo_name)
    print("pushed to: ", commit_info)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    return parser.parse_args()

def main():
    args = get_args()
    push_to_hub(args)


if __name__ == "__main__":
    main()