from transformers import AutoTokenizer
import argparse
import torch
from pathlib import Path
from model.modeling_rankingprompter import RankingPrompterForPreTraining, RankingPrompter
from model.configuration_rankingprompter import RankingPrompterConfig

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['model']
    new_state_dict = {}
    unwanted_prefix = '_orig_mod.'
    for k in state_dict:
        if k.startswith(unwanted_prefix):
            new_state_dict[k[len(unwanted_prefix):]] = state_dict[k]
    msg = model.load_state_dict(new_state_dict)
    print("checkpoint loaded: ", msg)

def export_model(args):
    # register
    RankingPrompterConfig.register_for_auto_class()
    if args.model_type == "rankingprompter":
        RankingPrompter.register_for_auto_class("AutoModel")
    elif args.model_type == "rankingprompter_pretraining":
        RankingPrompterForPreTraining.register_for_auto_class("AutoModel")
    else:
        raise ValueError("model_type must be either 'rankingprompter' or 'rankingprompter_pretraining'")
    #
    prompter_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    prompter_config = RankingPrompterConfig.from_pretrained(args.model_name_or_path)
    prompter_config._name_or_path=args.model_name_or_path
    if args.model_type == "rankingprompter":
        model = RankingPrompter(prompter_config)
    elif args.model_type == "rankingprompter_pretraining":
        model = RankingPrompterForPreTraining(prompter_config)
    load_checkpoint(model, args.checkpoint_path)
    # 
    prompter_tokenizer.save_pretrained(args.output_dir)
    prompter_config.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    print("model saved to: ", args.output_dir)
    if args.push_to_hub:
        print("pushing to hub...")
        repo_name = Path(args.output_dir).name
        prompter_tokenizer.push_to_hub(repo_name)
        prompter_config.push_to_hub(repo_name)
        commit_info = model.push_to_hub(repo_name)
        print("pushed to: ", commit_info)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--model_type", type=str, default="rankingprompter")
    return parser.parse_args()

def main():
    args = get_args()
    export_model(args)


if __name__ == "__main__":
    main()
