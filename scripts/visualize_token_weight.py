import torch
from transformers import AutoTokenizer, AutoModel
import datasets
import argparse
import json
from tqdm import tqdm
from pathlib import Path

doc_max_length = 1024
ques_max_length = 32

def parse_args():
    parser = argparse.ArgumentParser(description="rerank dataset")
    parser.add_argument('dataset_path', type=str, help='dataset name')
    parser.add_argument('model_path', type=str, help='model path')
    parser.add_argument('document_path', type=str, help='document path')
    parser.add_argument('output_path', type=str, help='output path')
    parser.add_argument('--max_new_tokens', type=int, default=16, help='max new tokens')
    parser.add_argument('--keep_ratio', type=float, default=0.5, help='keep ratio')
    parser.add_argument('--device', type=str, default='cuda', help='device name')
    return parser.parse_args()


def render_text_to_latex(tokens, weights, sep="。"):
    # norm to weight to [0, 100]
    weights = 100 * (weights / max(weights))
    styled_tokens = []
    splits = []
    scores = []
    cnt = 0
    weight_sum = 0
    for token, weight in zip(tokens, weights):
        weight = round(weight, 2)
        token = token.replace("▁", " ")
        token = token.replace("<unk>", "\n")
        if token == sep:
            sentence_weight = round(weight_sum / cnt, 2)
            splits.append(styled_tokens)
            scores.append(sentence_weight)
            styled_tokens = []
            cnt = 0
            weight_sum = 0
            
        styled_token = f"\colorize{{{weight}}}{{{token}}}"
        styled_tokens.append(styled_token)
        cnt += 1
        weight_sum += weight
    return splits, scores


def drop_context(
    doc_splits,
    scores,
    keep_ratio=0.5,
):
    sorted_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    # at least keep one context
    num_drop = max(int(len(sorted_idx) * keep_ratio), 1)
    # keep order of document
    sorted_idx = sorted(sorted_idx[:num_drop])
    doc_splits_keep = [doc_splits[i] for i in sorted_idx]
    return doc_splits_keep


def is_english(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    

def main():
    args = parse_args()
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    # set decoder_start_token_id = 2, 至关重要，否则解码完全是错的
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    ds = datasets.load_from_disk(args.dataset_path)
    #
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)   
    documents = json.load(open(args.document_path))
    docid2doc = {d["docid"]:d["document"] for d in documents}
    model.to(args.device)
    model.eval()

    for s in tqdm(ds):
        is_eng = is_english(s["question"])
        if not is_eng:
            continue
        #retrieved_docids = s["reranked_docids"][:1]
        retrieved_docids = [s["docid"]]
        docs = [docid2doc[docid] for docid in retrieved_docids]
        doc_input = tokenizer(docs,         
                              padding=True,
                              truncation=True,
                              max_length=doc_max_length,
                              return_tensors="pt")
        doc_input.input_ids = doc_input.input_ids.to(args.device)
        doc_input.attention_mask = doc_input.attention_mask.to(args.device)
        ques_input = tokenizer(s["question"],
                               padding=True,
                               truncation=True,
                               max_length=ques_max_length,
                               return_tensors="pt")
        ques_input.input_ids = ques_input.input_ids.to(args.device)
        ques_input.attention_mask = ques_input.attention_mask.to(args.device)
        
        tokens_output, lm_gradcams_output = model.compute_lm_grad_cam(
            document_input_ids=doc_input.input_ids, 
            document_attention_mask=doc_input.attention_mask, 
            question_input_ids=ques_input.input_ids, 
            question_attention_mask=ques_input.attention_mask,
            max_new_tokens=args.max_new_tokens,
            reduction="sum",
            block_num=-3)
        #
        doc_tokens = tokenizer.convert_ids_to_tokens(doc_input.input_ids[0])
        token_weights = lm_gradcams_output[0].cpu().detach().numpy()
        doc_splits, scores = render_text_to_latex(
            doc_tokens,
            token_weights, 
            sep = "。")
        # replace unk with \n
        compressed_styled_tokens = drop_context(
            doc_splits,
            scores,
            keep_ratio=args.keep_ratio)
        compressed_styled_sentences = ["\n".join(tokens) for tokens in compressed_styled_tokens]
        compressed_styled_doc = "\n".join(compressed_styled_sentences)

        # save to file
        output_sub_dir = output_dir / docs[0][:10]
        output_sub_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_sub_dir / f"{s['question']}.tex"
        with open(output_file, "w") as f:
            f.write(compressed_styled_doc)


if __name__ == "__main__":
    main()
