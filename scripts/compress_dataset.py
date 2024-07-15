import torch
from transformers import AutoTokenizer, AutoModel
import datasets
import argparse
import json
from tqdm import tqdm

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


def split_doc(doc, sep="。"):
    return [d.strip() for d in doc.split(sep) if d.strip()]

def map_lm_score_to_text(lm_text, lm_score):
    lm_text2score = {}
    for one_split, score in zip(lm_text, lm_score):
        if not one_split.strip():
            continue
        sentences = one_split.split("。")
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                lm_text2score[sentence] = score
    return lm_text2score


def compress_context_by_avg_rank(docs, ranking_score, lm_score, keep_ratio=0.5):
    assert len(ranking_score) == len(lm_score) == len(docs)
    num_docs = len(docs)
    # sort doc index by logits from high to low
    ranking_sorted_idx = sorted(range(num_docs), key=lambda k: ranking_score[k], reverse=True)
    lm_sorted_idx = sorted(range(num_docs), key=lambda k: lm_score[k], reverse=True)
    # at least keep one context
    num_drop = max(int(num_docs * keep_ratio), 1)
    # sort by average rank of ranking and lm
    avg_rank = [(ranking_sorted_idx.index(i) + lm_sorted_idx.index(i)) / 2 for i in range(num_docs)]
    sorted_idx = sorted(range(len(avg_rank)), key=lambda k: avg_rank[k])
    # keep order of document
    sorted_idx = sorted(sorted_idx[:num_drop])
    compressed_docs = [docs[i] for i in sorted_idx]
    return compressed_docs
    

def main():
    args = parse_args()
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    # set decoder_start_token_id = 2, 至关重要，否则解码完全是错的
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    ds = datasets.load_from_disk(args.dataset_path)
    print(ds)
    documents = json.load(open(args.document_path))
    docid2doc = {d["docid"]:d["document"] for d in documents}
    model.to(args.device)
    model.eval()
    output_list = []
    for s in tqdm(ds):
        retrieved_docids = s["reranked_docids"][:1]
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
        # split 
        doc_lm_splits = model.split_context_by_token_id(
            doc_input.input_ids, lm_gradcams_output)
        lm_score = doc_lm_splits[0][1]
        lm_text = tokenizer.batch_decode(doc_lm_splits[0][0], skip_special_tokens=True)
        lm_text2score = map_lm_score_to_text(lm_text, lm_score)
        
        # ranking
        doc_splits = split_doc(tokenizer.decode(doc_input.input_ids[0], skip_special_tokens=True))
        doc_splits_input = tokenizer(doc_splits,         
                              padding=True,
                              truncation=True,
                              max_length=doc_max_length,
                              return_tensors="pt")
        doc_splits_input.input_ids = doc_splits_input.input_ids.unsqueeze(0).to(args.device)
        doc_splits_input.attention_mask = doc_splits_input.attention_mask.unsqueeze(0).to(args.device)
        ranking_output = model(
            document_input_ids=doc_splits_input.input_ids, 
            document_attention_mask=doc_splits_input.attention_mask, 
            question_input_ids=ques_input.input_ids, 
            question_attention_mask=ques_input.attention_mask
        )
        ranking_score = ranking_output.logits[0]
        ranking_text2score = {t:s for t, s in zip(doc_splits, ranking_score) if t}
        ranking_score = [ranking_text2score[t] for t in doc_splits]
        lm_score = [lm_text2score[t] for t in doc_splits]
        # 
        compressed_context = compress_context_by_avg_rank(
            docs=doc_splits,
            ranking_score=ranking_score,
            lm_score=lm_score,
            keep_ratio=args.keep_ratio
        )
        # replace unk with \n
        s["compressed_context"] = "。".join(compressed_context)
        s["keep_ratio"] = args.keep_ratio
        s["max_new_tokens"] = args.max_new_tokens
        s["predicted_answer"] = tokenizer.decode(tokens_output[0])
        output_list.append(s)
    
    ds_with_compress = datasets.Dataset.from_list(output_list)
    ds_with_compress.save_to_disk(args.output_path)



if __name__ == "__main__":
    main()