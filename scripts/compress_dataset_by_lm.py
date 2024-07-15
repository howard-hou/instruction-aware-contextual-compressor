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
    

def main():
    args = parse_args()
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    ds = datasets.load_from_disk(args.dataset_path)
    print(ds)
    documents = json.load(open(args.document_path))
    docid2doc = {d["docid"]:d["document"] for d in documents}
    model.to(args.device)
    model.eval()
    output_list = []
    for s in tqdm(ds.select(range(100))):
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
        # 
        compressed_context_ids = model.compress_context_by_activation(
            doc_input.input_ids, 
            lm_gradcams_output, 
            keep_ratio=args.keep_ratio)[0]
        compressed_context = tokenizer.batch_decode(compressed_context_ids)
        # replace unk with \n
        compressed_context = [c.replace("<unk>", "\n") for c in compressed_context]
        s["compressed_context"] = "".join(compressed_context)
        s["keep_ratio"] = args.keep_ratio
        s["max_new_tokens"] = args.max_new_tokens
        s["predicted_answer"] = tokenizer.decode(tokens_output[0])
        output_list.append(s)
    
    ds_with_compress = datasets.Dataset.from_list(output_list)
    ds_with_compress.save_to_disk(args.output_path)



if __name__ == "__main__":
    main()
