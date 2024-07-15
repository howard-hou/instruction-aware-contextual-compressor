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
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda', help='device name')
    return parser.parse_args()
    
@torch.no_grad()
def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    ds = datasets.load_from_disk(args.dataset_path)
    documents = json.load(open(args.document_path))
    docid2doc = {d["docid"]:d["document"] for d in documents}
    model.to(args.device)
    model.eval()
    output_list = []
    for s in tqdm(ds):
        retrieved_docids = s["retrieved_docids"][:args.topk]
        docs = [docid2doc[docid] for docid in retrieved_docids]
        doc_input = tokenizer(docs,         
                              padding="max_length",
                              truncation=True,
                              max_length=doc_max_length,
                              return_tensors="pt")
        doc_input.input_ids = doc_input.input_ids.unsqueeze(0).to(args.device)
        doc_input.attention_mask = doc_input.attention_mask.unsqueeze(0).to(args.device)
        ques_input = tokenizer(s["question"],
                               padding="max_length",
                               truncation=True,
                               max_length=ques_max_length,
                               return_tensors="pt")
        ques_input.input_ids = ques_input.input_ids.to(args.device)
        ques_input.attention_mask = ques_input.attention_mask.to(args.device)
        output = model(document_input_ids=doc_input.input_ids, 
                       document_attention_mask=doc_input.attention_mask, 
                       question_input_ids=ques_input.input_ids, 
                       question_attention_mask=ques_input.attention_mask)
        # sort docids by score
        reranked_scores = torch.sort(output.logits, dim=1, descending=True).values.tolist()[0]
        reranked_index = torch.argsort(output.logits, dim=1, descending=True).tolist()[0]
        reranked_docids = [retrieved_docids[i] for i in reranked_index]
        s["reranked_docids"] = reranked_docids
        s["reranked_scores"] = reranked_scores
        output_list.append(s)
    
    ds_with_rerank = datasets.Dataset.from_list(output_list)
    ds_with_rerank.save_to_disk(args.dataset_path + "_rerank")



if __name__ == "__main__":
    main()
