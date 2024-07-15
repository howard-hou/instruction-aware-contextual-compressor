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
    parser.add_argument('--keep_ratio', type=float, default=0.5, help='keep ratio')
    parser.add_argument('--device', type=str, default='cuda', help='device name')
    return parser.parse_args()


def split_doc(doc, sep="。"):
    return [d for d in doc.split(sep) if d.strip()]


def compress_context_by_ranking(docs, logits, keep_ratio=0.5):
    # sort doc index by logits from high to low
    sorted_idx = sorted(range(len(docs)), key=lambda k: logits[k], reverse=True)
    # at least keep one context
    num_drop = max(int(len(sorted_idx) * keep_ratio), 1)
    # keep order of document
    sorted_idx = sorted(sorted_idx[:num_drop])
    compressed_docs = [docs[i] for i in sorted_idx]
    return compressed_docs
    
@torch.no_grad()
def main():
    args = parse_args()
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    ds = datasets.load_from_disk(args.dataset_path)
    print(ds)
    documents = json.load(open(args.document_path))
    docid2doc = {d["docid"]:d["document"] for d in documents}
    model.to(args.device)
    model.eval()
    output_list = []
    for s in tqdm(ds.select(range(100))):
        docid = s["reranked_docids"][0]
        doc = docid2doc[docid]
        docs = split_doc(doc)
        doc_input = tokenizer(docs,         
                              padding=True,
                              truncation=True,
                              max_length=doc_max_length,
                              return_tensors="pt")
        doc_input.input_ids = doc_input.input_ids.unsqueeze(0).to(args.device)
        doc_input.attention_mask = doc_input.attention_mask.unsqueeze(0).to(args.device)
        #print("doc shape", doc_input.input_ids.shape)
        ques_input = tokenizer(s["question"],
                               padding=True,
                               truncation=True,
                               max_length=ques_max_length,
                               return_tensors="pt")
        ques_input.input_ids = ques_input.input_ids.to(args.device)
        ques_input.attention_mask = ques_input.attention_mask.to(args.device)
        #print("ques shape", ques_input.input_ids.shape)
         
        ranking_output = model(
            document_input_ids=doc_input.input_ids, 
            document_attention_mask=doc_input.attention_mask, 
            question_input_ids=ques_input.input_ids, 
            question_attention_mask=ques_input.attention_mask
        )
        #
        compressed_context = compress_context_by_ranking(
            docs, ranking_output.logits[0], keep_ratio=args.keep_ratio
        )
        # replace unk with \n
        s["compressed_context"] = "。".join(compressed_context)
        s["keep_ratio"] = args.keep_ratio
        output_list.append(s)
    
    ds_with_compress = datasets.Dataset.from_list(output_list)
    ds_with_compress.save_to_disk(args.output_path)



if __name__ == "__main__":
    main()
