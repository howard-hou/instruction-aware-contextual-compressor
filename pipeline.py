from transformers import AutoConfig, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from contextlib import nullcontext
from model.modeling_rankingprompter import Ranker
from datasets import load_dataset, Dataset
import hashlib
import numpy as np

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

class Pipeline():
    def __init__(self, 
                 documents: list[str],
                 model_name_or_path="google/umt5-small", 
                 retriever_name_or_path="moka-ai/m3e-base", 
                 question_max_length = 32,
                 doc_max_length = 512,
                 device="cuda", 
                 enable_amp_ctx=True) -> None:
        #  build dataset from list of doc
        self.device = device
        self.retriever = SentenceTransformer(retriever_name_or_path)
        self.dataset = self.build_dataset_from_documents(documents)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, 
                                               trust_remote_code=True).to(device)
        self.doc_max_length = doc_max_length
        self.question_max_length = question_max_length
        self.enable_amp_ctx = enable_amp_ctx
        if enable_amp_ctx:
            device_type = 'cuda' if 'cuda' in device else 'cpu'
            self.model.enable_amp_ctx(device_type=device_type,
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    def build_dataset_from_documents(self, documents):
        if isinstance(documents[0], str):
            documents = [{"document": doc} for doc in documents]
        dataset = Dataset.from_list(documents).map(generate_docid)
        dataset = dataset.map(
            lambda example: {'doc_embedding': self.retriever.encode(example["document"])}, 
            batched=True)
        dataset.add_faiss_index(column='doc_embedding')
        return dataset
    
    def tokenize_and_rerank(self, question, docs):
        '''
        questions: list of str
        docs: list of list of str
        '''
        # if input is str, convert to list
        if isinstance(question, str):
            question = [question]
        
        question = self.tokenizer(question, return_tensors="pt", 
                               padding="max_length",
                               truncation=True,
                               max_length=self.question_max_length)
        docs = self.tokenizer(docs, return_tensors="pt", 
                              padding="max_length", 
                              truncation=True, 
                              max_length=self.doc_max_length)
        document_input_ids = docs.input_ids.unsqueeze(0)
        document_attention_mask = docs.attention_mask.unsqueeze(0) 
        output = self.rerank(document_input_ids, document_attention_mask,
                             question.input_ids, question.attention_mask)
        return output.logits
        
    def rerank(self, document_input_ids, document_attention_mask,
               question_input_ids, question_attention_mask):
        with torch.no_grad():
            return self.model(
                document_input_ids=document_input_ids.to(self.device),
                document_attention_mask=document_attention_mask.to(self.device),
                question_input_ids=question_input_ids.to(self.device),
                question_attention_mask=question_attention_mask.to(self.device),
            )

    def query(self, question, topk=20):
        ques_embedding = self.retriever.encode(question)
        ques_embedding = np.array(ques_embedding, dtype=np.float32)
        scores, retrieved_examples = self.dataset.get_nearest_examples('doc_embedding', ques_embedding, k=topk)
        docs = retrieved_examples["document"]
        reranking_scores = self.tokenize_and_rerank(question, docs)
        max_index = torch.argmax(reranking_scores)
        return docs[max_index]