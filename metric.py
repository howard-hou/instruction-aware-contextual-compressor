import torch
import numpy as np
from typing import Dict, Sequence, Tuple, Union
import jieba
import rouge
import rouge_chinese
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torchmetrics.retrieval import RetrievalRecall, RetrievalHitRate, RetrievalMAP, RetrievalMRR, RetrievalNormalizedDCG
import re


def custom_sort_key(s):
    k, v = s
    # 使用正则表达式从字符串中提取数字部分，如果没有数字则默认为0
    numeric_part = int(re.search(r'\d+', k).group()) if re.search(r'\d+', k) else 0
    return k[0], numeric_part

class RetrievalMetrics:
    def __init__(self, topk_list=None) -> None:
        self.topk_list = topk_list if topk_list is not None else [1, 5, 10]
        self.indexes = []
        self.predictions = []
        self.targets = []
        self.index = 0
    
    def update(self, predictions: Sequence[int], targets: Sequence[bool]) -> None:
        for pred, target in zip(predictions, targets):
            self.indexes.extend([self.index] * len(pred))
            self.predictions.extend(pred)
            self.targets.extend(target)
            self.index += 1

    def compute(self) -> Dict[str, float]:
        predictions = torch.tensor(self.predictions).view(-1)
        targets = torch.tensor(self.targets).view(-1)
        indexes = torch.tensor(self.indexes).view(-1)
        score_dict = {}
        for topk in self.topk_list:
            score_dict[f"Recall@{topk}"] = RetrievalRecall(top_k=topk)(predictions, targets, indexes=indexes)
            score_dict[f"HitRate@{topk}"] = RetrievalHitRate(top_k=topk)(predictions, targets, indexes=indexes)
            score_dict[f"MAP@{topk}"] = RetrievalMAP(top_k=topk)(predictions, targets, indexes=indexes)
            score_dict[f"MRR"] = RetrievalMRR()(predictions, targets, indexes=indexes)
            score_dict[f"NDCG@{topk}"] = RetrievalNormalizedDCG(top_k=topk)(predictions, targets, indexes=indexes)
        return {k:v.item() for k, v in sorted(score_dict.items(), key=custom_sort_key)}



class GenerationMetrics:
    def __init__(self, tokenizer, is_chinese=True, ignore_index=-100):
        self.tokenizer = tokenizer
        self.is_chinese = is_chinese
        self.rouge = rouge_chinese.Rouge() if is_chinese else rouge.Rouge()
        self.ignore_index = ignore_index
        self.decoded_preds = []
        self.decoded_labels = []

    def update(self, preds, labels) -> None:
        preds = np.where(preds != self.ignore_index, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != self.ignore_index, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        self.decoded_preds.extend(decoded_preds)
        self.decoded_labels.extend(decoded_labels)

    def compute(self) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        score_dict = {"accuracy": [], "rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        for pred, label in zip(self.decoded_preds, self.decoded_labels):
            hypothesis = " ".join(jieba.lcut(pred, HMM=False)) if self.is_chinese else pred
            reference = " ".join(jieba.lcut(label, HMM=False)) if self.is_chinese else label

            if len(hypothesis.split()) == 0 or len(reference.split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                result = self.rouge.get_scores(hypothesis, reference)[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))
            score_dict["accuracy"].append(float(len(label) != 0 and pred[:len(label)] == label))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}