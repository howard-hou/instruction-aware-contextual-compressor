import copy
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import UMT5Model, UMT5PreTrainedModel


@dataclass
class RankingPrompterOutput:
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    soft_prompts: torch.FloatTensor = None


class RankingPrompter(UMT5PreTrainedModel):
    r"""
    Examples:

    ```python
    >>> from transformers import UMT5ForConditionalGeneration, AutoTokenizer

    >>> model = UMT5ForConditionalGeneration.from_pretrained("google/umt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, text_target=summary, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> loss = outputs.loss
    ```"""

    model_type = "umt5"
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "lm_head.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        model = UMT5Model.from_pretrained(config._name_or_path)
        self.encoder = model.encoder

        self.decoder = model.decoder

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.rank_head = nn.Linear(config.d_model, 2)

        # create soft prompt token embeddings
        self.soft_prompt_embeds = nn.Parameter(
            torch.zeros(1, config.num_soft_prompt_tokens, config.d_model)
        )
        self.soft_prompt_embeds.data.normal_(mean=0.0, std=1e-4)
        self.num_soft_prompt_tokens = config.num_soft_prompt_tokens

        # project soft prompt embeddings to large language model dimension
        self.llm_proj = nn.Linear(config.d_model, config.llm_dim)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        document_input_ids: Optional[torch.LongTensor] = None,
        document_attention_mask: Optional[torch.FloatTensor] = None,
        question_input_ids: Optional[torch.LongTensor] = None,
        question_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], RankingPrompterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # document_input_ids: [batch_size, num_doc, doc_seq_len]
        batch_size, num_doc, doc_seq_len = document_input_ids.shape
        device = document_input_ids.device
        #
        document_input_ids = document_input_ids.view(-1, doc_seq_len)
        # to [batch_size * num_doc, doc_seq_len]
        document_attention_mask = document_attention_mask.view(-1, doc_seq_len)

        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=document_input_ids,
            attention_mask=document_attention_mask,
            return_dict=return_dict,
        )

        document_embeds = encoder_outputs[0]
        soft_prompt_embeds = self.soft_prompt_embeds.expand(
            batch_size * num_doc, -1, -1
        )
        soft_prompt_attention_mask = torch.ones(
            soft_prompt_embeds.size(0), self.num_soft_prompt_tokens, device=device
        )

        # repeat question inputs for each document
        # question_input_ids: [batch_size, question_seq_len]
        question_seq_len = question_input_ids.shape[1]
        question_input_ids = (
            question_input_ids.unsqueeze(1)
            .expand(-1, num_doc, -1)
            .reshape(-1, question_seq_len)
        )  # [batch_size * num_doc, question_seq_len]
        question_attention_mask = (
            question_attention_mask.unsqueeze(1)
            .expand(-1, num_doc, -1)
            .reshape(-1, question_seq_len)
        )  # [batch_size * num_doc, question_seq_len]
        question_embeds = self.shared(question_input_ids)
        decoder_inputs_embeds = torch.cat([soft_prompt_embeds, question_embeds], dim=1)
        decoder_attention_mask = torch.cat(
            [soft_prompt_attention_mask, question_attention_mask], dim=1
        )
        # ranking labels, the first document is the positive one
        rank_labels = torch.zeros(batch_size * num_doc, dtype=torch.long, device=device)
        rank_labels[::num_doc] = 1

        # Decode
        # decoder will concat prompt and question embeddings inside
        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=document_embeds,
            encoder_attention_mask=document_attention_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        # [batch_size * num_doc, soft_prompt_len + question_seq_len, hidden_size]
        sequence_output = decoder_outputs[0]
        # [batch_size * num_doc, soft_prompt_len, hidden_size]
        soft_prompt_output = sequence_output[:, : self.num_soft_prompt_tokens, :]
        # [batch_size, num_doc, soft_prompt_len, hidden_size]
        soft_prompt_output = soft_prompt_output.view(
            batch_size, num_doc, self.num_soft_prompt_tokens, -1
        )

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        # [batch_size, num_doc, self.num_soft_prompt_tokens, hidden_size] -> [batch_size, num_doc, hidden_size]
        rank_logits = self.rank_head(soft_prompt_output).mean(dim=2)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        # rank loss
        loss = loss_fct(rank_logits.view(-1, rank_logits.size(-1)), rank_labels)

        rank_logits = rank_logits[:, :, 1]
        # Find the attention weights for the soft prompt tokens
        doc_attentions = torch.softmax(rank_logits, dim=1)
        # Use the attention to get the soft prompt embeddings
        # [batch_size, soft_prompt_len, hidden_size]
        soft_prompts = torch.einsum("bn,bndh->bdh", doc_attentions, soft_prompt_output)
        soft_prompts = self.llm_proj(
            soft_prompts
        )  # [batch_size, soft_prompt_len, llm_hidden_size]

        if not return_dict:
            output = (rank_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return RankingPrompterOutput(
            loss=loss,
            logits=rank_logits,
            soft_prompts=soft_prompts,
        )
