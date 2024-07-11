""" modified PyTorch UMT5 model. add save attention weights function so that we can compute grad-cam."""

import copy
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput,
)
from transformers import PreTrainedModel, UMT5Config
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "UMT5Config"
_CHECKPOINT_FOR_DOC = "google/umt5-small"


# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->UMT5
class UMT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the UMT5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # UMT5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->UMT5
class UMT5DenseActDense(nn.Module):
    def __init__(self, config: UMT5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5->UMT5
class UMT5DenseGatedActDense(nn.Module):
    def __init__(self, config: UMT5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5LayerFF with T5->UMT5
class UMT5LayerFF(nn.Module):
    def __init__(self, config: UMT5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = UMT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = UMT5DenseActDense(config)

        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class UMT5Attention(nn.Module):
    """
    T5's attention using relative_attention_bias.
    """

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

        # save attention weights
        self.save_attention = False
        self.attn_gradients = None
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def _shape(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.n_heads, self.key_value_proj_dim)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def _relative_position_bucket(self, relative_position):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        num_buckets = self.relative_attention_num_buckets
        max_distance = self.relative_attention_max_distance
        if not self.is_decoder:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        log_ratio = torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact)
        log_ratio = log_ratio * (num_buckets - max_exact)
        relative_position_if_large = max_exact + log_ratio.to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
    ):
        is_cross_attention = encoder_hidden_states is not None
        batch_size, seq_length = hidden_states.shape[:2]

        # use encoder_hidden_states if cross attention
        current_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # checking that the `sequence_length` of the `past_key_value` is the same as the he provided
        # `encoder_hidden_states` to support prefix tuning
        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            key_states = self._shape(self.k(current_states))
            value_states = self._shape(self.v(current_states))
            if past_key_value is not None and not is_cross_attention:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        query_states = self._shape(self.q(hidden_states))
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        # compute positional bias
        if self.has_relative_attention_bias:
            query_length = seq_length
            if past_key_value is not None:
                query_length += past_key_value[0].shape[2]
            position_bias = self.compute_bias(query_length, key_states.size(2), device=attention_scores.device)
        else:
            position_bias = torch.zeros(
                (1, self.n_heads, seq_length, key_states.size(2)),
                device=attention_scores.device,
                dtype=attention_scores.dtype,
                requires_grad=self.training,
            )
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]
        if attention_mask is not None:
            position_bias = position_bias + attention_mask  # (batch_size, n_heads, seq_length, key_length)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        attention_scores += position_bias
        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).type_as(attention_scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # save attention weights
        if self.save_attention:
            self.save_attention_map(attn_weights)
            attn_weights.register_hook(self.save_attn_gradients)    

        #  attn_output = torch.bmm(attn_probs, value_states) ?
        context_states = torch.matmul(attn_weights, value_states)
        # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim) ?
        context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.o(context_states)
        return attn_output, attn_weights, past_key_value


class UMT5LayerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.SelfAttention = UMT5Attention(config, has_relative_attention_bias=True)
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        past_key_value=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class UMT5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = UMT5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        layer_head_mask=None,
        past_key_value=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class UMT5Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(UMT5LayerSelfAttention(config))
        if self.is_decoder:
            self.layer.append(UMT5LayerCrossAttention(config))

        self.layer.append(UMT5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        hidden_states, self_attn_weights, present_key_value = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
        )

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            max_dtype = torch.finfo(hidden_states.dtype).max
            clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.layer[1](
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
            )
            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                max_dtype = torch.finfo(hidden_states.dtype).max
                clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            present_key_value += cross_attn_present_key_value

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            max_dtype = torch.finfo(hidden_states.dtype).max
            clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (
            hidden_states,
            present_key_value,
        )

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


# Copied from transformers.models.t5.modeling_t5.T5ClassificationHead with T5->UMT5
class UMT5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: UMT5Config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class UMT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = UMT5Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UMT5Block"]
    _keep_in_fp32_modules = ["wo"]

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, UMT5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(
            module,
            (
                UMT5Model,
            ),
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "qa_outputs"):
                module.qa_outputs.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                module.qa_outputs.bias.data.zero_()
        elif isinstance(module, UMT5ClassificationHead):
            module.dense.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.dense, "bias") and module.dense.bias is not None:
                module.dense.bias.data.zero_()
            module.out_proj.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.out_proj, "bias") and module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, UMT5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, UMT5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, UMT5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (UMT5Attention, UMT5Stack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In UMT5 it is usually set to the pad_token_id."
                "See UMT5 docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class UMT5Stack(UMT5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.block = nn.ModuleList([UMT5Block(config) for i in range(config.num_layers)])
        self.final_layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.is_decoder else None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

            if use_cache:
                present_key_value_states += (layer_outputs[1],)

            if output_attentions:
                all_attentions += (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions += (layer_outputs[3],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


UMT5_START_DOCSTRING = r"""

    The UMT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`UMT5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

UMT5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. UMT5 is a model with relative position embeddings so
            you should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [UMT5 Training](./umt5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            UMT5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [UMT5
            Training](./umt5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

UMT5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. UMT5 is a model with relative position embeddings so
            you should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [UMT5 Training](./umt5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare UMT5 Model transformer outputting raw hidden-states without any specific head on top.",
    UMT5_START_DOCSTRING,
)
class UMT5Model(UMT5PreTrainedModel):
    r"""
    Examples:

    ```python
    >>> from transformers import UMT5Model, AutoTokenizer

    >>> model = UMT5Model.from_pretrained("google/umt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
    >>> noisy_text = "UN Offizier sagt, dass weiter <extra_id_0> werden muss in Syrien."
    >>> label = "<extra_id_0> verhandelt"
    >>> inputs = tokenizer(inputs, return_tensors="pt")
    >>> labels = tokenizer(label=label, return_tensors="pt")

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```"""
    model_type = "uumt5"
    config_class = UMT5Config
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UMT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_input_embeddings
    def get_input_embeddings(self):
        return self.shared

    # Copied from transformers.models.t5.modeling_t5.T5Model.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_encoder
    def get_encoder(self):
        return self.encoder

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_decoder
    def get_decoder(self):
        return self.decoder

    # Copied from transformers.models.t5.modeling_t5.T5Model._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, UMT5Model

        >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
        >>> model = UMT5Model.from_pretrained("google/umt5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for UMT5Model.
        >>> # This is not needed for torch's UMT5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


# start of ranking prompter code


from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from .configuration_rankingprompter import RankingPrompterConfig


@dataclass
class RankingPrompterForPreTrainingOutput:
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


@dataclass
class RankingPrompterOutput:
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None
    loss_lm: torch.FloatTensor = None
    loss_ranking: torch.FloatTensor = None



class RankingPrompterForPreTraining(UMT5Model):
    config_class = RankingPrompterConfig

    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config):
        # encoder, decoder and shared are from UMT5Model
        super().__init__(config)

        # add ranking head
        self.ranking_head = nn.Linear(config.d_model, 1)

        # Initialize weights and apply final processing
        self.post_init()

        # ctx for mixed precision training
        self.ctx = nullcontext()

    def enable_amp_ctx(self, device_type="cuda", dtype=torch.bfloat16):
        self.ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

    def disable_amp_ctx(self):
        self.ctx = nullcontext()

    def forward(
        self,
        document_input_ids: Optional[torch.LongTensor] = None,
        document_attention_mask: Optional[torch.FloatTensor] = None,
        question_input_ids: Optional[torch.LongTensor] = None,
        question_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], RankingPrompterForPreTrainingOutput]:
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
        #
        document_input_ids = document_input_ids.view(-1, doc_seq_len)
        # to [batch_size * num_doc, doc_seq_len]
        document_attention_mask = document_attention_mask.view(-1, doc_seq_len)

        # Convert encoder inputs in embeddings if needed
        with self.ctx:
            encoder_outputs = self.encoder(
                input_ids=document_input_ids,
                attention_mask=document_attention_mask,
                return_dict=return_dict,
            )

        document_embeds = encoder_outputs[0]

        # repeat question inputs for each document
        # question_input_ids: [batch_size, question_seq_len]
        question_seq_len = question_input_ids.shape[1]
        question_input_ids_expand = (
            question_input_ids.unsqueeze(1)
            .expand(-1, num_doc, -1)
            .reshape(-1, question_seq_len)
        )  # [batch_size * num_doc, question_seq_len]
        question_attention_mask_expand = (
            question_attention_mask.unsqueeze(1)
            .expand(-1, num_doc, -1)
            .reshape(-1, question_seq_len)
        )  # [batch_size * num_doc, question_seq_len]

        # Decode
        with self.ctx:
            decoder_outputs = self.decoder(
                input_ids=question_input_ids_expand,
                attention_mask=question_attention_mask_expand,
                past_key_values=past_key_values,
                encoder_hidden_states=document_embeds,
                encoder_attention_mask=document_attention_mask,
                use_cache=use_cache,
                return_dict=return_dict,
            )
        # [batch_size * num_doc, soft_prompt_len + question_seq_len, hidden_size]
        sequence_output = decoder_outputs[0]
        # [batch_size * num_doc, soft_prompt_len, hidden_size]
        question_seq_len = sequence_output.size(1)
        # [batch_size, num_doc, soft_prompt_len, hidden_size]
        soft_prompt_output = sequence_output.view(
            batch_size, num_doc, question_seq_len, -1
        )
        question_attention_mask_expand = question_attention_mask_expand.view(
            batch_size, num_doc, question_seq_len
        )
        # apply question attention mask
        soft_prompt_output = soft_prompt_output * question_attention_mask_expand.unsqueeze(-1)

        # [batch_size, num_doc, self.num_soft_prompt_tokens, hidden_size] -> [batch_size, num_doc]
        ranking_logits = self.ranking_head(soft_prompt_output.mean(dim=2)).view(batch_size, num_doc)

        # rank loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss = loss_fct(ranking_logits, labels)

        if not return_dict:
            output = (ranking_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return RankingPrompterForPreTrainingOutput(
            loss=loss,
            logits=ranking_logits
        )


class RankingPrompter(UMT5Model):
    config_class = RankingPrompterConfig

    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config):
        # encoder, decoder and shared are from UMT5Model
        super().__init__(config)

        # add ranking head
        self.ranking_head = nn.Linear(config.d_model, 1)

        # Initialize weights and apply final processing
        self.post_init()

        # ctx for mixed precision training
        self.ctx = nullcontext()

    def enable_amp_ctx(self, device_type="cuda", dtype=torch.bfloat16):
        self.ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

    def disable_amp_ctx(self):
        self.ctx = nullcontext()

    def encode_document(self, document_input_ids, document_attention_mask):
        # input shape: [batch_size * num_doc, doc_seq_len]
        # Convert encoder inputs in embeddings if needed
        with self.ctx:
            encoder_outputs = self.encoder(
                input_ids=document_input_ids,
                attention_mask=document_attention_mask,
                return_dict=False,
            )
        return encoder_outputs
    
    def decode_answer(
            self, 
            question_input_ids, 
            question_attention_mask, 
            document_embeds, 
            document_attention_mask,
            answer_input_ids=None,
            answer_attention_mask=None
        ):
        if answer_input_ids is not None and answer_attention_mask is not None:
            # append answer input ids to question input ids
            question_input_ids = torch.cat([question_input_ids, answer_input_ids], dim=1)
            question_attention_mask = torch.cat([question_attention_mask, answer_attention_mask], dim=1)

        answer_outputs = self.decoder(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
            encoder_hidden_states=document_embeds,
            encoder_attention_mask=document_attention_mask,
            return_dict=True,
        )
        return answer_outputs

    def forward(
        self,
        document_input_ids: Optional[torch.LongTensor] = None,
        document_attention_mask: Optional[torch.FloatTensor] = None,
        question_input_ids: Optional[torch.LongTensor] = None,
        question_attention_mask: Optional[torch.BoolTensor] = None,
        answer_input_ids: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        answer_attention_mask: Optional[Tuple[Tuple[torch.Tensor]]] = None,
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
        if len(document_input_ids.shape) == 2:
            # make [batch_size, doc_seq_len] -> [batch_size, 1, doc_seq_len]
            document_input_ids = document_input_ids.unsqueeze(1)
            document_attention_mask = document_attention_mask.unsqueeze(1)
        # document_input_ids: [batch_size, num_doc, doc_seq_len]
        batch_size, num_doc, doc_seq_len = document_input_ids.shape
        document_input_ids = document_input_ids.view(-1, doc_seq_len)
        # to [batch_size * num_doc, doc_seq_len]
        document_attention_mask = document_attention_mask.view(-1, doc_seq_len)

        encoder_outputs = self.encode_document(document_input_ids, document_attention_mask)
        document_embeds = encoder_outputs[0]

        # repeat question inputs for each document
        # question_input_ids: [batch_size, question_seq_len]
        question_seq_len = question_input_ids.shape[1]
        question_input_ids_expand = (
            question_input_ids.unsqueeze(1)
            .expand(-1, num_doc, -1)
            .reshape(-1, question_seq_len)
        )  # [batch_size * num_doc, question_seq_len]
        question_attention_mask_expand = (
            question_attention_mask.unsqueeze(1)
            .expand(-1, num_doc, -1)
            .reshape(-1, question_seq_len)
        )  # [batch_size * num_doc, question_seq_len]

        # Decode
        with self.ctx:
            decoder_outputs = self.decoder(
                input_ids=question_input_ids_expand,
                attention_mask=question_attention_mask_expand,
                encoder_hidden_states=document_embeds,
                encoder_attention_mask=document_attention_mask,
                use_cache=False,
                return_dict=True,
            )
        # [batch_size * num_doc, soft_prompt_len + question_seq_len, hidden_size]
        sequence_output = decoder_outputs.last_hidden_state
        # [batch_size * num_doc, soft_prompt_len, hidden_size]
        question_seq_len = sequence_output.size(1)
        # [batch_size, num_doc, soft_prompt_len, hidden_size]
        soft_prompt_output = sequence_output.view(
            batch_size, num_doc, question_seq_len, -1
        )
        question_attention_mask_expand = question_attention_mask_expand.view(
            batch_size, num_doc, question_seq_len
        )
        # apply question attention mask
        soft_prompt_output = soft_prompt_output * question_attention_mask_expand.unsqueeze(-1)
        # [batch_size, num_doc, self.num_soft_prompt_tokens, hidden_size] -> [batch_size, num_doc]
        ranking_logits = self.ranking_head(soft_prompt_output.mean(dim=2)).view(batch_size, num_doc)

        # rank loss
        loss_ranking = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss_ranking = loss_fct(ranking_logits, labels)
        # append bos token id to question input ids
        question_input_ids = torch.cat(
            [question_input_ids, torch.ones_like(question_input_ids[:, :1]).fill_(self.config.decoder_start_token_id)], dim=1)
        question_attention_mask = torch.cat(
            [question_attention_mask, torch.ones_like(question_attention_mask[:, :1])], dim=1)
        # only take the first document for answer generation training
        answer_outputs = self.decode_answer(question_input_ids, 
                                            question_attention_mask, 
                                            document_embeds[::num_doc], 
                                            document_attention_mask[::num_doc],
                                            answer_input_ids,
                                            answer_attention_mask)
        # lm loss
        loss_lm = None
        if answer_input_ids is not None:
            # fill in question_input_ids with -100
            question_input_mask = torch.zeros_like(question_input_ids).fill_(-100)
            # mask padding token in answer_input_ids with -100
            answer_input_ids = answer_input_ids.masked_fill(answer_input_ids == self.config.pad_token_id, -100)
            # [batch_size, question_seq_len + answer_seq_len, hidden_size]
            lm_labels = torch.cat([question_input_mask, answer_input_ids], dim=1)[:, 1:].contiguous()
            lm_logits = (answer_outputs.last_hidden_state @ self.decoder.embed_tokens.weight.t())[:, :-1, :].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss_lm = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

        if loss_ranking is not None and loss_lm is not None:
            loss = loss_ranking + loss_lm
        elif loss_ranking is not None:
            loss = loss_ranking
        elif loss_lm is not None:
            loss = loss_lm
        else:
            loss = None
            
        if not return_dict:
            output = (ranking_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return RankingPrompterOutput(
            loss=loss,
            logits=ranking_logits,
            lm_logits=lm_logits,
            loss_lm=loss_lm,
            loss_ranking=loss_ranking,
        )
    
    def generate_answer(        
        self,
        document_input_ids: Optional[torch.LongTensor] = None,
        document_attention_mask: Optional[torch.FloatTensor] = None,
        question_input_ids: Optional[torch.LongTensor] = None,
        question_attention_mask: Optional[torch.BoolTensor] = None
        ):
        if len(document_input_ids.shape) == 2:
            # make [batch_size, doc_seq_len] -> [batch_size, 1, doc_seq_len]
            document_input_ids = document_input_ids.unsqueeze(1)
            document_attention_mask = document_attention_mask.unsqueeze(1)
        # document_input_ids: [batch_size, num_doc, doc_seq_len]
        batch_size, num_doc, doc_seq_len = document_input_ids.shape
        document_input_ids = document_input_ids.view(-1, doc_seq_len)
        # to [batch_size * num_doc, doc_seq_len]
        document_attention_mask = document_attention_mask.view(-1, doc_seq_len)
        document_embeds = self.encode_document(document_input_ids, document_attention_mask)[0]
        # append bos token id to question input ids
        question_input_ids = torch.cat(
            [question_input_ids, torch.ones_like(question_input_ids[:, :1]).fill_(self.config.decoder_start_token_id)], dim=1)
        question_attention_mask = torch.cat(
            [question_attention_mask, torch.ones_like(question_attention_mask[:, :1])], dim=1)
        answer_outputs = self.decode_answer(question_input_ids, 
                                            question_attention_mask, 
                                            document_embeds[::num_doc], 
                                            document_attention_mask[:num_doc])
        lm_logits = answer_outputs.last_hidden_state @ self.decoder.embed_tokens.weight.t()
        return lm_logits[:, -1:, :]
    

    def compute_grad_cam(self, 
                         document_input_ids, 
                         document_attention_mask, 
                         question_input_ids, 
                         question_attention_mask,
                         block_num=-1):
        # 设置模型为evaluation模式, 开启保存attention map
        self.eval()
        attention_layer = self.decoder.block[block_num].layer[-2].EncDecAttention
        attention_layer.save_attention = True

        # 正向传播以获取特征图
        encoder_outputs = self.encode_document(document_input_ids, document_attention_mask)
        document_embeds = encoder_outputs[0]

        # 正向传播解码器以获取Grad-CAM
        decoder_outputs = self.decoder(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
            encoder_hidden_states=document_embeds,
            encoder_attention_mask=document_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        # get grads
        soft_prompt_output = decoder_outputs.last_hidden_state * question_attention_mask.unsqueeze(-1)
        ranking_logits = self.ranking_head(soft_prompt_output.mean(dim=1)).view(-1)
        loss = ranking_logits.sum()
        self.zero_grad()
        loss.backward()

        # compute grad cam
        with torch.no_grad():
            # grads and cams [bsz, num_head, ques_len, doc_len]
            grads = attention_layer.get_attn_gradients()
            cams = attention_layer.get_attention_map()
            gradcams = cams * grads
            # average over heads -> [bsz, ques_len, doc_len]
            gradcams = gradcams.mean(dim=1)
            # apply relu
            # gradcams = gradcams.relu()
        return gradcams



