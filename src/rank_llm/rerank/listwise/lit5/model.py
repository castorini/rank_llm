import copy
import types

import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack

from .modeling_t5 import (
    T5ForConditionalGeneration as T5ConditionalGenerationCrossAttentionScore,
)
from .modeling_t5 import T5Stack as T5StackCrossAttentionScore


class FiDStack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        self._n_passages = None

    def reset_n_passages(self, n_passages: int):
        self._n_passages = n_passages

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
        cache_position=None,
    ):
        if not self.is_decoder:
            input_ids = input_ids.view(input_ids.size(0) * self._n_passages, -1)
            attention_mask = attention_mask.view(
                attention_mask.size(0) * self._n_passages, -1
            )

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not self.is_decoder:
            bsz = input_ids.size(0) // self._n_passages
            if not return_dict:
                last_hidden_states = output[0]
                last_hidden_state = last_hidden_states.view(
                    bsz, -1, last_hidden_states.size(-1)
                )
                output = tuple(
                    last_hidden_state,
                    *output[1:],
                )
            else:
                last_hidden_state = output.last_hidden_state
                output.last_hidden_state = last_hidden_state.view(
                    bsz, -1, last_hidden_state.size(-1)
                )

        return output


class FiD(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    ANSWER_EOS_TOKEN = 1

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = FiDStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.use_cache = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = FiDStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def reset_n_passages(self, n_passages: int):
        self.encoder.reset_n_passages(n_passages)
        self.decoder.reset_n_passages(n_passages)

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.normalized_score_storage = None

    @torch.no_grad()
    def get_crossattention_scores(
        self, n_passages, mask, ids, mask_query=None, output_sequence_lengths=[]
    ):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        norms = []
        for mod in self.decoder.block:
            norms.append(mod.layer[1].EncDecAttention.normalized_score_storage)
        norms = torch.stack(norms)

        output = {}
        self.aggregate_value(
            norms,
            mask,
            n_passages,
            ids,
            mask_query,
            output,
            prefix="norms",
            output_sequence_lengths=output_sequence_lengths,
        )
        return output

    def aggregate_value(
        self,
        scores,
        mask,
        n_passages,
        ids,
        mask_query=None,
        output={},
        prefix="",
        output_sequence_lengths=[],
    ):
        n_layers, bsz, n_tokens, total_tokens = scores.size()

        ids = ids.view(bsz, n_passages, -1)
        scores = scores.view(n_layers, bsz, n_tokens, n_passages, -1)
        mask = mask.view(bsz, n_passages, -1)
        scores = scores.masked_fill(~mask[None, :, None], 0.0)

        scores = scores.sum(dim=[0])

        scores_woquery = None
        # Compute scores based on scores without query
        if not mask_query is None:
            output[f"{prefix}woquery"] = self.get_woquery_score(
                scores,
                mask_query,
                mask,
                n_layers,
                output_sequence_lengths=output_sequence_lengths,
            )

        return output

    def get_woquery_score(
        self, scores, mask_query, mask, n_layers, output_sequence_lengths
    ):
        if scores.size(-1) > mask_query.size(-1):
            zero_padding = torch.zeros(
                [mask_query.size(0), scores.size(-1) - mask_query.size(-1)],
                device=mask_query.device,
                dtype=torch.bool,
            )
            mask_query = torch.cat([mask_query, zero_padding], dim=-1)
        mask_query = mask * (~mask_query[:, None])
        scores_woquery = scores.masked_fill(~mask_query[:, None], 0.0)

        ntokens_woquery = 256 * n_layers

        # zero out scores after EOS token. This is needed when batching results in sequences with different lengths.
        for i in range(len(scores_woquery)):
            scores_woquery[i, output_sequence_lengths[i] :, :, :] = 0

        scores_woquery = scores_woquery.sum(dim=[1, 3])
        return scores_woquery / ntokens_woquery

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            xattn = mod.layer[1].EncDecAttention
            xattn.forward = types.MethodType(cross_attention_forward, xattn)

    def create_crossattention_storage(self):
        for mod in self.decoder.block:
            xattn = mod.layer[1].EncDecAttention
            xattn.normalized_score_storage = None


class FiDStackCrossAttentionScore(T5StackCrossAttentionScore):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)

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
        n_passages: int = None,
    ):
        if not self.is_decoder:
            input_ids = input_ids.view(input_ids.size(0) * n_passages, -1)
            attention_mask = attention_mask.view(
                attention_mask.size(0) * n_passages, -1
            )

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not self.is_decoder:
            bsz = input_ids.size(0) // n_passages
            if not return_dict:
                last_hidden_states = output[0]
                last_hidden_state = last_hidden_states.view(
                    bsz, -1, last_hidden_states.size(-1)
                )
                output = tuple(
                    last_hidden_state,
                    *output[1:],
                )
            else:
                last_hidden_state = output.last_hidden_state
                output.last_hidden_state = last_hidden_state.view(
                    bsz, -1, last_hidden_state.size(-1)
                )

        return output


class FiDCrossAttentionScore(T5ConditionalGenerationCrossAttentionScore):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    ANSWER_EOS_TOKEN = 1

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = FiDStackCrossAttentionScore(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.use_cache = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = FiDStackCrossAttentionScore(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.normalized_score_storage = None

    @torch.no_grad()
    def get_crossattention_scores(
        self, n_passages, mask, ids, mask_query=None, output_sequence_lengths=[]
    ):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        norms = []
        for mod in self.decoder.block:
            norms.append(mod.layer[1].EncDecAttention.normalized_score_storage)
        norms = torch.stack(norms)

        output = {}
        self.aggregate_value(
            norms,
            mask,
            n_passages,
            ids,
            mask_query,
            output,
            prefix="norms",
            output_sequence_lengths=output_sequence_lengths,
        )
        return output

    def aggregate_value(
        self,
        scores,
        mask,
        n_passages,
        ids,
        mask_query=None,
        output={},
        prefix="",
        output_sequence_lengths=[],
    ):
        n_layers, bsz, n_tokens, total_tokens = scores.size()

        ids = ids.view(bsz, n_passages, -1)
        scores = scores.view(n_layers, bsz, n_tokens, n_passages, -1)
        mask = mask.view(bsz, n_passages, -1)
        scores = scores.masked_fill(~mask[None, :, None], 0.0)

        scores = scores.sum(dim=[0])

        scores_woquery = None
        # Compute scores based on scores without query
        if not mask_query is None:
            output[f"{prefix}woquery"] = self.get_woquery_score(
                scores,
                mask_query,
                mask,
                n_layers,
                output_sequence_lengths=output_sequence_lengths,
            )

        return output

    def get_woquery_score(
        self, scores, mask_query, mask, n_layers, output_sequence_lengths
    ):
        if scores.size(-1) > mask_query.size(-1):
            zero_padding = torch.zeros(
                [mask_query.size(0), scores.size(-1) - mask_query.size(-1)],
                device=mask_query.device,
                dtype=torch.bool,
            )
            mask_query = torch.cat([mask_query, zero_padding], dim=-1)
        mask_query = mask * (~mask_query[:, None])
        scores_woquery = scores.masked_fill(~mask_query[:, None], 0.0)

        ntokens_woquery = 256 * n_layers

        # zero out scores after EOS token. This is needed when batching results in sequences with different lengths.
        for i in range(len(scores_woquery)):
            scores_woquery[i, output_sequence_lengths[i] :, :, :] = 0

        scores_woquery = scores_woquery.sum(dim=[1, 3])
        return scores_woquery / ntokens_woquery

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            xattn = mod.layer[1].EncDecAttention
            xattn.forward = types.MethodType(cross_attention_forward, xattn)

    def create_crossattention_storage(self):
        for mod in self.decoder.block:
            xattn = mod.layer[1].EncDecAttention
            xattn.normalized_score_storage = None


def cross_attention_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
    cache_position=None,
):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)

    batch_size, seq_length = hidden_states.shape[:2]
    real_seq_length = seq_length

    if past_key_value is not None:
        assert (
            len(past_key_value) == 2
        ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
        real_seq_length += (
            past_key_value[0].shape[2] if query_length is None else query_length
        )

    key_length = (
        real_seq_length if key_value_states is None else key_value_states.shape[1]
    )

    def shape(states):
        """projection"""
        return states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get query states
    query_states = shape(
        self.q(hidden_states)
    )  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states,
        self.k,
        key_value_states,
        past_key_value[0] if past_key_value is not None else None,
    )
    value_states = project(
        hidden_states,
        self.v,
        key_value_states,
        past_key_value[1] if past_key_value is not None else None,
    )

    # compute scores
    scores = torch.matmul(
        query_states, key_states.transpose(3, 2)
    )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length),
                device=scores.device,
                dtype=scores.dtype,
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        if mask is not None:
            position_bias = (
                position_bias + mask
            )  # (batch_size, n_heads, seq_length, key_length)

    scores += position_bias

    attn_weights = nn.functional.softmax(scores.float(), dim=-1)  # .type_as(scores)

    if hasattr(self, "normalized_score_storage"):
        with torch.no_grad():
            self.normalized_score_storage = (
                (torch.norm(value_states.float(), dim=-1)[:, :, None] * attn_weights)
                .detach()
                .mean(dim=1)
            )

    attn_weights = nn.functional.dropout(
        attn_weights.type_as(scores), p=self.dropout, training=self.training
    )

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(
        torch.matmul(attn_weights, value_states)
    )  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    present_key_value_state = (
        (key_states, value_states) if (self.is_decoder and use_cache) else None
    )
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs
