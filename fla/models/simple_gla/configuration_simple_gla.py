from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from fla.layers.attn import Attention
from fla.layers.simple_gla import SimpleGatedLinearAttention
from fla.models.simple_gla.configuration_simple_gla import SimpleGLAConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import RMSNorm
from fla.modules import GatedMLP as GLAMLP

logger = logging.get_logger(__name__)


# =========================================================
# Memory State
# =========================================================

@dataclass
class MemoryState:
    memory_tokens: Optional[torch.Tensor] = None


# =========================================================
# Segment Compressor
# =========================================================

class SegmentCompressor(nn.Module):
    """
    Compress a segment of tokens into one memory token
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, segment):
        # segment: (B, S, C)
        pooled = segment.mean(dim=1)
        return self.proj(pooled).unsqueeze(1)


# =========================================================
# Memory Reader
# =========================================================

class MemoryReader(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, memory):

        if memory is None:
            return hidden_states

        q = self.q(hidden_states)
        k = self.k(memory)
        v = self.v(memory)

        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.size(-1)), dim=-1)
        out = attn @ v

        return self.out(out)


# =========================================================
# Block
# =========================================================

class SimpleGLABlock(nn.Module):

    def __init__(self, config: SimpleGLAConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                qkv_bias=config.attn['qkv_bias'],
                window_size=config.attn['window_size'],
                rope_theta=config.attn['rope_theta'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx,
            )
        else:
            self.attn = SimpleGatedLinearAttention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
            )

        self.memory_reader = MemoryReader(config.hidden_size)

        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.mlp = GLAMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        hidden_states,
        memory_state: MemoryState = None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
    ):

        residual = hidden_states

        hidden_states = self.attn_norm(hidden_states)

        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # ===== Memory Read =====
        if memory_state is not None and memory_state.memory_tokens is not None:

            mem_out = self.memory_reader(hidden_states, memory_state.memory_tokens)

            hidden_states = hidden_states + mem_out

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.mlp_norm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, attentions, past_key_values


# =========================================================
# Model
# =========================================================

class SimpleGLAModel(PreTrainedModel):

    config_class = SimpleGLAConfig
    base_model_prefix = "model"

    def __init__(self, config):

        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [SimpleGLABlock(config, i) for i in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        # memory components
        self.segment_size = 32
        self.recent_window = 256

        self.compressor = SegmentCompressor(config.hidden_size)

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        memory_state: MemoryState = None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        B, T, C = hidden_states.shape

        # ==========================================
        # Sequence shortening
        # ==========================================

        if T > self.recent_window:

            old_tokens = hidden_states[:, :-self.recent_window]

            new_tokens = hidden_states[:, -self.recent_window:]

            segments = old_tokens.split(self.segment_size, dim=1)

            mem_list = []

            for seg in segments:

                mem = self.compressor(seg)

                mem_list.append(mem)

            mem_tokens = torch.cat(mem_list, dim=1)

            if memory_state is None:
                memory_state = MemoryState()

            if memory_state.memory_tokens is None:
                memory_state.memory_tokens = mem_tokens
            else:
                memory_state.memory_tokens = torch.cat(
                    [memory_state.memory_tokens, mem_tokens], dim=1
                )

            hidden_states = new_tokens

        # ==========================================
        # Backbone
        # ==========================================

        all_hidden_states = []

        for layer in self.layers:

            hidden_states, _, past_key_values = layer(
                hidden_states,
                memory_state=memory_state,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


# =========================================================
# Causal LM
# =========================================================

class SimpleGLAForCausalLM(PreTrainedModel, FLAGenerationMixin):

    config_class = SimpleGLAConfig

    def __init__(self, config):

        super().__init__(config)

        self.model = SimpleGLAModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        memory_state: MemoryState = None,
    ):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_state=memory_state,
        )

        hidden_states = outputs.last_hidden_state

        logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:

            shift_logits = logits[..., :-1, :].contiguous()

            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )