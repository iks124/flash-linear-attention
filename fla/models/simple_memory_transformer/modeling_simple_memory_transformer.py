from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.models.simple_memory_transformer.configuration_simple_memory_transformer import SimpleMemoryTransformerConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as MemTransformerMLP
from fla.modules.l2warp import l2_warp
from fla.ops.gla import fused_recurrent_gla

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


@dataclass
class BaseModelOutputWithPastAndMemory(BaseModelOutputWithPast):
    memory_state: Optional[Any] = None


@dataclass
class CausalLMOutputWithPastAndMemory(CausalLMOutputWithPast):
    memory_state: Optional[Any] = None


class MemoryCache(Cache):
    """KV cache plus per-layer GLA memory state for the simple memory transformer."""

    def __init__(self, seen_tokens: int = 0, **kwargs: Any) -> None:
        super().__init__(seen_tokens=seen_tokens, **kwargs)
        self._raw_buffer: dict[int, torch.Tensor | None] = {}
        self._persistent_state: dict[int, torch.Tensor | None] = {}

    def _ensure_layer(self, layer_idx: int) -> None:
        if layer_idx not in self._raw_buffer:
            self._raw_buffer[layer_idx] = None
            self._persistent_state[layer_idx] = None

    def get_layer_memory(self, layer_idx: int) -> dict[str, Any]:
        self._ensure_layer(layer_idx)
        return {
            'raw_buffer': self._raw_buffer[layer_idx],
            'persistent_state': self._persistent_state[layer_idx],
        }

    def update_layer_memory(
        self,
        layer_idx: int,
        raw_buffer: torch.Tensor | None,
        persistent_state: torch.Tensor | None,
    ) -> None:
        self._ensure_layer(layer_idx)
        self._raw_buffer[layer_idx] = raw_buffer
        self._persistent_state[layer_idx] = persistent_state

    def detach_(self) -> 'MemoryCache':
        for layer in self.layers:
            if layer.state is not None and layer.state.get('attn_state') is not None:
                layer.state['attn_state'] = tuple(t.detach() for t in layer.state['attn_state'])
        for d in (self._raw_buffer, self._persistent_state):
            for k, t in d.items():
                if t is not None:
                    d[k] = t.detach()
        return self


class GLAMemoryCompressor(nn.Module):
    """GLA compressor that only learns K/V/gate projections and returns per-chunk states."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        chunk_size: int,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.gate_logit_normalizer = gate_logit_normalizer

        if self.key_dim % num_heads != 0:
            raise ValueError('key_dim must be divisible by num_heads')
        if self.value_dim % num_heads != 0:
            raise ValueError('value_dim must be divisible by num_heads')

        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim, bias=True),
        )

    def _project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.norm(hidden_states)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b t h d', h=self.num_heads)
        gk = rearrange(self.gk_proj(x), 'b t (h d) -> b t h d', h=self.num_heads)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        q = torch.zeros_like(k)
        return q, k, v, gk

    def _run_recurrent_chunk(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> torch.Tensor:
        q, k, v, gk = self._project(hidden_states)
        _, final_state = fused_recurrent_gla(
            q=q,
            k=k,
            v=v,
            gk=gk,
            initial_state=initial_state,
            output_final_state=True,
        )
        return final_state

    def state_history(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if hidden_states.numel() == 0:
            return None, initial_state
        if hidden_states.shape[1] % self.chunk_size != 0:
            raise ValueError(
                f'Compressor input length ({hidden_states.shape[1]}) must be divisible by chunk_size ({self.chunk_size}).'
            )

        state = initial_state
        states: list[torch.Tensor] = []
        for start in range(0, hidden_states.shape[1], self.chunk_size):
            chunk = hidden_states[:, start:start + self.chunk_size]
            state = self._run_recurrent_chunk(chunk, state)
            states.append(state)
        return torch.stack(states, dim=1), state


class GLAMemoryReader(nn.Module):
    """Reads from a single recurrent state or a history of per-chunk recurrent states."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        chunk_size: int,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        if self.key_dim % num_heads != 0:
            raise ValueError('key_dim must be divisible by num_heads')
        if self.value_dim % num_heads != 0:
            raise ValueError('value_dim must be divisible by num_heads')

        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _read_from_state(self, hidden_states: torch.Tensor, recurrent_state: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        q = rearrange(self.q_proj(x), 'b t (h d) -> b t h d', h=self.num_heads)
        q = F.normalize(q, dim=-1)
        o = torch.einsum('bthk,bhkv->bthv', q, recurrent_state)
        o = self.v_norm(o)
        o = rearrange(o, 'b t h v -> b t (h v)')
        o = self.o_proj(o)
        gate = torch.sigmoid(self.gate_proj(x))
        return gate * o

    def forward(self, hidden_states: torch.Tensor, recurrent_state: torch.Tensor) -> torch.Tensor:
        return self._read_from_state(hidden_states, recurrent_state)

    def read_group_history(
        self,
        hidden_states: torch.Tensor,
        state_history: torch.Tensor | None,
    ) -> torch.Tensor:
        if hidden_states.numel() == 0:
            return hidden_states

        output = torch.zeros_like(hidden_states)
        if state_history is None:
            return output

        group_count = (hidden_states.shape[1] + self.chunk_size - 1) // self.chunk_size
        for group_idx in range(group_count):
            start = group_idx * self.chunk_size
            end = min(start + self.chunk_size, hidden_states.shape[1])
            if group_idx == 0:
                continue
            state_idx = min(group_idx - 1, state_history.shape[1] - 1)
            output[:, start:end] = self._read_from_state(hidden_states[:, start:end], state_history[:, state_idx])
        return output


class SimpleMemoryTransformerBlock(GradientCheckpointingLayer):

    def __init__(self, config: SimpleMemoryTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enable_memory = config.use_memory and layer_idx in set(config.memory_layers)
        self.window_size = config.raw_buffer_max_size
        self.chunk_size = config.memory_chunk_size

        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.local_window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MemTransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

        if self.enable_memory:
            self.compressor = GLAMemoryCompressor(
                hidden_size=config.hidden_size,
                num_heads=config.gla_num_heads,
                chunk_size=config.memory_chunk_size,
                expand_k=config.gla_expand_k,
                expand_v=config.gla_expand_v,
                gate_low_rank_dim=config.gla_gate_low_rank_dim,
                gate_logit_normalizer=config.gla_gate_logit_normalizer,
                norm_eps=config.norm_eps,
            )
            self.memory_reader = GLAMemoryReader(
                hidden_size=config.hidden_size,
                num_heads=config.gla_num_heads,
                chunk_size=config.memory_chunk_size,
                expand_k=config.gla_expand_k,
                expand_v=config.gla_expand_v,
                norm_eps=config.norm_eps,
            )

    @staticmethod
    def _cat(old: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
        return new if old is None else torch.cat([old, new], dim=1)

    def _training_memory_read(
        self,
        block_input: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        if seq_len <= self.window_size:
            return hidden_states
        if seq_len != 2 * self.window_size:
            raise ValueError(
                f'Training with memory expects sequence length 2 * window_size ({2 * self.window_size}), got {seq_len}.'
            )
        if self.window_size % self.chunk_size != 0:
            raise ValueError('window_size must be divisible by memory_chunk_size for training memory reads.')

        front_inputs = block_input[:, :self.window_size]
        state_history, _ = self.compressor.state_history(front_inputs)

        back_states = hidden_states[:, self.window_size:]
        mem_delta = self.memory_reader.read_group_history(back_states, state_history)
        hidden_states = hidden_states.clone()
        hidden_states[:, self.window_size:] = back_states + mem_delta
        return hidden_states

    def _compress_inference_buffer(
        self,
        raw_buffer: torch.Tensor,
        persistent_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        buffer_len = raw_buffer.shape[1]
        if buffer_len < self.window_size + self.chunk_size:
            return raw_buffer, persistent_state, None

        compress_len = ((buffer_len - self.window_size) // self.chunk_size) * self.chunk_size
        if compress_len <= 0:
            return raw_buffer, persistent_state, None

        to_compress = raw_buffer[:, :compress_len]
        remaining = raw_buffer[:, compress_len:]
        state_history, final_state = self.compressor.state_history(to_compress, initial_state=persistent_state)
        return remaining, final_state, state_history

    def _memory_read_inference(
        self,
        hidden_states: torch.Tensor,
        state_history: torch.Tensor | None,
        persistent_state: torch.Tensor | None,
        prior_seen_tokens: int,
    ) -> torch.Tensor:
        if hidden_states.numel() == 0:
            return hidden_states

        seq_len = hidden_states.shape[1]
        output = hidden_states

        if prior_seen_tokens == 0:
            if seq_len <= self.window_size:
                return output
            back_states = output[:, self.window_size:]
            mem_delta = self.memory_reader.read_group_history(back_states, state_history)
            output = output.clone()
            output[:, self.window_size:] = back_states + mem_delta
            return output

        if persistent_state is None:
            return output

        positions = prior_seen_tokens + torch.arange(seq_len, device=hidden_states.device)
        read_mask = positions >= self.window_size
        if not read_mask.any():
            return output

        read_indices = torch.nonzero(read_mask, as_tuple=False).flatten()
        mem_delta = self.memory_reader(output[:, read_indices], persistent_state)
        output = output.clone()
        output[:, read_indices] = output[:, read_indices] + mem_delta
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple:
        block_input = hidden_states
        prior_seen_tokens = 0
        current_state_history = None
        persistent_state = None

        if self.enable_memory and isinstance(past_key_values, MemoryCache):
            prior_seen_tokens = past_key_values.get_seq_length(self.layer_idx)
            layer_memory = past_key_values.get_layer_memory(self.layer_idx)
            persistent_state = layer_memory['persistent_state']
            if not self.training:
                raw_buffer = self._cat(layer_memory['raw_buffer'], block_input.detach())
                raw_buffer, persistent_state, current_state_history = self._compress_inference_buffer(
                    raw_buffer,
                    persistent_state,
                )
                past_key_values.update_layer_memory(self.layer_idx, raw_buffer, persistent_state)

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        if self.enable_memory:
            if self.training:
                hidden_states = self._training_memory_read(block_input, hidden_states)
            else:
                hidden_states = self._memory_read_inference(
                    hidden_states,
                    state_history=current_state_history,
                    persistent_state=persistent_state,
                    prior_seen_tokens=prior_seen_tokens,
                )

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        return hidden_states, attentions, past_key_values


class SimpleMemoryTransformerPreTrainedModel(PreTrainedModel):
    config_class = SimpleMemoryTransformerConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['SimpleMemoryTransformerBlock']
    _supports_cache_class = True

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            for attr in ('o_proj', 'down_proj'):
                p = getattr(getattr(module, attr, None), 'weight', None)
                if p is not None:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class SimpleMemoryTransformerModel(SimpleMemoryTransformerPreTrainedModel):

    def __init__(self, config: SimpleMemoryTransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            SimpleMemoryTransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPastAndMemory:
        if output_attentions:
            warnings.warn(
                '`SimpleMemoryTransformerModel` does not support `output_attentions`; setting it to False.'
            )
            output_attentions = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('Specify either input_ids or inputs_embeds, not both.')
        if input_ids is None and inputs_embeds is None:
            raise ValueError('You must specify input_ids or inputs_embeds.')

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if self.training and self.config.use_memory:
            expected_len = 2 * self.config.raw_buffer_max_size
            if hidden_states.shape[1] != expected_len:
                raise ValueError(
                    f'SimpleMemoryTransformer training expects fixed sequence length {expected_len}, '
                    f'got {hidden_states.shape[1]}.'
                )

        if use_cache or self.config.use_memory:
            if past_key_values is not None and not isinstance(past_key_values, Cache):
                past_key_values = Cache.from_legacy_cache(past_key_values)
            elif past_key_values is None:
                past_key_values = MemoryCache() if self.config.use_memory else Cache()

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attentions, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attns,
                    past_key_values if isinstance(past_key_values, MemoryCache) else None,
                ] if v is not None
            )

        return BaseModelOutputWithPastAndMemory(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            memory_state=past_key_values if isinstance(past_key_values, MemoryCache) else None,
        )


class SimpleMemoryTransformerForCausalLM(SimpleMemoryTransformerPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: SimpleMemoryTransformerConfig):
        super().__init__(config)
        self.model = SimpleMemoryTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg('num_logits_to_keep', version='4.50', new_name='logits_to_keep')
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPastAndMemory:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        loss, logits = None, None

        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(
                hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:]
            )

        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion

            labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)),
                dim=1,
            )

            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndMemory(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            memory_state=outputs.memory_state,
        )
