from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.layers.simple_gla import SimpleGatedLinearAttention
from fla.models.simple_gla.configuration_simple_gla import SimpleGLAConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as GLAMLP
from fla.modules.l2warp import l2_warp

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


class CompressionCache(Cache):

    def __init__(self, seen_tokens: int = 0, **kwargs: Any) -> None:
        super().__init__(seen_tokens=seen_tokens, **kwargs)
        self.memory_tokens: dict[int, torch.Tensor | None] = {}
        self.memory_mask: dict[int, torch.Tensor | None] = {}
        self.segment_meta: dict[int, dict[str, int]] = {}
        self.raw_sink: dict[int, torch.Tensor | None] = {}
        self.compression_buffer: dict[int, torch.Tensor | None] = {}

    def _ensure_layer(self, layer_idx: int) -> None:
        if layer_idx not in self.memory_tokens:
            self.memory_tokens[layer_idx] = None
            self.memory_mask[layer_idx] = None
            self.segment_meta[layer_idx] = {'num_segments': 0}
            self.raw_sink[layer_idx] = None
            self.compression_buffer[layer_idx] = None

    def get_layer_state(self, layer_idx: int) -> dict[str, Any]:
        self._ensure_layer(layer_idx)
        return {
            'memory_tokens': self.memory_tokens[layer_idx],
            'memory_mask': self.memory_mask[layer_idx],
            'segment_meta': self.segment_meta[layer_idx],
            'raw_sink': self.raw_sink[layer_idx],
            'compression_buffer': self.compression_buffer[layer_idx],
        }

    def update_layer_state(
        self,
        layer_idx: int,
        memory_tokens: torch.Tensor | None,
        raw_sink: torch.Tensor | None,
        compression_buffer: torch.Tensor | None,
    ) -> None:
        self._ensure_layer(layer_idx)
        self.memory_tokens[layer_idx] = memory_tokens
        self.raw_sink[layer_idx] = raw_sink
        self.compression_buffer[layer_idx] = compression_buffer
        if memory_tokens is None:
            self.memory_mask[layer_idx] = None
        else:
            self.memory_mask[layer_idx] = torch.ones(memory_tokens.shape[:2], dtype=torch.bool, device=memory_tokens.device)


class SegmentScorer(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        inner = max(1, hidden_size // 2)
        self.norm = nn.RMSNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, 2)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.norm(hidden_states)
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        boundary_logits, importance_logits = torch.chunk(x, chunks=2, dim=-1)
        return boundary_logits.squeeze(-1), importance_logits.squeeze(-1)


class SegmentCompressor(nn.Module):

    def __init__(self, hidden_size: int, slots_per_segment: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.slots_per_segment = slots_per_segment
        self.latent_queries = nn.Parameter(torch.randn(slots_per_segment, hidden_size) * 0.02)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, segment_states: torch.Tensor, importance_logits: torch.Tensor | None = None) -> torch.Tensor:
        # segment_states: (B, Ls, C)
        bsz, seg_len, _ = segment_states.shape
        query = self.query_proj(self.latent_queries).unsqueeze(0).expand(bsz, -1, -1)  # (B, K, C)
        keys = self.key_proj(segment_states)
        values = self.value_proj(segment_states)

        scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self.hidden_size)
        if importance_logits is not None:
            scores = scores + importance_logits.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1)
        slots = torch.matmul(attn, values)
        return self.out_proj(slots)


class MemoryReader(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** -0.5

    def forward(self, hidden_states: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(hidden_states)
        keys = self.k_proj(memory_tokens)
        values = self.v_proj(memory_tokens)

        attn = torch.matmul(query, keys.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        return self.out_proj(torch.matmul(attn, values))


class MemoryAugmentedGLABlock(GradientCheckpointingLayer):

    def __init__(self, config: SimpleGLAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enable_memory = config.use_dual_path_memory and layer_idx in set(config.memory_layers)

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
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
                feature_map=config.feature_map,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                conv_bias=config.conv_bias,
                gate_fn=config.hidden_act,
                elementwise_affine=config.elementwise_affine,
                norm_eps=config.norm_eps,
                gate_logit_normalizer=config.gate_logit_normalizer,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
            )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = GLAMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

        if self.enable_memory:
            self.segment_scorer = SegmentScorer(config.hidden_size)
            self.segment_compressor = SegmentCompressor(config.hidden_size, config.memory_slots_per_segment)
            self.memory_reader = MemoryReader(config.hidden_size)
            self.memory_gate = nn.Linear(config.hidden_size, 1)

    def _append_tokens(self, old: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return torch.cat((old, new), dim=1)

    def _update_memory(
        self,
        hidden_states: torch.Tensor,
        memory_state: CompressionCache,
    ) -> CompressionCache:
        layer_state = memory_state.get_layer_state(self.layer_idx)
        raw_sink = self._append_tokens(layer_state['raw_sink'], hidden_states.detach())
        compression_buffer = layer_state['compression_buffer']
        memory_tokens = layer_state['memory_tokens']

        if raw_sink.shape[1] > self.config.working_memory_window:
            overflow_len = raw_sink.shape[1] - self.config.working_memory_window
            overflow = raw_sink[:, :overflow_len]
            compression_buffer = self._append_tokens(compression_buffer, overflow)
            raw_sink = raw_sink[:, -self.config.working_memory_window:]

        while compression_buffer is not None and compression_buffer.shape[1] >= self.config.compression_segment_length:
            segment = compression_buffer[:, :self.config.compression_segment_length]
            compression_buffer = compression_buffer[:, self.config.compression_segment_length:]
            _, importance_logits = self.segment_scorer(segment)
            slots = self.segment_compressor(segment, importance_logits=importance_logits)
            memory_tokens = self._append_tokens(memory_tokens, slots)
            if memory_tokens.shape[1] > self.config.max_compressed_slots:
                memory_tokens = memory_tokens[:, -self.config.max_compressed_slots:]
            memory_state.segment_meta[self.layer_idx]['num_segments'] += 1

        memory_state.update_layer_state(
            layer_idx=self.layer_idx,
            memory_tokens=memory_tokens,
            raw_sink=raw_sink,
            compression_buffer=compression_buffer,
        )
        return memory_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        memory_state: CompressionCache | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None, Cache | None, CompressionCache | None]:
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

        if self.enable_memory and memory_state is not None:
            layer_memory = memory_state.get_layer_state(self.layer_idx)['memory_tokens']
            if layer_memory is not None:
                mem_out = self.memory_reader(hidden_states, layer_memory)
                gate = torch.sigmoid(self.memory_gate(hidden_states))
                hidden_states = hidden_states + gate * mem_out

        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        if self.enable_memory and memory_state is not None:
            memory_state = self._update_memory(hidden_states=hidden_states, memory_state=memory_state)

        return hidden_states, attentions, past_key_values, memory_state


class SimpleGLAPreTrainedModel(PreTrainedModel):
    config_class = SimpleGLAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['MemoryAugmentedGLABlock']
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module, prenorm_residual_strategy: str | None = None, num_residuals_per_layer: int = 2):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            p = getattr(module.o_proj, 'weight', None) if hasattr(module, 'o_proj') else getattr(module.down_proj, 'weight', None) if hasattr(module, 'down_proj') else None
            if p is not None:
                if prenorm_residual_strategy == 'rescale':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(p)


class SimpleGLAModel(SimpleGLAPreTrainedModel):
    def __init__(self, config: SimpleGLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MemoryAugmentedGLABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
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
        memory_state: CompressionCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple | BaseModelOutputWithPastAndMemory:
        if output_attentions:
            warnings.warn("`SimpleGLAModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if use_cache:
            if past_key_values is not None and not isinstance(past_key_values, Cache):
                cache_cls = CompressionCache if self.config.use_dual_path_memory else Cache
                past_key_values = cache_cls.from_legacy_cache(past_key_values)
            elif past_key_values is None:
                past_key_values = CompressionCache() if self.config.use_dual_path_memory else Cache()

        if self.config.use_dual_path_memory and memory_state is None:
            memory_state = CompressionCache()

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attentions, past_key_values, memory_state = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                memory_state=memory_state,
                **kwargs,
            )
            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns, memory_state] if i is not None)

        return BaseModelOutputWithPastAndMemory(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            memory_state=memory_state,
        )


class SimpleGLAForCausalLM(SimpleGLAPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = SimpleGLAModel(config)
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

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        memory_state: CompressionCache | None = None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        logits_to_keep=0,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            memory_state=memory_state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = outputs[0]
        loss, logits = None, None

        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])

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
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)

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
