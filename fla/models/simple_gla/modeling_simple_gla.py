
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.layers.simple_gla import SimpleGatedLinearAttention  # 新引入的层
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

class SimpleGLABlock(GradientCheckpointingLayer):

    def __init__(self, config: SimpleGLAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

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
                use_output_gate=config.use_output_gate,
                gate_fn=config.hidden_act,
                elementwise_affine=config.elementwise_affine,
                norm_eps=config.norm_eps,
                clamp_min=config.clamp_min,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
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
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        return (hidden_states, attentions, past_key_values)

class SimpleGLAPreTrainedModel(PreTrainedModel):
    config_class = SimpleGLAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['SimpleGLABlock']
    _supports_cache_class = True
    
    # ...existing init_weights methods (similar to GLAPreTrainedModel)...
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
        self.layers = nn.ModuleList([SimpleGLABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()
    # ...existing forward pass identical to GLAModel...
    def get_input_embeddings(self): return self.embeddings
    def set_input_embeddings(self, value): self.embeddings = value

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        # Implementation mirrors GLAModel forward exactly.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None: inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if use_cache and not isinstance(past_key_values, Cache): past_key_values = Cache.from_legacy_cache(past_key_values)
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states: all_hidden_states += (hidden_states,)
            hidden_states, attentions, past_key_values = layer(hidden_states, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, **kwargs)
            if output_attentions: all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states: all_hidden_states += (hidden_states,)

        if not return_dict: return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values, hidden_states=all_hidden_states, attentions=all_attns)

class SimpleGLAForCausalLM(SimpleGLAPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = SimpleGLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()
    # ...existing generation + forward matching GLAForCausalLM...
    def get_input_embeddings(self): return self.model.embeddings
    def set_input_embeddings(self, value): self.model.embeddings = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings
    def set_decoder(self, decoder): self.model = decoder
    def get_decoder(self): return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, past_key_values=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, logits_to_keep=0, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        hidden_states = outputs[0]
        loss, logits = None, None
        
        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy: criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy: criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else: criterion = nn.CrossEntropyLoss()
            else: criterion = self.criterion
            
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            
            if self.config.fuse_linear_cross_entropy: loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)