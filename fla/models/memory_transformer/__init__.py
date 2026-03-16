from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.memory_transformer.configuration_memory_transformer import MemoryTransformerConfig
from fla.models.memory_transformer.modeling_memory_transformer import (
    MemoryTransformerForCausalLM,
    MemoryTransformerModel,
)

AutoConfig.register(MemoryTransformerConfig.model_type, MemoryTransformerConfig, exist_ok=True)
AutoModel.register(MemoryTransformerConfig, MemoryTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(MemoryTransformerConfig, MemoryTransformerForCausalLM, exist_ok=True)

__all__ = [
    'MemoryTransformerConfig',
    'MemoryTransformerForCausalLM',
    'MemoryTransformerModel',
]
