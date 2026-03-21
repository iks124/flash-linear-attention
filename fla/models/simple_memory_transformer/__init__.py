from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.simple_memory_transformer.configuration_simple_memory_transformer import SimpleMemoryTransformerConfig
from fla.models.simple_memory_transformer.modeling_simple_memory_transformer import (
    SimpleMemoryTransformerForCausalLM,
    SimpleMemoryTransformerModel,
)

AutoConfig.register(SimpleMemoryTransformerConfig.model_type, SimpleMemoryTransformerConfig, exist_ok=True)
AutoModel.register(SimpleMemoryTransformerConfig, SimpleMemoryTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(SimpleMemoryTransformerConfig, SimpleMemoryTransformerForCausalLM, exist_ok=True)

__all__ = [
    'SimpleMemoryTransformerConfig',
    'SimpleMemoryTransformerForCausalLM',
    'SimpleMemoryTransformerModel',
]
