
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.simple_gla.configuration_simple_gla import SimpleGLAConfig
from fla.models.simple_gla.modeling_simple_gla import SimpleGLAForCausalLM, SimpleGLAModel

AutoConfig.register(SimpleGLAConfig.model_type, SimpleGLAConfig, exist_ok=True)
AutoModel.register(SimpleGLAConfig, SimpleGLAModel, exist_ok=True)
AutoModelForCausalLM.register(SimpleGLAConfig, SimpleGLAForCausalLM, exist_ok=True)

__all__ = ['SimpleGLAConfig', 'SimpleGLAForCausalLM', 'SimpleGLAModel']