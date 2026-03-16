import pytest
import torch

from fla.models.simple_gla.configuration_simple_gla import SimpleGLAConfig
from fla.models.simple_gla.modeling_simple_gla import (
    CompressionCache,
    MemoryReader,
    SegmentCompressor,
    SegmentScorer,
    SimpleGLAForCausalLM,
)


TEST_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_tiny_config(**overrides) -> SimpleGLAConfig:
    base = dict(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=4,
        num_kv_heads=4,
        hidden_ratio=2,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        use_dual_path_memory=True,
        memory_layers=[0],
        working_memory_window=4,
        compression_segment_length=2,
        memory_slots_per_segment=2,
        max_compressed_slots=6,
    )
    base.update(overrides)
    return SimpleGLAConfig(**base)


def _make_model(config: SimpleGLAConfig) -> SimpleGLAForCausalLM:
    model = SimpleGLAForCausalLM(config)
    return model.to(TEST_DEVICE).eval()


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SimpleGLA forward path currently requires CUDA kernels")
def test_simple_gla_initialization_wires_memory_modules():
    config = _make_tiny_config(num_hidden_layers=3, memory_layers=[1])
    model = _make_model(config)

    layer0 = model.model.layers[0]
    layer1 = model.model.layers[1]
    layer2 = model.model.layers[2]

    assert not layer0.enable_memory
    assert layer1.enable_memory
    assert not layer2.enable_memory

    assert hasattr(layer1, 'segment_scorer')
    assert hasattr(layer1, 'segment_compressor')
    assert hasattr(layer1, 'memory_reader')
    assert hasattr(layer1, 'memory_gate')


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SimpleGLA forward path currently requires CUDA kernels")
def test_simple_gla_forward_builds_cache_and_memory_state():
    torch.manual_seed(0)
    config = _make_tiny_config()
    model = _make_model(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 6), device=TEST_DEVICE)
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)

    assert out.logits.shape == (2, 6, config.vocab_size)
    assert isinstance(out.past_key_values, CompressionCache)
    assert isinstance(out.memory_state, CompressionCache)

    layer_state = out.memory_state.get_layer_state(0)
    assert layer_state['raw_sink'] is not None
    assert layer_state['raw_sink'].shape[1] <= config.working_memory_window
    assert layer_state['memory_mask'] is None or layer_state['memory_mask'].dtype == torch.bool


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SimpleGLA forward path currently requires CUDA kernels")
def test_simple_gla_segment_compression_and_metadata_update():
    torch.manual_seed(1)
    config = _make_tiny_config(
        num_hidden_layers=1,
        memory_layers=[0],
        working_memory_window=3,
        compression_segment_length=2,
        memory_slots_per_segment=2,
        max_compressed_slots=4,
    )
    model = _make_model(config)

    input_ids = torch.randint(0, config.vocab_size, (1, 12), device=TEST_DEVICE)
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)

    state = out.memory_state.get_layer_state(0)
    assert out.memory_state.segment_meta[0]['num_segments'] > 0
    assert state['memory_tokens'] is not None
    assert state['memory_tokens'].shape[1] <= config.max_compressed_slots
    assert state['memory_mask'].shape == state['memory_tokens'].shape[:2]
    assert state['raw_sink'].shape[1] <= config.working_memory_window


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SimpleGLA generation currently requires CUDA kernels")
def test_simple_gla_generate_smoke_for_quick_showcase():
    torch.manual_seed(2)
    config = _make_tiny_config(num_hidden_layers=1, memory_layers=[0], max_position_embeddings=64)
    model = _make_model(config)

    prompt = torch.randint(0, config.vocab_size, (1, 5), device=TEST_DEVICE)
    generated = model.generate(
        input_ids=prompt,
        max_new_tokens=3,
        do_sample=False,
        use_cache=True,
    )

    assert generated.shape == (1, 8)


def test_simple_gla_memory_building_blocks_shape_contract_showcase():
    torch.manual_seed(3)
    hidden_size = 32
    batch, seq_len, slots = 2, 6, 3

    scorer = SegmentScorer(hidden_size).to(TEST_DEVICE).eval()
    compressor = SegmentCompressor(hidden_size, slots_per_segment=slots).to(TEST_DEVICE).eval()
    reader = MemoryReader(hidden_size).to(TEST_DEVICE).eval()

    segment_states = torch.randn(batch, seq_len, hidden_size, device=TEST_DEVICE)
    query_states = torch.randn(batch, 4, hidden_size, device=TEST_DEVICE)

    boundary_logits, importance_logits = scorer(segment_states)
    compressed_memory = compressor(segment_states, importance_logits=importance_logits)
    memory_readout = reader(query_states, compressed_memory)

    showcase = {
        'segment_states': tuple(segment_states.shape),
        'boundary_logits': tuple(boundary_logits.shape),
        'importance_logits': tuple(importance_logits.shape),
        'compressed_memory': tuple(compressed_memory.shape),
        'memory_readout': tuple(memory_readout.shape),
    }

    assert showcase == {
        'segment_states': (batch, seq_len, hidden_size),
        'boundary_logits': (batch, seq_len),
        'importance_logits': (batch, seq_len),
        'compressed_memory': (batch, slots, hidden_size),
        'memory_readout': (batch, 4, hidden_size),
    }
