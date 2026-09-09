# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.engine import core as engine_core

from vllm_ascend.core.slot_apc import validate_slot_apc_config
from vllm_ascend.utils import refresh_block_size, vllm_version_is


class SlotStartupConfig(SimpleNamespace):
    def __post_init__(self):
        # Exercise the block-size path called by NPUPlatform on config reentry.
        refresh_block_size(self)
        validate_slot_apc_config(self)


@pytest.fixture
def slot_config(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_SLOT_APC", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "0")
    monkeypatch.setenv("VLLM_VERSION", "0.25.1")
    vllm_version_is.cache_clear()
    hf_config = SimpleNamespace(model_type="deepseek_v4")
    yield SlotStartupConfig(
        cache_config=SimpleNamespace(block_size=128, num_gpu_blocks=None, enable_prefix_caching=True),
        model_config=SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config, max_model_len=1024),
        scheduler_config=SimpleNamespace(async_scheduling=False, enable_chunked_prefill=True),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(compilation_time=0, encoder_compilation_time=0),
        kv_transfer_config=None,
        speculative_config=None,
        validate_block_size=Mock(),
    )
    vllm_version_is.cache_clear()


@pytest.mark.parametrize("block_size", [8, 16, 32, 64, 256])
def test_initial_block_size_still_rejected(slot_config, block_size):
    slot_config.cache_config.block_size = block_size
    with pytest.raises(ValueError, match=rf"block_size=128 \(actual={block_size}\)"):
        validate_slot_apc_config(slot_config)
    with pytest.raises(ValueError, match="block_size=128"):
        slot_config.__post_init__()


@pytest.mark.parametrize("num_gpu_blocks", [0, 64])
@pytest.mark.parametrize("runtime_block_size", [8, 16, 32, 128])
def test_revalidation_preserves_initialized_block_size(slot_config, num_gpu_blocks, runtime_block_size):
    slot_config.__post_init__()
    slot_config.cache_config.num_gpu_blocks = num_gpu_blocks
    slot_config.cache_config.block_size = runtime_block_size
    for _ in range(3):
        slot_config.__post_init__()
        assert slot_config.cache_config.block_size == runtime_block_size


@pytest.mark.parametrize("client_handshake_address", [None, "client"])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_handshake_revalidates_after_kv_initialization(
    slot_config, monkeypatch, client_handshake_address, async_scheduling
):
    slot_config.scheduler_config.async_scheduling = async_scheduling
    slot_config.__post_init__()
    groups = [SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=b)) for b in (128, 128, 8, 32)]
    kv_config = SimpleNamespace(num_blocks=64, kv_cache_groups=groups)
    executor = Mock()
    executor.get_kv_cache_specs.return_value = [{str(i): g.kv_cache_spec for i, g in enumerate(groups)}]
    executor.determine_available_memory.return_value = [1024]
    proc = SimpleNamespace(
        model_executor=executor,
        _perform_handshake=Mock(side_effect=lambda *args: nullcontext(SimpleNamespace(inputs=[], outputs=[]))),
    )
    monkeypatch.setattr(engine_core.zmq, "Context", Mock())
    monkeypatch.setattr(engine_core, "register_all_kvcache_specs", Mock())
    monkeypatch.setattr(engine_core, "get_kv_cache_configs", Mock(return_value=[kv_config]))
    monkeypatch.setattr(engine_core, "generate_scheduler_kv_cache_config", Mock(return_value=kv_config))
    monkeypatch.setattr(engine_core, "get_kv_cache_capacity", Mock(return_value=(1024, 1.0)))

    with engine_core.EngineCoreProc._perform_handshakes(
        proc, "frontend", b"0", True, slot_config, client_handshake_address
    ):
        # Execute the upstream initialization method, including its min(group
        # block sizes) assignment, with cache planning and device work mocked.
        engine_core.EngineCore._initialize_kv_caches(proc, slot_config)
        assert slot_config.cache_config.num_gpu_blocks == 64
        assert slot_config.cache_config.block_size == 8

    assert slot_config.cache_config.block_size == 8
    executor.initialize_from_config.assert_called_once_with([kv_config])
    assert proc._perform_handshake.call_count == (1 if client_handshake_address is None else 2)


@pytest.mark.parametrize(
    "component,field,value,reason",
    [
        ("cache_config", "enable_prefix_caching", False, "enable_prefix_caching=True"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "DCP=PCP=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "DCP=PCP=1"),
        (None, "kv_transfer_config", object(), "no KV connector"),
        (None, "speculative_config", object(), "no speculative decoding/MTP"),
    ],
)
def test_initialized_config_still_checks_other_constraints(slot_config, component, field, value, reason):
    slot_config.cache_config.num_gpu_blocks = 64
    slot_config.cache_config.block_size = 8
    setattr(getattr(slot_config, component) if component else slot_config, field, value)
    with pytest.raises(ValueError, match=reason):
        slot_config.__post_init__()
    assert slot_config.cache_config.block_size == 8


def test_disabling_slot_apc_preserves_existing_refresh_behavior(slot_config, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_SLOT_APC", "0")
    slot_config.cache_config.num_gpu_blocks = 64
    slot_config.cache_config.block_size = 8
    refresh_block_size(slot_config)
    assert slot_config.cache_config.block_size == 32


def test_phase_one_core_hooks_are_not_enough_for_async(slot_config, monkeypatch):
    slot_config.scheduler_config.async_scheduling = True
    monkeypatch.delattr(KVCacheCoordinator, "on_step_scheduled")
    with pytest.raises(ValueError, match="lifecycle hooks"):
        validate_slot_apc_config(slot_config)


def test_clearing_runtime_state_reenables_initial_block_size_validation(slot_config):
    slot_config.cache_config.num_gpu_blocks = 64
    slot_config.cache_config.block_size = 8
    slot_config.__post_init__()
    slot_config.cache_config.num_gpu_blocks = None
    with pytest.raises(ValueError, match="block_size=128"):
        slot_config.__post_init__()
