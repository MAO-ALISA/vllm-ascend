# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.block_pool import AscendDSV4BlockPool
from vllm_ascend.core.single_type_kv_cache_manager import (
    AscendDSV4SlidingWindowManager,
)
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
    _is_deepseek_v4_kv_cache_spec,
    get_kv_cache_coordinator,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_resolve_kv_cache_block_sizes,
)
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager


def _make_hybrid_kv_cache_config(
    full_block_size: int = 16,
    mamba_block_size: int = 16,
) -> KVCacheConfig:
    full_spec = FullAttentionSpec(
        block_size=full_block_size,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=mamba_block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=full_spec.page_size_bytes * 10, shared_by=["attn"]),
            KVCacheTensor(size=mamba_spec.page_size_bytes * 10, shared_by=["mamba"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=full_spec),
            KVCacheGroupSpec(layer_names=["mamba"], kv_cache_spec=mamba_spec),
        ],
    )


def _make_deepseek_v4_kv_cache_config() -> KVCacheConfig:
    c4_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        model_version="deepseek_v4",
    )
    c128_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=128,
        model_version="deepseek_v4",
    )
    c4_group_spec = UniformTypeKVCacheSpecs.from_specs({"c4_attn": c4_spec})
    c128_group_spec = UniformTypeKVCacheSpecs.from_specs({"c128_attn": c128_spec})
    assert c4_group_spec is not None
    assert c128_group_spec is not None
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=c4_spec.page_size_bytes * 10, shared_by=["c4_attn"]),
            KVCacheTensor(size=c128_spec.page_size_bytes * 10, shared_by=["c128_attn"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["c4_attn"], kv_cache_spec=c4_group_spec),
            KVCacheGroupSpec(layer_names=["c128_attn"], kv_cache_spec=c128_group_spec),
        ],
    )


def _make_deepseek_v4_apc_config(block_size: int) -> KVCacheConfig:
    c4_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        model_version="deepseek_v4",
    )
    c128_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=128,
        model_version="deepseek_v4",
    )
    swa_spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float16,
        sliding_window=128,
        compress_ratio=1,
        model_version="deepseek_v4",
    )
    state_spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float16,
        sliding_window=128,
        compress_ratio=1,
        model_version=None,
    )
    group_specs = [
        UniformTypeKVCacheSpecs.from_specs({"c4_attn": c4_spec}),
        UniformTypeKVCacheSpecs.from_specs({"c128_attn": c128_spec}),
        UniformTypeKVCacheSpecs.from_specs({"swa_attn": swa_spec}),
        UniformTypeKVCacheSpecs.from_specs({"state_attn": state_spec}),
    ]
    assert all(spec is not None for spec in group_specs)
    return KVCacheConfig(
        num_blocks=300,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["c4_attn"], kv_cache_spec=group_specs[0]),
            KVCacheGroupSpec(layer_names=["c128_attn"], kv_cache_spec=group_specs[1]),
            KVCacheGroupSpec(layer_names=["swa_attn"], kv_cache_spec=group_specs[2]),
            KVCacheGroupSpec(layer_names=["state_attn"], kv_cache_spec=group_specs[3]),
        ],
    )


def _make_vllm_config(
    *,
    enable_prefix_caching: bool,
    dcp: int,
    pcp: int,
    block_size: int = 16,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
    )


def _make_coordinator_for_effective_block_size(
    *,
    dcp_world_size: int,
    pcp_world_size: int,
    enable_caching: bool,
) -> AscendHybridKVCacheCoordinator:
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.dcp_world_size = dcp_world_size
    coordinator.pcp_world_size = pcp_world_size
    coordinator.enable_caching = enable_caching
    return coordinator


@pytest.mark.parametrize(
    ("enable_prefix_caching", "expected_hash_block_size"),
    [
        pytest.param(False, math.lcm(16, 32) * 2 * 2, id="cp-without-prefix-caching"),
        pytest.param(True, math.gcd(16, 32), id="cp-with-prefix-caching"),
    ],
)
def test_resolve_kv_cache_block_sizes_with_cp_hybrid_groups(
    enable_prefix_caching: bool,
    expected_hash_block_size: int,
) -> None:
    kv_cache_config = _make_hybrid_kv_cache_config(full_block_size=16, mamba_block_size=32)
    vllm_config = _make_vllm_config(
        enable_prefix_caching=enable_prefix_caching,
        dcp=2,
        pcp=2,
    )

    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        kv_cache_config,
        vllm_config,
    )

    expected_scheduler_block_size = math.lcm(16, 32) * 2 * 2
    assert scheduler_block_size == expected_scheduler_block_size
    assert hash_block_size == expected_hash_block_size


@pytest.mark.parametrize(
    ("spec_factory", "dcp", "pcp", "enable_caching", "expected"),
    [
        pytest.param(
            lambda: FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=64,
                dtype=torch.float16,
            ),
            2,
            2,
            True,
            64,
            id="full-attention-scales-with-cp",
        ),
        pytest.param(
            lambda: MambaSpec(
                block_size=16,
                shapes=((1,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="none",
            ),
            2,
            2,
            True,
            16,
            id="mamba-keeps-physical-block-size-with-prefix-caching",
        ),
        pytest.param(
            lambda: FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=64,
                dtype=torch.float16,
            ),
            1,
            1,
            True,
            16,
            id="full-attention-no-cp",
        ),
    ],
)
def test_get_effective_block_size(
    spec_factory,
    dcp: int,
    pcp: int,
    enable_caching: bool,
    expected: int,
) -> None:
    coordinator = _make_coordinator_for_effective_block_size(
        dcp_world_size=dcp,
        pcp_world_size=pcp,
        enable_caching=enable_caching,
    )

    assert coordinator._get_effective_block_size(spec_factory()) == expected


def test_get_kv_cache_coordinator_delegates_single_group(monkeypatch) -> None:
    sentinel = object()
    kv_cache_config = _make_hybrid_kv_cache_config(full_block_size=16, mamba_block_size=16)
    single_group_config = KVCacheConfig(
        num_blocks=kv_cache_config.num_blocks,
        kv_cache_tensors=kv_cache_config.kv_cache_tensors[:1],
        kv_cache_groups=kv_cache_config.kv_cache_groups[:1],
    )

    def _fake_orig(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator._orig_get_kv_cache_coordinator",
        _fake_orig,
    )

    coordinator = get_kv_cache_coordinator(
        single_group_config,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=16,
    )

    assert coordinator is sentinel


def test_get_kv_cache_coordinator_delegates_hybrid_without_caching(monkeypatch) -> None:
    sentinel = object()
    kv_cache_config = _make_hybrid_kv_cache_config(full_block_size=16, mamba_block_size=16)

    def _fake_orig(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator._orig_get_kv_cache_coordinator",
        _fake_orig,
    )

    coordinator = get_kv_cache_coordinator(
        kv_cache_config,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=2,
        pcp_world_size=2,
        hash_block_size=16,
    )

    assert coordinator is sentinel


def test_get_kv_cache_coordinator_uses_ascend_for_deepseek_v4(monkeypatch) -> None:
    sentinel = object()
    kv_cache_config = _make_deepseek_v4_kv_cache_config()

    def _fake_orig(*args, **kwargs):
        raise AssertionError("DeepSeek V4 should use AscendHybridKVCacheCoordinator")

    def _fake_ascend_coordinator(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator._orig_get_kv_cache_coordinator",
        _fake_orig,
    )
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator.AscendHybridKVCacheCoordinator",
        _fake_ascend_coordinator,
    )

    coordinator = get_kv_cache_coordinator(
        kv_cache_config,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=128,
    )

    assert coordinator is sentinel


class _FakeEagleManager:
    def __init__(self) -> None:
        self.use_eagle = False


def test_verify_and_split_propagates_eagle_to_managers() -> None:
    """The MTP bit must reach the manager used by the SWA write mask."""
    kv_cache_config = _make_deepseek_v4_kv_cache_config()
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.kv_cache_config = kv_cache_config
    coordinator.dcp_world_size = 1
    coordinator.pcp_world_size = 1
    coordinator.enable_caching = True
    coordinator.eagle_group_ids = {1}
    coordinator.single_type_managers = (_FakeEagleManager(), _FakeEagleManager())

    coordinator.verify_and_split_kv_cache_groups()

    assert coordinator.single_type_managers[1].use_eagle is True
    assert coordinator.single_type_managers[0].use_eagle is False


def test_verify_and_split_propagates_eagle_to_merged_spec_siblings() -> None:
    """All same-spec siblings share the EAGLE read and write contract."""
    base_config = _make_deepseek_v4_kv_cache_config()
    c128_group_spec = base_config.kv_cache_groups[1].kv_cache_spec
    kv_cache_config = KVCacheConfig(
        num_blocks=base_config.num_blocks,
        kv_cache_tensors=base_config.kv_cache_tensors,
        kv_cache_groups=[
            base_config.kv_cache_groups[0],
            base_config.kv_cache_groups[1],
            KVCacheGroupSpec(
                layer_names=["c128_attn_mtp"],
                kv_cache_spec=c128_group_spec,
            ),
        ],
    )
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.kv_cache_config = kv_cache_config
    coordinator.dcp_world_size = 1
    coordinator.pcp_world_size = 1
    coordinator.enable_caching = True
    coordinator.eagle_group_ids = {2}
    coordinator.single_type_managers = (
        _FakeEagleManager(),
        _FakeEagleManager(),
        _FakeEagleManager(),
    )

    coordinator.verify_and_split_kv_cache_groups()

    assert coordinator.single_type_managers[1].use_eagle is True
    assert coordinator.single_type_managers[2].use_eagle is True
    assert coordinator.single_type_managers[0].use_eagle is False


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_deepseek_v4_effective_lcm_uses_c128_compression(block_size: int) -> None:
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.kv_cache_config = _make_deepseek_v4_apc_config(block_size)
    coordinator.dcp_world_size = 1
    coordinator.pcp_world_size = 1
    coordinator.enable_caching = True
    coordinator.eagle_group_ids = set()
    coordinator.single_type_managers = tuple(_FakeEagleManager() for _ in coordinator.kv_cache_config.kv_cache_groups)

    coordinator.verify_and_split_kv_cache_groups()

    assert coordinator.lcm_block_size == block_size * 128


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_deepseek_v4_coordinator_injects_scoped_apc_components(
    block_size: int,
) -> None:
    coordinator = AscendHybridKVCacheCoordinator(
        kv_cache_config=_make_deepseek_v4_apc_config(block_size),
        max_model_len=block_size * 256,
        max_num_batched_tokens=4096,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=block_size,
    )

    assert isinstance(coordinator.block_pool, AscendDSV4BlockPool)
    swa_managers = [
        manager for manager in coordinator.single_type_managers if isinstance(manager, AscendDSV4SlidingWindowManager)
    ]
    assert len(swa_managers) == 1
    assert swa_managers[0].scheduler_block_size == block_size * 128
    assert any(manager.kv_cache_spec.model_version is None for manager in coordinator.single_type_managers)


@pytest.mark.parametrize(
    ("block_size", "tail_blocks"),
    [
        pytest.param(32, 4, id="block-32"),
        pytest.param(64, 2, id="block-64"),
        pytest.param(128, 1, id="block-128"),
    ],
)
@pytest.mark.parametrize("use_eagle", [False, True], ids=["no-mtp", "mtp"])
def test_deepseek_v4_swa_reachable_mask(
    block_size: int,
    tail_blocks: int,
    use_eagle: bool,
) -> None:
    alignment_tokens = block_size * 128
    spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float16,
        sliding_window=128,
        compress_ratio=1,
        model_version="deepseek_v4",
    )
    # Include exactly two complete LCM segments. MTP additionally computes the
    # first SWA block after the second boundary as its lookahead block.
    end_block = 2 * 128 + int(use_eagle)

    mask = AscendDSV4SlidingWindowManager.reachable_block_mask(
        start_block=0,
        end_block=end_block,
        alignment_tokens=alignment_tokens,
        kv_cache_spec=spec,
        use_eagle=use_eagle,
    )

    assert mask is not None
    actual_indices = {index for index, reachable in enumerate(mask) if reachable}
    expected_indices: set[int] = set()
    for boundary in (128, 256):
        expected_indices.update(range(boundary - tail_blocks, boundary))
        if use_eagle:
            expected_indices.add(boundary)
    assert actual_indices == expected_indices


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("use_eagle", [False, True], ids=["no-mtp", "mtp"])
def test_deepseek_v4_cache_writes_are_lcm_aligned(
    block_size: int,
    use_eagle: bool,
) -> None:
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.is_deepseek_v4 = True
    coordinator.lcm_block_size = block_size * 128
    manager = SimpleNamespace(
        block_size=block_size,
        use_eagle=use_eagle,
        cache_blocks=MagicMock(),
    )
    coordinator.single_type_managers = (manager,)
    request = MagicMock()
    num_computed_tokens = 2 * coordinator.lcm_block_size + block_size

    coordinator.cache_blocks(request, num_computed_tokens)

    expected_tokens = 2 * coordinator.lcm_block_size
    if use_eagle:
        expected_tokens += block_size
    call = manager.cache_blocks.call_args
    assert call is not None
    assert call.args[0] is request
    assert call.args[1] == expected_tokens


def test_deepseek_v4_detection_handles_non_mapping_nested_specs() -> None:
    kv_cache_spec = SimpleNamespace(
        kv_cache_specs=[
            SimpleNamespace(model_version="deepseek_v4"),
        ]
    )
    unknown_spec = SimpleNamespace(kv_cache_specs=object())

    assert _is_deepseek_v4_kv_cache_spec(kv_cache_spec)
    assert not _is_deepseek_v4_kv_cache_spec(unknown_spec)


def test_ascend_mamba_manager_uses_logical_block_size_with_prefix_caching() -> None:
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    block_pool = BlockPool(
        10,
        True,
        16,
        False,
        MagicMock(),
    )

    manager = AscendMambaManager(
        kv_cache_spec=mamba_spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=1,
        dcp_world_size=2,
        pcp_world_size=2,
    )

    assert manager.block_size == mamba_spec.block_size
