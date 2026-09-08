# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.core.slot_apc import validate_slot_apc_config
from vllm_ascend.core.slot_kv_cache_coordinator import AscendSlotKVCacheCoordinator
from vllm_ascend.core.slot_kv_cache_manager import SlotCompressAttentionManager
from vllm_ascend.utils import vllm_version_is

HASH_SIZE = 8


@pytest.fixture(autouse=True)
def slot_env(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_SLOT_APC", "1")
    monkeypatch.setenv("VLLM_VERSION", "0.25.1")
    vllm_version_is.cache_clear()
    init_none_hash(sha256)
    register_ascend_kv_cache_specs()
    yield
    vllm_version_is.cache_clear()


def request(request_id, length, prefix=None):
    tokens = list(range(length)) if prefix is None else prefix
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(HASH_SIZE, sha256),
    )


def spec(ratio):
    return AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
        compress_ratio=ratio,
        model_version="deepseek_v4",
    )


def manager(ratio, num_blocks=64):
    pool = BlockPool(num_blocks, True, HASH_SIZE)
    mgr = SlotCompressAttentionManager(
        kv_cache_spec=spec(ratio),
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=128,
    )
    return mgr, pool


def seed_manager(mgr, req, length):
    mgr.allocate_new_blocks(req.request_id, length, length)
    mgr.cache_blocks(req, length)


def find(mgr, pool, req, length):
    return mgr.find_slot_cache_hit(req.block_hashes, length, [0], pool, mgr.kv_cache_spec)


@pytest.mark.parametrize(
    "ratio,boundaries",
    [
        (4, [0, 128, 256, 384, 512, 640]),
        (128, [0, 128, 16256, 16384, 16512]),
    ],
)
def test_exact_hit_lengths_and_all_aliases(ratio, boundaries):
    mgr, pool = manager(ratio)
    req = request("seed", boundaries[-1])
    seed_manager(mgr, req, req.num_tokens)
    mgr.free(req.request_id)
    for boundary in boundaries:
        blocks, hit = find(mgr, pool, req, boundary)
        assert hit == boundary
        assert len(blocks[0]) == (hit + 128 * ratio - 1) // (128 * ratio)
    # A partial hit never rounds up to the capacity of the resident page.
    assert find(mgr, pool, req, 255)[1] == 128


@pytest.mark.parametrize("ratio", [4, 128])
def test_aliases_survive_tail_growth_full_promotion_and_eviction(ratio):
    mgr, pool = manager(ratio)
    length = 128 * ratio
    req = request("seed", length)
    for end in [128, 256, length]:
        seed_manager(mgr, req, end)
        for boundary in range(128, end + 1, 128):
            assert find(mgr, pool, req, boundary)[1] == boundary
    block = mgr.req_to_blocks[req.request_id][0]
    mgr.free(req.request_id)
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    pool.evict_blocks({block.block_id})
    for boundary in range(128, length + 1, 128):
        assert find(mgr, pool, req, boundary)[1] == 0
    assert not pool.cached_block_hashes_by_block


def test_tail_uses_one_longest_representative():
    mgr, pool = manager(4)
    short = request("short", 128)
    long = request("long", 384)
    seed_manager(mgr, short, 128)
    seed_manager(mgr, long, 384)
    blocks, hit = find(mgr, pool, long, 384)
    assert hit == 384
    assert blocks[0] == mgr.req_to_blocks["long"]


def test_all_same_spec_groups_must_hit_the_same_boundary():
    first, pool = manager(4)
    second = SlotCompressAttentionManager(
        kv_cache_spec=spec(4),
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=1,
        scheduler_block_size=128,
    )
    req = request("seed", 384)
    seed_manager(first, req, 384)
    seed_manager(second, req, 256)
    blocks, hit = first.find_slot_cache_hit(req.block_hashes, 384, [0, 1], pool, first.kv_cache_spec)
    assert hit == 256
    assert blocks == (first.req_to_blocks["seed"], second.req_to_blocks["seed"])


@pytest.mark.parametrize("cancel_before_dispatch", [False, True])
def test_cow_reserves_capacity_and_releases_refs(cancel_before_dispatch):
    mgr, pool = manager(4, num_blocks=4)
    req = request("seed", 384)
    seed_manager(mgr, req, 384)
    src = mgr.req_to_blocks["seed"][0]
    mgr.free("seed")
    blocks, hit = find(mgr, pool, req, 128)
    # One eviction candidate to pin, plus one independent destination page.
    assert mgr.get_num_blocks_to_allocate("new", 256, blocks[0], hit, 256) == 2
    mgr.add_local_computed_blocks("new", blocks[0], hit, 0)
    mgr.allocate_new_blocks("new", 256, 256)
    dst = mgr.req_to_blocks["new"][0]
    assert src is not dst and src.ref_cnt == 1 and dst.ref_cnt == 2
    assert mgr.num_cached_block["new"] == 0
    # Even under pressure, neither page can be reused before the copy.
    spare = pool.get_new_blocks(1)
    assert spare[0] not in (src, dst)
    if cancel_before_dispatch:
        mgr.free("new")
        copies, retained = mgr.take_block_copies()
        assert copies == retained == []
    else:
        copies, retained = mgr.take_block_copies()
        assert [(op.src_block_id, op.dst_block_id) for op in copies] == [(src.block_id, dst.block_id)]
        mgr.free("new")
        assert src.ref_cnt == dst.ref_cnt == 1
        pool.free_blocks(retained)
    pool.free_blocks(spare)
    assert pool.get_num_free_blocks() == 3


def config(num_blocks=4096):
    specs = [spec(4), spec(128)]
    for block_size, window in [(128, 128), (8, 8), (32, 128)]:
        specs.append(
            AscendSlidingWindowMLASpec(
                block_size=block_size,
                sliding_window=window,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.float16,
            )
        )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[str(i)], kv_cache_spec=s) for i, s in enumerate(specs)],
    )


def kv_manager(num_blocks=4096):
    return KVCacheManager(
        kv_cache_config=config(num_blocks),
        max_model_len=32768,
        hash_block_size=HASH_SIZE,
        scheduler_block_size=128,
        enable_caching=True,
        use_eagle=False,
        log_stats=False,
    )


def complete(kv):
    copies = kv.take_block_copies()
    kv.on_step_completed()
    return copies


def test_completed_publication_and_hybrid_exact_hit():
    kv = kv_manager()
    assert isinstance(kv.coordinator, AscendSlotKVCacheCoordinator)
    seed = request("seed", 640)
    assert kv.allocate_slots(seed, 640) is not None
    query = request("query", 257)
    # No hash may expose this batch's unwritten source pages to another request.
    assert kv.get_computed_blocks(query)[1] == 0
    assert complete(kv) == []
    kv.free(seed)
    blocks, hit = kv.get_computed_blocks(query)
    assert hit == 256
    assert len(blocks.blocks[0]) == len(blocks.blocks[1]) == 1
    assert kv.allocate_slots(query, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None
    assert len(complete(kv)) == 2  # C4 and C128; SWA/state hits are aligned.
    kv.free(query)
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1
    assert kv.reset_prefix_cache()
    assert kv.get_computed_blocks(request("after_reset", 257))[1] == 0


def test_hybrid_rechecks_all_compressed_groups_when_state_shortens_hit():
    kv = kv_manager()
    seed = request("seed", 16640)
    assert kv.allocate_slots(seed, seed.num_tokens) is not None
    complete(kv)
    state_manager = kv.coordinator.single_type_managers[-1]
    # Leave state only through 128 tokens. C128 initially matches two pages,
    # but the final common prefix needs just one partial page.
    kv.block_pool.evict_blocks({b.block_id for b in state_manager.req_to_blocks["seed"][4:]})
    blocks, hit = kv.get_computed_blocks(request("query", 16641))
    assert hit == 128
    assert len(blocks.blocks[0]) == len(blocks.blocks[1]) == 1


def test_allocation_failure_does_not_touch_hits_or_leak_cow():
    kv = kv_manager(num_blocks=96)
    seed = request("seed", 128)
    assert kv.allocate_slots(seed, 128) is not None
    complete(kv)
    query = request("query", 256)
    blocks, hit = kv.get_computed_blocks(query)
    assert hit == 128
    held = kv.block_pool.get_new_blocks(kv.block_pool.get_num_free_blocks() - 1)
    # The seed still owns the source pages. One free page cannot satisfy the
    # two compressed groups' COW destinations, even though every group hits.
    before = [b.ref_cnt for b in kv.block_pool.blocks]
    assert (
        kv.allocate_slots(query, query.num_tokens - hit, num_new_computed_tokens=hit, new_computed_blocks=blocks)
        is None
    )
    assert [b.ref_cnt for b in kv.block_pool.blocks] == before
    assert kv.take_block_copies() == []
    kv.block_pool.free_blocks(held)
    kv.free(seed)


@pytest.mark.parametrize("ratio", [4, 128])
def test_full_page_hit_does_not_copy(ratio):
    mgr, pool = manager(ratio)
    length = ratio * 128
    req = request("seed", length)
    seed_manager(mgr, req, length)
    blocks, hit = find(mgr, pool, req, length)
    mgr.add_local_computed_blocks("new", blocks[0], hit, 0)
    mgr.allocate_new_blocks("new", length + 128, length + 128)
    assert mgr.take_block_copies() == ([], [])
    assert mgr.req_to_blocks["new"][0] is mgr.req_to_blocks["seed"][0]


def test_preemption_discards_unpublished_cache_and_copies():
    kv = kv_manager()
    seed = request("seed", 256)
    kv.allocate_slots(seed, 256)
    complete(kv)
    kv.free(seed)
    query = request("query", 257)
    blocks, hit = kv.get_computed_blocks(query)
    kv.allocate_slots(query, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks)
    kv.free(query)
    assert complete(kv) == []
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


def test_abort_after_dispatch_keeps_copy_pages_until_completion():
    kv = kv_manager()
    seed = request("seed", 256)
    kv.allocate_slots(seed, 256)
    complete(kv)
    kv.free(seed)
    query = request("query", 257)
    blocks, hit = kv.get_computed_blocks(query)
    kv.allocate_slots(query, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks)
    copies = kv.take_block_copies()
    kv.free(query)
    assert len(copies) == 2
    for operation in copies:
        assert kv.block_pool.blocks[operation.src_block_id].ref_cnt > 0
        assert kv.block_pool.blocks[operation.dst_block_id].ref_cnt > 0
    kv.on_step_completed()
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


def test_completed_cache_reuse_does_not_require_full_page_boundary():
    kv = kv_manager()
    seed = request("seed", 129)
    kv.allocate_slots(seed, 129)
    complete(kv)
    kv.free(seed)
    # Divergent suffixes share exactly 128 original tokens.
    for i in range(8):
        query = request(str(i), 129, list(range(128)) + [1000 + i])
        blocks, hit = kv.get_computed_blocks(query)
        assert hit == 128
        kv.allocate_slots(query, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks)
        assert len(complete(kv)) == 2
        kv.free(query)
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


def test_chunked_prefill_recycles_state_and_keeps_final_slot():
    kv = kv_manager(num_blocks=512)
    seed = request("seed", 16641)
    while seed.num_computed_tokens < seed.num_tokens:
        kv.new_step_starts()
        count = min(256, seed.num_tokens - seed.num_computed_tokens)
        assert kv.allocate_slots(seed, count) is not None
        seed.num_computed_tokens += count
        complete(kv)
    kv.free(seed)
    blocks, hit = kv.get_computed_blocks(request("query", 16641))
    assert hit == 16640
    assert [len(group) for group in blocks.blocks[:2]] == [33, 2]
    assert kv.block_pool.get_num_free_blocks() == 511


def test_feature_disabled_keeps_existing_coordinator_and_managers(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_SLOT_APC", "0")
    kv = kv_manager()
    assert not isinstance(kv.coordinator, AscendSlotKVCacheCoordinator)
    assert not any(isinstance(m, SlotCompressAttentionManager) for m in kv.coordinator.single_type_managers)
    assert kv.coordinator.lcm_block_size == 16384
    assert kv.take_block_copies() == []


def test_overlapping_steps_are_rejected_before_allocation():
    kv = kv_manager()
    req = request("seed", 256)
    kv.allocate_slots(req, 256)
    kv.take_block_copies()
    before = [block.ref_cnt for block in kv.block_pool.blocks]
    with pytest.raises(RuntimeError, match="one in-flight"):
        kv.new_step_starts()
    assert [block.ref_cnt for block in kv.block_pool.blocks] == before
    kv.on_step_completed()


@pytest.mark.parametrize(
    "field,value",
    [
        ("async_scheduling", True),
        ("block_size", 64),
        ("kv_transfer_config", object()),
        ("speculative_config", object()),
        ("pipeline_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
    ],
)
def test_unsupported_config_rejected(monkeypatch, field, value):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    options = dict(
        block_size=128,
        enable_prefix_caching=True,
        async_scheduling=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
    )
    options[field] = value
    settings = SimpleNamespace(**options)
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v4")),
        cache_config=settings,
        scheduler_config=settings,
        parallel_config=settings,
        kv_transfer_config=options.get("kv_transfer_config"),
        speculative_config=options.get("speculative_config"),
    )
    with pytest.raises(ValueError, match="VLLM_ASCEND_ENABLE_SLOT_APC requires"):
        validate_slot_apc_config(cfg)
