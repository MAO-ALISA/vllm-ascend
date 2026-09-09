# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus

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


@pytest.mark.parametrize("block_size", [32, 64])
def test_unsupported_compressed_page_size_rejected(block_size):
    with pytest.raises(AssertionError):
        SlotCompressAttentionManager(
            kv_cache_spec=replace(spec(4), block_size=block_size),
            block_pool=BlockPool(64, True, HASH_SIZE),
            enable_caching=True,
            kv_cache_group_id=0,
            scheduler_block_size=128,
        )


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


def dispatch(kv, *requests):
    copies = kv.take_block_copies()
    step_id = kv.on_step_scheduled(requests)
    return step_id, copies


def complete(kv, req=None, end=None):
    requests = [] if req is None else [(req, req.num_tokens if end is None else end)]
    step_id, copies = dispatch(kv, *requests)
    kv.on_step_completed(step_id)
    return copies


def test_completed_publication_and_hybrid_exact_hit():
    kv = kv_manager()
    assert isinstance(kv.coordinator, AscendSlotKVCacheCoordinator)
    seed = request("seed", 640)
    assert kv.allocate_slots(seed, 640) is not None
    query = request("query", 257)
    # No hash may expose this batch's unwritten source pages to another request.
    assert kv.get_computed_blocks(query)[1] == 0
    assert complete(kv, seed) == []
    kv.free(seed)
    blocks, hit = kv.get_computed_blocks(query)
    assert hit == 256
    assert len(blocks.blocks[0]) == len(blocks.blocks[1]) == 1
    assert kv.allocate_slots(query, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None
    assert len(complete(kv, query)) == 2  # C4 and C128; SWA/state hits are aligned.
    kv.free(query)
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1
    assert kv.reset_prefix_cache()
    assert kv.get_computed_blocks(request("after_reset", 257))[1] == 0


def test_hybrid_rechecks_all_compressed_groups_when_state_shortens_hit():
    kv = kv_manager()
    seed = request("seed", 16640)
    assert kv.allocate_slots(seed, seed.num_tokens) is not None
    complete(kv, seed)
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
    complete(kv, seed)
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
    complete(kv, seed)
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
    complete(kv, seed)
    kv.free(seed)
    query = request("query", 257)
    blocks, hit = kv.get_computed_blocks(query)
    kv.allocate_slots(query, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks)
    step_id, copies = dispatch(kv, (query, 257))
    kv.free(query)
    assert len(copies) == 2
    for operation in copies:
        assert kv.block_pool.blocks[operation.src_block_id].ref_cnt > 0
        assert kv.block_pool.blocks[operation.dst_block_id].ref_cnt > 0
    kv.on_step_completed(step_id)
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


def test_completed_cache_reuse_does_not_require_full_page_boundary():
    kv = kv_manager()
    seed = request("seed", 129)
    kv.allocate_slots(seed, 129)
    complete(kv, seed)
    kv.free(seed)
    # Divergent suffixes share exactly 128 original tokens.
    for i in range(8):
        query = request(str(i), 129, list(range(128)) + [1000 + i])
        blocks, hit = kv.get_computed_blocks(query)
        assert hit == 128
        kv.allocate_slots(query, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks)
        assert len(complete(kv, query)) == 2
        kv.free(query)
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


@pytest.mark.parametrize("inflight_depth", [1, 2, 3])
def test_chunked_prefill_recycles_state_and_keeps_final_slot(inflight_depth):
    kv = kv_manager(num_blocks=512)
    seed = request("seed", 16641)
    pending = deque()
    while seed.num_computed_tokens < seed.num_tokens:
        kv.new_step_starts()
        count = min(256, seed.num_tokens - seed.num_computed_tokens)
        assert kv.allocate_slots(seed, count) is not None
        seed.num_computed_tokens += count
        pending.append(dispatch(kv, (seed, seed.num_computed_tokens))[0])
        if len(pending) == inflight_depth:
            kv.on_step_completed(pending.popleft())
    for step in pending:
        kv.on_step_completed(step)
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


def test_overlapping_steps_publish_only_the_completed_snapshot():
    kv = kv_manager()
    req = request("seed", 384)
    kv.allocate_slots(req, 256)
    first, _ = dispatch(kv, (req, 256))
    req.num_computed_tokens = 256
    state = kv.coordinator.single_type_managers[-2]
    first_tail = state.req_to_blocks[req.request_id][15]
    kv.new_step_starts()
    kv.allocate_slots(req, 128)
    second, _ = dispatch(kv, (req, 384))
    req.num_computed_tokens = 384
    # The later step removes the first step's state from the live table, but
    # its snapshot pins that block until it has been published.
    assert state.req_to_blocks[req.request_id][15].is_null
    assert first_tail.ref_cnt > 0
    query = request("query", 385)
    assert kv.get_computed_blocks(query)[1] == 0
    kv.on_step_completed(first)
    assert kv.get_computed_blocks(query)[1] == 256
    assert kv.get_computed_blocks(request("short", 129))[1] == 128
    kv.on_step_completed(second)
    assert kv.get_computed_blocks(query)[1] == 384
    kv.free(req)
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


@pytest.mark.parametrize("pop_for_free", [False, True])
def test_preemption_with_multiple_steps_cannot_publish_into_resumed_request(pop_for_free):
    kv = kv_manager(num_blocks=256)
    req = request("reused-id", 384)
    steps = []
    for end in (128, 256):
        kv.allocate_slots(req, 128)
        steps.append(dispatch(kv, (req, end))[0])
        req.num_computed_tokens = end
    if pop_for_free:
        blocks = kv.pop_blocks_for_free(req)
        kv.block_pool.free_blocks(reversed(blocks))
    else:
        kv.free(req)
    assert not kv.reset_prefix_cache()  # Dispatched snapshots still own pages.
    # Normal preemption reuses the Request object; abort/new admission can
    # reuse an ID. Both must create a new cache-publication lifetime.
    resumed = req if pop_for_free else request("reused-id", 384)
    resumed.num_computed_tokens = 0
    kv.allocate_slots(resumed, 384)
    new_step, _ = dispatch(kv, (resumed, 384))
    new_blocks = kv.get_blocks(resumed.request_id).blocks
    for step in steps:
        kv.on_step_completed(step)
        assert kv.get_computed_blocks(request("probe", 385))[1] == 0
        assert all(b.block_hash is None for group in new_blocks for b in group if not b.is_null)
    kv.on_step_completed(new_step)
    assert kv.get_computed_blocks(request("probe", 385))[1] == 384
    kv.free(resumed)
    assert kv.block_pool.get_num_free_blocks() == 255
    assert kv.reset_prefix_cache()


def test_empty_and_out_of_order_completions_do_not_release_other_steps():
    kv = kv_manager()
    req = request("seed", 256)
    kv.allocate_slots(req, 128)
    first, _ = dispatch(kv, (req, 128))
    empty, copies = dispatch(kv)
    assert empty is None and not copies
    req.num_computed_tokens = 128
    kv.allocate_slots(req, 128)
    second, _ = dispatch(kv, (req, 256))
    before = [b.ref_cnt for b in kv.block_pool.blocks]
    kv.on_step_completed(empty)
    with pytest.raises(RuntimeError, match="out-of-order"):
        kv.on_step_completed(second)
    assert [b.ref_cnt for b in kv.block_pool.blocks] == before
    kv.on_step_completed(first)
    with pytest.raises(RuntimeError, match="out-of-order"):
        kv.on_step_completed(first)
    kv.on_step_completed(second)
    kv.free(req)
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


def test_cow_references_are_released_by_their_own_step():
    kv = kv_manager()
    seed = request("seed", 129)
    kv.allocate_slots(seed, 129)
    complete(kv, seed)
    kv.free(seed)
    pending = []
    for i in range(2):
        req = request(str(i), 256, list(range(128)) + [1000 + i] * 128)
        blocks, hit = kv.get_computed_blocks(req)
        assert hit == 128
        kv.allocate_slots(req, 128, num_new_computed_tokens=hit, new_computed_blocks=blocks)
        step, copies = dispatch(kv, (req, 256))
        assert len(copies) == 2
        pending.append((step, copies))
        kv.free(req)
    kv.on_step_completed(pending[0][0])
    for op in pending[1][1]:
        assert kv.block_pool.blocks[op.src_block_id].ref_cnt > 0
        assert kv.block_pool.blocks[op.dst_block_id].ref_cnt > 0
    for op in pending[0][1]:
        assert kv.block_pool.blocks[op.dst_block_id].ref_cnt == 0
    kv.on_step_completed(pending[1][0])
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


def test_async_scheduler_uses_step_boundary_not_live_counter(monkeypatch):
    kv = kv_manager()
    req = request("decode", 127)
    req.status = RequestStatus.RUNNING
    scheduler = AsyncScheduler.__new__(AsyncScheduler)
    scheduler.kv_cache_manager = kv
    scheduler.requests = {req.request_id: req}
    scheduler.defer_block_free = False
    scheduler.enable_return_routed_experts = False
    scheduler._inflight_prefills = set()
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.use_v2_model_runner = False

    def schedule(count):
        kv.new_step_starts()
        assert kv.allocate_slots(req, count) is not None
        output = SchedulerOutput.make_empty()
        output.num_scheduled_tokens = {req.request_id: count}
        output.total_num_scheduled_tokens = count
        scheduler._update_after_schedule(output)
        return output

    def append_output(self, request, token_ids):
        request.append_output_token_ids(token_ids)
        return token_ids, False

    monkeypatch.setattr(Scheduler, "_update_request_with_output", append_output)
    prefill = schedule(127)
    decode = schedule(1)  # The 128th input token is not known to the CPU yet.
    assert req.num_computed_tokens == 128 and req.num_output_placeholders == 2
    kv.on_step_completed(prefill.kv_cache_step_id)
    scheduler._update_request_with_output(req, [127])
    assert kv.get_computed_blocks(request("probe", 129))[1] == 0
    # Dispatch another step before consuming decode. The first decode's end
    # position must stay 128 although the live request counter is now 129.
    following = schedule(1)
    assert req.num_computed_tokens == 129
    kv.on_step_completed(decode.kv_cache_step_id)
    assert kv.get_computed_blocks(request("probe", 129))[1] == 128
    scheduler._update_request_with_output(req, [128])
    kv.on_step_completed(following.kv_cache_step_id)
    scheduler._update_request_with_output(req, [129])
    assert req.num_output_placeholders == 0
    assert not kv.coordinator._pending_steps
    # A sampled but not executed token must not publish another slot, even if
    # the AsyncScheduler callback supplies an optimistic watermark.
    kv.cache_blocks(req, 256)
    assert kv.get_computed_blocks(request("probe", 257))[1] == 128
    kv.free(req)
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


def test_decode_without_a_new_slot_skips_snapshots():
    kv = kv_manager()
    req = request("seed", 255)
    kv.allocate_slots(req, 128)
    complete(kv, req, 128)
    req.num_computed_tokens = 128
    kv.allocate_slots(req, 1)
    before = [b.ref_cnt for b in kv.block_pool.blocks]
    step, copies = dispatch(kv, (req, 129))
    assert step is None and copies == []
    assert not kv.coordinator._pending_steps
    assert [b.ref_cnt for b in kv.block_pool.blocks] == before
    kv.free(req)


def test_finished_request_does_not_release_later_snapshot_references():
    kv = kv_manager()
    req = request("seed", 256)
    kv.allocate_slots(req, 128)
    first, _ = dispatch(kv, (req, 128))
    req.num_computed_tokens = 128
    kv.allocate_slots(req, 128)
    second, _ = dispatch(kv, (req, 256))
    kv.on_step_completed(first)
    # Simulate stopping on the first output while another step is in flight.
    kv.free(req)
    assert not kv.reset_prefix_cache()
    kv.on_step_completed(second)
    assert kv.get_computed_blocks(request("probe", 257))[1] == 128
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1
    assert not kv.coordinator._request_lifetimes


@pytest.mark.parametrize(
    "field,value",
    [
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
        num_gpu_blocks=None,
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
