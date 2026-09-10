# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical-page geometry must not change 128-original-token APC semantics."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize, resolve_kv_cache_block_sizes

from vllm_ascend.distributed.kv_transfer.kv_p2p.slot_apc_transfer import SlotTransferLayout

from .test_slot_apc import (
    config,
    dispatch,
    find,
    finish_speculative_step,
    kv_manager,
    manager,
    request,
    seed_manager,
    slot_env,  # noqa: F401 -- shared autouse fixture
)
from .test_slot_apc_transfer import connector, finish, producer


@pytest.fixture(params=[(32, False), (64, False), (128, False), (32, True), (64, True), (128, True)])
def geometry(request):
    block_size, a5 = request.param
    return dict(block_size=block_size, a5=a5)


def query(kv, name, length, prefix=None):
    return request(name, length, prefix, hash_size=kv.coordinator.hash_block_size)


def assert_released(kv):
    assert not kv.coordinator._pending_steps
    assert not kv.coordinator._pending_external
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


def test_resolved_hashes_follow_state_pages_but_joint_hits_stay_128(geometry):
    block_size = geometry["block_size"]
    settings = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size, enable_prefix_caching=True, hash_block_size=None),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        kv_transfer_config=None,
    )
    scheduler_size, hash_size = resolve_kv_cache_block_sizes(config(**geometry), settings)
    assert scheduler_size == block_size and hash_size == block_size // 16
    kv = kv_manager(**geometry)
    assert kv.coordinator.scheduler_block_size == 128
    settings.cache_config.hash_block_size = block_size
    with pytest.raises(ValueError, match="all KV cache group"):
        resolve_kv_cache_block_sizes(config(**geometry), settings)


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("ratio", [4, 128])
def test_slot_aliases_survive_small_page_promotion_and_eviction(block_size, ratio):
    mgr, pool = manager(ratio, block_size=block_size)
    span = block_size * ratio
    req = request("seed", span + 128, hash_size=pool.hash_block_size)
    for end in sorted({128, max(128, span - 128), span, span + 128}):
        seed_manager(mgr, req, end)
        for boundary in range(128, end + 1, 128):
            blocks, hit = find(mgr, pool, req, boundary)
            assert hit == boundary
            assert len(blocks[0]) == (boundary + span - 1) // span
        assert find(mgr, pool, req, end - 1)[1] == end - 128
    first_page = mgr.req_to_blocks[req.request_id][0]
    mgr.free(req.request_id)
    pool.evict_blocks({first_page.block_id})
    assert find(mgr, pool, req, span)[1] == 0
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1


def test_divergent_tails_only_copy_partial_compressed_pages(geometry):
    kv = kv_manager(**geometry)
    seed = query(kv, "seed", geometry["block_size"] * 128 + 128)
    assert kv.allocate_slots(seed, seed.num_tokens) is not None
    finish(kv, seed, seed.num_tokens)
    kv.free(seed)
    for end in sorted({128, 256, geometry["block_size"] * 4, geometry["block_size"] * 128}):
        req = query(kv, f"query-{end}", 0, list(range(end)) + [90000 + end])
        blocks, hit = kv.get_computed_blocks(req)
        assert hit == end
        original = [[b.block_id for b in group] for group in blocks.blocks[:2]]
        assert kv.allocate_slots(req, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None
        step, copies = dispatch(kv, (req, end + 1))
        expected = {gid for gid, ratio in enumerate((4, 128)) if end % (geometry["block_size"] * ratio)}
        assert {copy.group_id for copy in copies} == expected
        for copy in copies:
            assert copy.src_block_id == original[copy.group_id][-1]
            assert copy.dst_block_id != copy.src_block_id
            assert kv.block_pool.blocks[copy.src_block_id].ref_cnt > 0
        kv.on_step_completed(step)
        kv.free(req)
    assert_released(kv)
    assert kv.reset_prefix_cache()
    assert kv.get_computed_blocks(query(kv, "reset", 129))[1] == 0


@pytest.mark.parametrize("draft_layers", [0, 1, 3], ids=["target", "mtp", "dspark"])
@pytest.mark.parametrize("after_dispatch", [False, True])
def test_abort_and_readmission_release_small_page_cow_by_lifetime(geometry, draft_layers, after_dispatch):
    kv = kv_manager(draft_layers=draft_layers, **geometry)
    seed = query(kv, "seed", 256)
    kv.allocate_slots(seed, 256)
    finish(kv, seed, 256)
    kv.free(seed)
    req = query(kv, "reused-id", 0, list(range(128 if not draft_layers else 256)) + [90000])
    blocks, hit = kv.get_computed_blocks(req)
    assert hit == 128
    kv.allocate_slots(req, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks)
    if after_dispatch:
        old_step, copies = dispatch(kv, (req, hit + 1))
        assert len(copies) == (1 if geometry["block_size"] == 32 else 2)
    kv.free(req)
    assert kv.allocate_slots(req, req.num_tokens) is not None
    new_step, new_copies = dispatch(kv, (req, req.num_tokens))
    assert not new_copies
    if after_dispatch:
        kv.on_step_completed(old_step)
        if draft_layers:
            kv.on_step_processed(old_step)  # aborted output has no request callback
        assert kv.coordinator.single_type_managers[0]._num_cached_slots.get(req.request_id, 0) == 0
    if draft_layers:
        finish_speculative_step(kv, new_step, req)
    else:
        kv.on_step_completed(new_step)
    kv.free(req)
    assert_released(kv)


@pytest.mark.parametrize("draft_layers", [1, 3], ids=["mtp", "dspark"])
@pytest.mark.parametrize("rejected", [0, 1, 3])
def test_rejection_at_small_c128_page_boundary(geometry, draft_layers, rejected):
    kv = kv_manager(draft_layers=draft_layers, **geometry)
    boundary = geometry["block_size"] * 128
    req = query(kv, "verify", boundary - 1)
    assert kv.allocate_slots(req, boundary + 2, num_lookahead_tokens=7) is not None
    step, _ = dispatch(kv, (req, boundary + 2))
    req.append_output_token_ids(list(range(boundary - 1, boundary + 3 - rejected)))
    finish_speculative_step(kv, step, req, rejected)
    for mgr in kv.coordinator.single_type_managers[:2]:
        assert mgr._num_cached_slots[req.request_id] * 128 == (boundary + 2 - rejected) // 128 * 128
    kv.free(req)
    assert_released(kv)


@pytest.mark.parametrize("draft_layers", [1, 3], ids=["mtp", "dspark"])
def test_small_draft_peek_remains_completion_fenced_and_all_layers_are_checked(geometry, draft_layers):
    kv = kv_manager(draft_layers=draft_layers, **geometry)
    block_size = geometry["block_size"]
    req = query(kv, "seed", 384)
    kv.allocate_slots(req, 384)
    finish(kv, req, 128)
    if block_size < 128:
        # Physical peek page may have executed, but publication intentionally
        # remains conservative until the next accepted 128-token snapshot.
        finish(kv, req, 128 + block_size)
        assert kv.get_computed_blocks(query(kv, "early", 129 + block_size))[1] == 0
    finish(kv, req, 384)
    kv.free(req)
    probe = query(kv, "probe", 385)
    assert kv.get_computed_blocks(probe)[1] == 256
    for gid in kv.coordinator.eagle_group_ids:
        hashes = BlockHashListWithBlockSize(probe.block_hashes, kv.coordinator.hash_block_size, block_size)
        peek = kv.block_pool.get_cached_block(hashes[256 // block_size], [gid])
        assert peek is not None
        kv.block_pool.evict_blocks({peek[0].block_id})
        assert kv.get_computed_blocks(probe)[1] == 128
    assert_released(kv)


def test_speculative_rewind_keeps_all_small_state_pages(geometry):
    kv = kv_manager(draft_layers=3, **geometry)
    req = query(kv, "decode", 384)
    kv.allocate_slots(req, 256)
    step, _ = dispatch(kv, (req, 256))
    state = kv.coordinator.single_type_managers[3]
    # Rewind from optimistic 264 to 257 still needs the window [250, 257).
    first_needed = (257 - state.sliding_window + 1) // state.block_size
    saved = state.req_to_blocks[req.request_id][first_needed : 256 // state.block_size]
    req.num_computed_tokens, req.num_output_placeholders = 264, 7
    kv.allocate_slots(req, 8)
    assert state.req_to_blocks[req.request_id][first_needed : 256 // state.block_size] == saved
    assert all(not block.is_null for block in saved)
    finish_speculative_step(kv, step, req, rejected=7)
    req.num_output_placeholders = 0
    kv.coordinator.remove_skipped_blocks(req.request_id, 264)
    assert all(block.is_null for block in state.req_to_blocks[req.request_id][first_needed : 256 // state.block_size])
    kv.free(req)
    assert_released(kv)


@pytest.mark.parametrize("peer_size", [32, 64, 128])
def test_mooncake_does_not_silently_reinterpret_different_peer_pages(geometry, peer_size):
    source, req, _, params = producer(256, **geometry)
    other = dict(geometry, block_size=peer_size)
    target = kv_manager(**other)
    incoming = query(target, "target", 257)
    incoming.kv_transfer_params = params
    recv = connector(target)
    if peer_size == geometry["block_size"]:
        assert recv.get_num_new_matched_tokens(incoming, 0) == (256, True)
    else:
        with pytest.raises(ValueError, match="peer KV layouts differ"):
            recv.get_num_new_matched_tokens(incoming, 0)
    source.free(req)
    assert_released(source)
    assert_released(target)


def test_state_page_geometry_is_part_of_transfer_fingerprint(geometry):
    cfg = config(**geometry)
    altered = config(**geometry)
    altered.kv_cache_groups[3].kv_cache_spec = replace(altered.kv_cache_groups[3].kv_cache_spec, block_size=1)
    assert SlotTransferLayout(cfg).fingerprint != SlotTransferLayout(altered).fingerprint
