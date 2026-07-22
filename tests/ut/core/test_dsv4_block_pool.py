# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Iterable

import pytest
from vllm.distributed.kv_events import BlockStored
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request

from vllm_ascend.core.block_pool import AscendDSV4BlockPool

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash_seed() -> None:
    init_none_hash(sha256)


def _make_request(request_id: str, num_tokens: int, block_size: int) -> Request:
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, eos_token_id=num_tokens + 1)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def _free_block_ids(pool: AscendDSV4BlockPool) -> list[int]:
    queue = pool.free_block_queue
    blocks: list[int] = []
    block = queue.fake_free_list_head.next_free_block
    while block is not queue.fake_free_list_tail:
        assert block is not None
        blocks.append(block.block_id)
        block = block.next_free_block
    return blocks


def _block_ids(blocks: Iterable[KVCacheBlock]) -> list[int]:
    return [block.block_id for block in blocks]


def test_cache_full_blocks_masks_hashes_without_mutating_request_blocks() -> None:
    block_size = 4
    pool = AscendDSV4BlockPool(
        num_gpu_blocks=6,
        enable_caching=True,
        hash_block_size=block_size,
        enable_kv_cache_events=True,
    )
    request = _make_request("masked", num_tokens=4 * block_size, block_size=block_size)
    blocks = pool.get_new_blocks(4)
    original_blocks = blocks.copy()

    pool.cache_full_blocks(
        request=request,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=4,
        block_size=block_size,
        kv_cache_group_id=0,
        block_mask=[False, True, False, True],
    )

    assert all(actual is original for actual, original in zip(blocks, original_blocks))
    assert all(block is not pool.null_block for block in blocks)
    assert [block.block_hash is not None for block in blocks] == [
        False,
        True,
        False,
        True,
    ]
    assert len(pool.cached_block_hash_to_block) == 2
    assert pool.get_cached_block(request.block_hashes[0], [0]) is None
    assert pool.get_cached_block(request.block_hashes[1], [0]) == [blocks[1]]
    assert pool.get_cached_block(request.block_hashes[2], [0]) is None
    assert pool.get_cached_block(request.block_hashes[3], [0]) == [blocks[3]]
    events = pool.take_events()
    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert len(events[0].block_hashes) == 2


def test_cache_full_blocks_without_mask_preserves_parent_behavior() -> None:
    block_size = 4
    pool = AscendDSV4BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=block_size,
    )
    request = _make_request("unmasked", num_tokens=2 * block_size, block_size=block_size)
    blocks = pool.get_new_blocks(2)
    free_blocks_before = pool.get_num_free_blocks()

    pool.cache_full_blocks(
        request=request,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=1,
    )

    assert all(block.block_hash is not None for block in blocks)
    assert [block.ref_cnt for block in blocks] == [1, 1]
    assert len(pool.cached_block_hash_to_block) == 2
    assert pool.get_num_free_blocks() == free_blocks_before


def test_prepend_frees_scratch_before_cached_checkpoint_blocks() -> None:
    block_size = 4
    pool = AscendDSV4BlockPool(
        num_gpu_blocks=7,
        enable_caching=True,
        hash_block_size=block_size,
    )
    request = _make_request("free-order", num_tokens=2 * block_size, block_size=block_size)
    allocated = pool.get_new_blocks(4)
    checkpoint_blocks = allocated[:2]
    scratch_blocks = allocated[2:]
    pool.cache_full_blocks(
        request=request,
        blocks=checkpoint_blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=0,
    )

    initial_free_ids = _free_block_ids(pool)
    pool.free_blocks(reversed(checkpoint_blocks))
    pool.free_blocks(reversed(scratch_blocks), prepend=True)

    expected_ids = _block_ids(reversed(scratch_blocks)) + initial_free_ids + _block_ids(reversed(checkpoint_blocks))
    assert _free_block_ids(pool) == expected_ids
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    assert [block.ref_cnt for block in allocated] == [0, 0, 0, 0]


def test_default_free_blocks_appends_and_keeps_shared_blocks_allocated() -> None:
    pool = AscendDSV4BlockPool(
        num_gpu_blocks=6,
        enable_caching=True,
        hash_block_size=4,
    )
    allocated = pool.get_new_blocks(2)
    initial_free_ids = _free_block_ids(pool)
    allocated[0].ref_cnt += 1

    pool.free_blocks(allocated)

    assert allocated[0].ref_cnt == 1
    assert allocated[1].ref_cnt == 0
    assert _free_block_ids(pool) == initial_free_ids + [allocated[1].block_id]
    assert pool.get_num_free_blocks() == len(initial_free_ids) + 1
