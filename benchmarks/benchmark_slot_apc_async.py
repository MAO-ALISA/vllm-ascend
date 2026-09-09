# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU scheduler-side slot APC lifecycle cost with overlapping decode steps."""

import argparse
from collections import deque
from time import perf_counter

import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_ascend import envs
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    register_ascend_kv_cache_specs,
)


def run(prefix_length: int, steps: int, inflight_depth: int) -> None:
    if not envs.VLLM_ASCEND_ENABLE_SLOT_APC:
        raise ValueError("Set VLLM_ASCEND_ENABLE_SLOT_APC=1 with the paired vLLM changes")
    init_none_hash(sha256)
    register_ascend_kv_cache_specs()
    total_length = prefix_length + steps
    common = dict(num_kv_heads=1, head_size=16, dtype=torch.float16)
    specs = [
        AscendMLAAttentionSpec(block_size=128, compress_ratio=ratio, model_version="deepseek_v4", **common)
        for ratio in (4, 128)
    ] + [
        AscendSlidingWindowMLASpec(block_size=size, sliding_window=window, **common)
        for size, window in ((128, 128), (8, 8), (32, 128))
    ]
    num_blocks = 1 + sum(cdiv(total_length, s.block_size * getattr(s, "compress_ratio", 1)) for s in specs)
    kv = KVCacheManager(
        kv_cache_config=KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=[str(i)], kv_cache_spec=s) for i, s in enumerate(specs)],
        ),
        max_model_len=total_length,
        hash_block_size=8,
        scheduler_block_size=128,
        enable_caching=True,
        log_stats=False,
    )
    request = Request(
        request_id="benchmark",
        prompt_token_ids=list(range(total_length)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(8, sha256),
    )
    assert kv.allocate_slots(request, prefix_length) is not None
    assert not kv.take_block_copies()
    kv.on_step_completed(kv.on_step_scheduled([(request, prefix_length)]))
    request.num_computed_tokens = prefix_length
    pending = deque()
    snapshot_steps = 0
    start = perf_counter()
    for _ in range(steps):
        kv.new_step_starts()
        assert kv.allocate_slots(request, 1) is not None
        assert not kv.take_block_copies()
        end = request.num_computed_tokens + 1
        step_id = kv.on_step_scheduled([(request, end)])
        snapshot_steps += step_id is not None
        request.num_computed_tokens = end
        pending.append(step_id)
        if len(pending) == inflight_depth:
            kv.on_step_completed(pending.popleft())
    for step_id in pending:
        kv.on_step_completed(step_id)
    elapsed = perf_counter() - start
    kv.free(request)
    assert kv.block_pool.get_num_free_blocks() == num_blocks - 1
    print(
        f"prefix={prefix_length} steps={steps} inflight={inflight_depth}: "
        f"allocate/seal/complete={elapsed * 1e6 / steps:.2f} us/step; "
        f"snapshot_steps={snapshot_steps}; all references released"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-length", type=int, default=131072)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--inflight-depth", type=int, default=2)
    args = parser.parse_args()
    if args.prefix_length <= 0 or args.prefix_length % 128 or args.steps <= 0 or args.inflight_depth <= 0:
        parser.error("prefix-length must be a positive multiple of 128; steps and inflight-depth must be positive")
    run(args.prefix_length, args.steps, args.inflight_depth)
