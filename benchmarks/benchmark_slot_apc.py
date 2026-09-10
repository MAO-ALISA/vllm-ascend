# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only slot lookup/publication microbenchmark; requires paired vLLM code."""

import argparse
from time import perf_counter

import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.core.slot_apc import SUPPORTED_BLOCK_SIZES
from vllm_ascend.core.slot_kv_cache_manager import SlotCompressAttentionManager


def run(length: int, repeats: int, block_size: int = 128) -> None:
    init_none_hash(sha256)
    request = Request(
        request_id="benchmark",
        prompt_token_ids=list(range(length)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size // 16, sha256),
    )
    for ratio in (4, 128):
        spec = AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.float16,
            compress_ratio=ratio,
            model_version="deepseek_v4",
        )
        pool = BlockPool(length // (block_size * ratio) + 3, True, block_size // 16)
        manager = SlotCompressAttentionManager(
            kv_cache_spec=spec,
            block_pool=pool,
            enable_caching=True,
            kv_cache_group_id=0,
            scheduler_block_size=128,
        )
        manager.allocate_new_blocks(request.request_id, length, length)
        start = perf_counter()
        manager.cache_blocks(request, length)
        publish_ms = (perf_counter() - start) * 1000
        manager.free(request.request_id)
        for boundary in sorted({128, 384, length - 128}):
            start = perf_counter()
            for _ in range(repeats):
                _, hit = manager.find_slot_cache_hit(request.block_hashes, boundary, [0], pool, spec)
                assert hit == boundary
            lookup_us = (perf_counter() - start) * 1e6 / repeats
            print(
                f"B={block_size} C{ratio} cached={length} hit={boundary}: "
                f"lookup={lookup_us:.2f} us; publish={publish_ms:.2f} ms"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--block-size", type=int, choices=SUPPORTED_BLOCK_SIZES, default=128)
    args = parser.parse_args()
    if args.length < 512 or args.length % 128 or args.repeats <= 0:
        parser.error("length must be a multiple of 128 >= 512, and repeats must be positive")
    run(args.length, args.repeats, args.block_size)
