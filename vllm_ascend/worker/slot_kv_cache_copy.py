# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Group-aware copies of bounded, block-major DeepSeek V4 KV allocations."""

from collections import defaultdict

import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import KVCacheConfig


class SlotKVCacheCopyPlan:
    def __init__(self, config: KVCacheConfig, raw_tensors: dict[str, torch.Tensor]) -> None:
        self.pages: dict[int, list[torch.Tensor]] = {}
        for group_id, group in enumerate(config.kv_cache_groups):
            seen = set()
            pages = []
            for layer_name in group.layer_names:
                raw = raw_tensors[layer_name]
                if not isinstance(raw, torch.Tensor) or not raw.is_contiguous():
                    raise ValueError("Slot APC requires contiguous raw KV cache tensors")
                # Use the bounded view, including its storage offset. The
                # underlying allocation can contain leading/trailing padding.
                raw_bytes = raw.view(torch.uint8)
                if raw_bytes.numel() % config.num_blocks:
                    raise ValueError("KV allocation size must be divisible by num_blocks")
                key = (raw_bytes.data_ptr(), raw_bytes.numel())
                if key not in seen:
                    seen.add(key)
                    pages.append(raw_bytes.view(config.num_blocks, -1))
            self.pages[group_id] = pages

    def copy_blocks(self, copies: list[KVCacheBlockCopy]) -> None:
        by_group: dict[int, list[KVCacheBlockCopy]] = defaultdict(list)
        for operation in copies:
            by_group[operation.group_id].append(operation)
        for group_id, operations in by_group.items():
            pages = self.pages[group_id]
            device = pages[0].device
            src = torch.tensor([op.src_block_id for op in operations], dtype=torch.long, device=device)
            dst = torch.tensor([op.dst_block_id for op in operations], dtype=torch.long, device=device)
            for tensor in pages:
                # Gather before writing, also safe when multiple destinations
                # share a source page. Raw pages include indexer scale bytes.
                tensor.index_copy_(0, dst, tensor.index_select(0, src))
