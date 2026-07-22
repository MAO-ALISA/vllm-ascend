# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Iterable

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request


class AscendDSV4BlockPool(BlockPool):
    """Block pool extensions used only by the DeepSeek V4 APC path."""

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        if block_mask is None:
            return super().cache_full_blocks(
                request=request,
                blocks=blocks,
                num_cached_blocks=num_cached_blocks,
                num_full_blocks=num_full_blocks,
                block_size=block_size,
                kv_cache_group_id=kv_cache_group_id,
            )

        num_new_full_blocks = max(0, num_full_blocks - num_cached_blocks)
        assert len(block_mask) == num_new_full_blocks

        # The vLLM 0.21 implementation already skips null blocks in both its
        # hash and KV-event paths. Substitute masked positions in a shallow
        # copy so the original request block table remains intact.
        masked_blocks = blocks.copy()
        for block_offset, should_cache in enumerate(block_mask):
            if not should_cache:
                masked_blocks[num_cached_blocks + block_offset] = self.null_block

        return super().cache_full_blocks(
            request=request,
            blocks=masked_blocks,
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=block_size,
            kv_cache_group_id=kv_cache_group_id,
        )

    def free_blocks(
        self,
        ordered_blocks: Iterable[KVCacheBlock],
        prepend: bool = False,
    ) -> None:
        if not prepend:
            return super().free_blocks(ordered_blocks)

        blocks = list(ordered_blocks)
        for block in blocks:
            block.ref_cnt -= 1
        self._prepend_free_blocks([block for block in blocks if block.ref_cnt == 0 and not block.is_null])

    def _prepend_free_blocks(self, blocks: list[KVCacheBlock]) -> None:
        """Put blocks at the front of vLLM 0.21's linked free queue."""
        if not blocks:
            return

        free_queue = self.free_block_queue
        first_block = free_queue.fake_free_list_head.next_free_block
        assert first_block is not None, "next_free_block of fake_free_list_head should always exist"

        previous_block = free_queue.fake_free_list_head
        for block in blocks:
            block.prev_free_block = previous_block
            previous_block.next_free_block = block
            previous_block = block

        previous_block.next_free_block = first_block
        first_block.prev_free_block = previous_block
        free_queue.num_free_blocks += len(blocks)
