# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed KV pages with 128-token aliases and copy-on-write tails."""

from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock, KVCacheBlockCopy
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.request import Request

from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager
from vllm_ascend.core.slot_apc import SLOT_SIZE, SUPPORTED_BLOCK_SIZES


class SlotCompressAttentionManager(CompressAttentionManager):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logical_block_size = self.block_size * self.compress_ratio
        if self.block_size not in SUPPORTED_BLOCK_SIZES or self.compress_ratio not in (4, 128):
            raise ValueError("Slot APC requires compressed block_size in (32, 64, 128) and compress_ratio in (4, 128)")
        assert self.logical_block_size % SLOT_SIZE == 0
        assert SLOT_SIZE % self.block_pool.hash_block_size == 0
        self._num_cached_slots: dict[str, int] = {}
        self._partial_hits: dict[str, int] = {}
        self._pending_copies: dict[str, tuple[KVCacheBlock, KVCacheBlock]] = {}
        self._remote_tails: set[str] = set()

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        *,
        num_local_computed_tokens: int | None = None,
    ) -> int:
        count = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )
        # Reserve both the hit page (if evictable) and its COW destination.
        local_end = total_computed_tokens if num_local_computed_tokens is None else num_local_computed_tokens
        if new_computed_blocks and local_end % self.logical_block_size:
            count += 1
        return count

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert num_local_computed_tokens % SLOT_SIZE == 0
        assert len(new_computed_blocks) == cdiv(num_local_computed_tokens, self.logical_block_size)
        super().add_local_computed_blocks(
            request_id, new_computed_blocks, num_local_computed_tokens // self.compress_ratio, 0
        )
        # Resident pages are ceil(H/L), while committed full pages are floor(H/L).
        self.num_cached_block[request_id] = num_local_computed_tokens // self.logical_block_size
        self._num_cached_slots[request_id] = num_local_computed_tokens // SLOT_SIZE
        if num_local_computed_tokens % self.logical_block_size:
            self._partial_hits[request_id] = len(new_computed_blocks) - 1
            if num_external_computed_tokens:
                self._remote_tails.add(request_id)

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        if (index := self._partial_hits.pop(request_id, None)) is not None:
            blocks = self.req_to_blocks[request_id]
            src = blocks[index]
            dst = self.block_pool.get_new_blocks(1)[0]
            blocks[index] = dst
            if request_id in self._remote_tails:
                # The remote manifest supplies the WHOLE page, including the
                # matching local prefix. Detach it but do not queue a COW that
                # could overwrite RDMA data in a transfer-only or later step.
                self._remote_tails.remove(request_id)
                self.block_pool.free_blocks([src])
                return super().allocate_new_blocks(request_id, num_tokens, num_tokens_main_model)
            # Transfer the hit reference on src to the copy operation. Keep an
            # extra destination reference too, including if the request aborts.
            self.block_pool.touch([dst])
            self._pending_copies[request_id] = (src, dst)
        return super().allocate_new_blocks(request_id, num_tokens, num_tokens_main_model)

    def take_block_copies(self) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        copies = []
        retained = []
        for src, dst in self._pending_copies.values():
            copies.append(KVCacheBlockCopy(self.kv_cache_group_id, src.block_id, dst.block_id))
            retained.extend((src, dst))
        self._pending_copies.clear()
        return copies, retained

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._partial_hits.pop(request_id, None)
        self._remote_tails.discard(request_id)
        self._num_cached_slots.pop(request_id, None)
        if (copy := self._pending_copies.pop(request_id, None)) is not None:
            # The request was cancelled/preempted before schedule output was
            # built. No device copy will execute, so drop its retained refs now.
            self.block_pool.free_blocks(copy)
        return super().pop_blocks_for_free(request_id)

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        alignment_tokens: int | None = None,
        blocks: list[KVCacheBlock] | None = None,
    ) -> None:
        request_id = request.request_id
        num_slots = num_tokens // SLOT_SIZE
        old_slots = self._num_cached_slots.get(request_id, 0)
        if num_slots <= old_slots:
            return
        num_tokens = num_slots * SLOT_SIZE
        if blocks is None:
            blocks = self.req_to_blocks[request_id]
        span = self.logical_block_size
        num_full_blocks = num_tokens // span
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=self.num_cached_block.get(request_id, 0),
            num_full_blocks=num_full_blocks,
            block_size=span,
            kv_cache_group_id=self.kv_cache_group_id,
        )
        self.num_cached_block[request_id] = num_full_blocks
        # Full promotion and advancement of the primary partial hash remove old
        # aliases in BlockPool. Re-register every interior boundary of each
        # changed page, longest first, so shorter aliases survive advancement.
        for index in range(old_slots * SLOT_SIZE // span, cdiv(num_tokens, span)):
            start = index * span
            end = min(num_tokens, start + span - SLOT_SIZE)
            for boundary in range(end, start, -SLOT_SIZE):
                self.block_pool.cache_partial_block(request, blocks[index], boundary, self.kv_cache_group_id, span)
        self._num_cached_slots[request_id] = num_slots

    @classmethod
    def find_slot_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """Return full pages followed by at most one partial page, and exact H."""
        span = kv_cache_spec.block_size * kv_cache_spec.compress_ratio
        hash_size = block_pool.hash_block_size
        max_length = min(max_length, len(block_hashes) * hash_size)
        max_length = max_length // SLOT_SIZE * SLOT_SIZE
        computed: tuple[list[KVCacheBlock], ...] = tuple([] for _ in kv_cache_group_ids)
        hit = 0
        while hit + span <= max_length:
            blocks = block_pool.get_cached_block(block_hashes[(hit + span) // hash_size - 1], kv_cache_group_ids)
            if blocks is None:
                break
            for group_blocks, block in zip(computed, blocks):
                group_blocks.append(block)
            hit += span
        # Choose one representative for the tail's longest available boundary.
        # Collecting a different block for every slot could mix incompatible
        # physical pages belonging to the same logical block.
        for boundary in range(min(max_length, hit + span - SLOT_SIZE), hit, -SLOT_SIZE):
            blocks = block_pool.get_cached_block(block_hashes[boundary // hash_size - 1], kv_cache_group_ids)
            if blocks is not None:
                for group_blocks, block in zip(computed, blocks):
                    group_blocks.append(block)
                return computed, boundary
        return computed, hit
