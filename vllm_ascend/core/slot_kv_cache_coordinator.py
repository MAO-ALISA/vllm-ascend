# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronous local APC across compressed KV and sliding-window state."""

from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashListWithBlockSize, KVCacheBlock, KVCacheBlockCopy
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.request import Request

from vllm_ascend.core.slot_apc import SLOT_SIZE
from vllm_ascend.core.slot_kv_cache_manager import SlotCompressAttentionManager
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator


class AscendSlotKVCacheCoordinator(AscendHybridKVCacheCoordinator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.enable_caching or self.dcp_world_size != 1 or self.use_eagle:
            raise ValueError("Slot APC requires local CP=1 prefix caching without MTP")
        if SLOT_SIZE % self.hash_block_size:
            raise ValueError("Slot APC requires hash_block_size to divide 128")
        for manager in self.single_type_managers:
            if isinstance(manager, SlidingWindowManager):
                if SLOT_SIZE % manager.block_size:
                    raise ValueError("Slot APC requires sliding-window block sizes to divide 128")
                manager.scheduler_block_size = SLOT_SIZE
            elif not isinstance(manager, SlotCompressAttentionManager):
                raise ValueError(f"Unsupported slot APC manager: {type(manager).__name__}")
        self.scheduler_block_size = SLOT_SIZE
        self._pending_cache: dict[str, tuple[Request, int]] = {}
        self._copy_refs: list[KVCacheBlock] = []
        self._step_in_flight = False

    @property
    def _cache_hit_alignment_tokens(self) -> int:
        return SLOT_SIZE

    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        candidate = max_cache_hit_length // SLOT_SIZE * SLOT_SIZE
        num_groups = len(self.single_type_managers)
        while candidate > 0:
            result: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
            for spec, group_ids, manager_cls in self.attention_groups:
                if issubclass(manager_cls, SlotCompressAttentionManager):
                    blocks, hit = manager_cls.find_slot_cache_hit(
                        block_hashes, candidate, group_ids, self.block_pool, spec
                    )
                else:
                    hashes = BlockHashListWithBlockSize(block_hashes, self.hash_block_size, spec.block_size)
                    blocks = manager_cls.find_longest_cache_hit(
                        block_hashes=hashes,
                        max_length=candidate,
                        kv_cache_group_ids=group_ids,
                        block_pool=self.block_pool,
                        kv_cache_spec=spec,
                        drop_eagle_block=False,
                        alignment_tokens=SLOT_SIZE,
                        dcp_world_size=1,
                        pcp_world_size=1,
                    )
                    hit = len(blocks[0]) * spec.block_size
                if hit < candidate:
                    candidate = hit
                    break
                for group_id, group_blocks in zip(group_ids, blocks):
                    result[group_id] = group_blocks
            else:
                # Every group was looked up at exactly the final length. This
                # includes C128, which must not retain excess pages after SWA
                # shortens the candidate.
                return tuple(result), candidate
        return tuple([] for _ in range(num_groups)), 0

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        # allocate_slots runs before forward. Publishing aliases here would let
        # a later request in this batch copy KV that has not been written yet.
        self._pending_cache[request.request_id] = (request, num_computed_tokens)

    def take_block_copies(self) -> list[KVCacheBlockCopy]:
        if self._step_in_flight:
            raise RuntimeError("Slot APC supports only one in-flight scheduling step")
        self._step_in_flight = bool(self._pending_cache)
        copies = []
        for manager in self.single_type_managers:
            if isinstance(manager, SlotCompressAttentionManager):
                group_copies, refs = manager.take_block_copies()
                copies.extend(group_copies)
                self._copy_refs.extend(refs)
        return copies

    def new_step_starts(self) -> None:
        if self._step_in_flight:
            raise RuntimeError("Slot APC supports only one in-flight scheduling step")
        super().new_step_starts()

    def on_step_completed(self) -> None:
        pending, self._pending_cache = self._pending_cache, {}
        for request, num_tokens in pending.values():
            aligned = num_tokens // SLOT_SIZE * SLOT_SIZE
            for manager in self.single_type_managers:
                manager.cache_blocks(request, aligned, retention_interval=self.retention_interval)
        self.block_pool.free_blocks(self._copy_refs)
        self._copy_refs.clear()
        self._step_in_flight = False

    def free(self, request_id: str) -> None:
        self._pending_cache.pop(request_id, None)
        super().free(request_id)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._pending_cache.pop(request_id, None)
        return super().pop_blocks_for_free(request_id)
