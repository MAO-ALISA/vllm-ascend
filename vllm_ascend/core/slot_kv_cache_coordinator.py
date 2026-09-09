# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-fenced local APC across compressed KV and sliding-window state."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashListWithBlockSize, KVCacheBlock, KVCacheBlockCopy
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.request import Request

from vllm_ascend.core.slot_apc import SLOT_SIZE
from vllm_ascend.core.slot_kv_cache_manager import SlotCompressAttentionManager
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator


@dataclass
class _RequestLifetime:
    request: Request
    cancelled: bool = False


@dataclass
class _CacheSnapshot:
    lifetime: _RequestLifetime
    num_tokens: int
    blocks: tuple[list[KVCacheBlock], ...]


@dataclass
class _PendingStep:
    step_id: int
    snapshots: list[_CacheSnapshot]
    refs: list[KVCacheBlock]


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
        self._request_lifetimes: dict[str, _RequestLifetime] = {}
        self._copy_refs: list[KVCacheBlock] = []
        self._pending_steps: deque[_PendingStep] = deque()
        self._next_step_id = 0

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
        # Neither allocate_slots' optimistic count nor AsyncScheduler's output
        # callback identifies the block table for a particular execution step.
        # Publication is owned exclusively by the paired step hooks below.
        pass

    def take_block_copies(self) -> list[KVCacheBlockCopy]:
        copies = []
        for manager in self.single_type_managers:
            if isinstance(manager, SlotCompressAttentionManager):
                group_copies, refs = manager.take_block_copies()
                copies.extend(group_copies)
                self._copy_refs.extend(refs)
        return copies

    def on_step_scheduled(self, requests: Iterable[tuple[Request, int]]) -> int | None:
        snapshots = []
        refs, self._copy_refs = self._copy_refs, []
        for request, end_position in requests:
            request_id = request.request_id
            lifetime = self._request_lifetimes.get(request_id)
            if lifetime is None:
                lifetime = self._request_lifetimes[request_id] = _RequestLifetime(request)
            assert lifetime.request is request
            aligned = end_position // SLOT_SIZE * SLOT_SIZE
            if all(
                aligned
                <= (
                    manager._num_cached_slots.get(request_id, 0) * SLOT_SIZE
                    if isinstance(manager, SlotCompressAttentionManager)
                    else manager.num_cached_block.get(request_id, 0) * manager.block_size
                )
                for manager in self.single_type_managers
            ):
                # Most decode steps do not cross a slot boundary. Avoid copying
                # context-sized block tables or pinning pages on those steps.
                continue
            group_blocks = []
            for manager in self.single_type_managers:
                span = getattr(manager, "logical_block_size", manager.block_size)
                # The live table can lose SWA/state pages in the next schedule,
                # or be replaced entirely on preemption. Never consult it later.
                blocks = manager.req_to_blocks[request_id][: cdiv(aligned, span)]
                group_blocks.append(blocks)
                # Published full pages are immutable and won't be republished.
                # Pin only the unpublished range (including a compressed tail).
                start = manager.num_cached_block.get(request_id, 0)
                pinned = [block for block in blocks[start:] if not block.is_null]
                self.block_pool.touch(pinned)
                refs.extend(pinned)
            snapshots.append(_CacheSnapshot(lifetime, aligned, tuple(group_blocks)))
        if not snapshots and not refs:
            return None
        step_id = self._next_step_id
        self._next_step_id += 1
        self._pending_steps.append(_PendingStep(step_id, snapshots, refs))
        return step_id

    def on_step_completed(self, step_id: int | None = None) -> None:
        if step_id is None:
            return
        # EngineCore consumes batch-queue outputs in dispatch order. Fail closed
        # if that contract changes; do not release another step's references.
        if not self._pending_steps or self._pending_steps[0].step_id != step_id:
            raise RuntimeError(f"Slot APC received out-of-order completion for step {step_id}")
        step = self._pending_steps.popleft()
        try:
            for snapshot in step.snapshots:
                if not snapshot.lifetime.cancelled:
                    self._publish_snapshot(snapshot)
        finally:
            self.block_pool.free_blocks(reversed(step.refs))

    def _publish_snapshot(self, snapshot: _CacheSnapshot) -> None:
        request = snapshot.lifetime.request
        # Without speculation, input token IDs for this step are known after
        # processing earlier FIFO outputs. The sampled token of THIS step has
        # no KV yet. Never use the request's ahead-of-execution computed count.
        num_tokens = min(snapshot.num_tokens, len(request.block_hashes) * self.hash_block_size)
        num_tokens = num_tokens // SLOT_SIZE * SLOT_SIZE
        for manager, blocks in zip(self.single_type_managers, snapshot.blocks):
            if isinstance(manager, SlotCompressAttentionManager):
                manager.cache_blocks(request, num_tokens, blocks=blocks)
                continue
            start = manager.num_cached_block.get(request.request_id, 0)
            end = num_tokens // manager.block_size
            if end <= start:
                continue
            mask = manager.reachable_block_mask(
                start_block=start,
                end_block=end,
                alignment_tokens=SLOT_SIZE,
                kv_cache_spec=manager.kv_cache_spec,
                use_eagle=False,
                retention_interval=self.retention_interval,
                num_prompt_tokens=request.num_prompt_tokens,
            )
            self.block_pool.cache_full_blocks(
                request=request,
                blocks=blocks,
                num_cached_blocks=start,
                num_full_blocks=end,
                block_size=manager.block_size,
                kv_cache_group_id=manager.kv_cache_group_id,
                block_mask=mask,
            )
            manager.num_cached_block[request.request_id] = end

    def _cancel_lifetime(self, request_id: str) -> None:
        if (lifetime := self._request_lifetimes.pop(request_id, None)) is not None:
            # Keep dispatched refs until their own completion, but never
            # publish a cancelled generation into a resumed request's tables.
            lifetime.cancelled = True

    def free(self, request_id: str) -> None:
        self._cancel_lifetime(request_id)
        super().free(request_id)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._cancel_lifetime(request_id)
        return super().pop_blocks_for_free(request_id)
