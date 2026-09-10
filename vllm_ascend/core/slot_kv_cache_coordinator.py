# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-fenced local APC across compressed KV and sliding-window state."""

from collections import deque
from collections.abc import Iterable, Sequence
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
    rejected_tokens: int = 0


@dataclass
class _CacheSnapshot:
    lifetime: _RequestLifetime
    num_tokens: int
    blocks: tuple[list[KVCacheBlock], ...]
    rejected_tokens_at_schedule: int = 0


@dataclass
class _PendingStep:
    step_id: int
    snapshots: list[_CacheSnapshot]
    refs: list[KVCacheBlock]


class AscendSlotKVCacheCoordinator(AscendHybridKVCacheCoordinator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.enable_caching or self.dcp_world_size != 1:
            raise ValueError("Slot APC requires local CP=1 prefix caching")
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
        # The upstream V4 annotation only identifies the last draft layer.
        # DSpark can have several mtp layers, and equal target/draft specs must
        # not merge: only draft KV requires the next-block validity check.
        self.eagle_group_ids = {
            i
            for i, group in enumerate(self.kv_cache_config.kv_cache_groups)
            if self.use_eagle and (group.is_eagle_group or any("mtp" in name.split(".") for name in group.layer_names))
        }
        if self.use_eagle and not self.eagle_group_ids:
            raise ValueError("Slot APC requires identifiable MTP/DSpark draft KV groups")
        for i, manager in enumerate(self.single_type_managers):
            manager.use_eagle = i in self.eagle_group_ids
            if manager.use_eagle and not isinstance(manager, SlidingWindowManager):
                raise ValueError("Slot APC currently requires uncompressed sliding-window MTP/DSpark draft KV")
        groups = []
        for spec, group_ids, manager_cls in self.attention_groups:
            for draft in (False, True):
                ids = [i for i in group_ids if (i in self.eagle_group_ids) == draft]
                if ids:
                    groups.append((spec, ids, manager_cls))
        self.attention_groups = groups
        self._request_lifetimes: dict[str, _RequestLifetime] = {}
        self._copy_refs: list[KVCacheBlock] = []
        self._pending_steps: deque[_PendingStep] = deque()
        self._next_step_id = 0
        self._completing_step: _PendingStep | None = None
        self._completing_snapshots: dict[str, _CacheSnapshot] = {}
        self._pending_external: dict[str, int] = {}

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        # A SWA table encodes the exact local hit, including null padding.
        # The total end can include remote tokens and cannot decide tail COW.
        local_end = next(
            len(new_computed_blocks[i]) * manager.block_size
            for i, manager in enumerate(self.single_type_managers)
            if isinstance(manager, SlidingWindowManager)
        )
        count = 0
        for i, manager in enumerate(self.single_type_managers):
            extra = (
                {"num_local_computed_tokens": local_end} if isinstance(manager, SlotCompressAttentionManager) else {}
            )
            count += manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks[i],
                total_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
                **extra,
            )
        return count

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        super().allocate_new_computed_blocks(
            request_id, new_computed_blocks, num_local_computed_tokens, num_external_computed_tokens
        )
        if num_external_computed_tokens:
            self._pending_external[request_id] = num_local_computed_tokens + num_external_computed_tokens

    def on_remote_cache_ready(self, request: Request, num_computed_tokens: int) -> None:
        expected = self._pending_external.get(request.request_id)
        if expected is None or expected != num_computed_tokens:
            raise RuntimeError("Slot APC received an unpaired or partial remote completion")
        # No forward can use this request while it waits for remote KV. Its
        # live tables are therefore still the allocation paired with the load.
        lifetime = self._request_lifetimes.setdefault(request.request_id, _RequestLifetime(request))
        assert lifetime.request is request and not lifetime.cancelled
        snapshot = _CacheSnapshot(
            lifetime,
            num_computed_tokens,
            tuple(list(manager.req_to_blocks[request.request_id]) for manager in self.single_type_managers),
        )
        self._publish_snapshot(snapshot)
        del self._pending_external[request.request_id]

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
                    draft = group_ids[0] in self.eagle_group_ids
                    hashes = BlockHashListWithBlockSize(block_hashes, self.hash_block_size, spec.block_size)
                    blocks = manager_cls.find_longest_cache_hit(
                        block_hashes=hashes,
                        max_length=min(candidate + spec.block_size, max_cache_hit_length) if draft else candidate,
                        kv_cache_group_ids=group_ids,
                        block_pool=self.block_pool,
                        kv_cache_spec=spec,
                        drop_eagle_block=draft,
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

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int, num_prompt_tokens: int | None = None
    ) -> None:
        lifetime = self._request_lifetimes.get(request_id)
        if self.use_eagle and lifetime is not None:
            # Async scheduling may be ahead of acceptance. Rejected drafts can
            # rewind the next query into the current SWA/compressor-state page.
            total_computed_tokens = max(0, total_computed_tokens - lifetime.request.num_output_placeholders)
        super().remove_skipped_blocks(request_id, total_computed_tokens, num_prompt_tokens)

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
                if self.use_eagle:
                    # Still track rejection offsets for later in-flight steps.
                    snapshots.append(_CacheSnapshot(lifetime, end_position, (), lifetime.rejected_tokens))
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
            # Keep the unrounded end: round only AFTER removing rejected tokens.
            snapshots.append(_CacheSnapshot(lifetime, end_position, tuple(group_blocks), lifetime.rejected_tokens))
        if not snapshots and not refs:
            return None
        step_id = self._next_step_id
        self._next_step_id += 1
        self._pending_steps.append(_PendingStep(step_id, snapshots, refs))
        return step_id

    def on_step_completed(self, step_id: int | None = None) -> None:
        if step_id is None:
            return
        if self._completing_step is not None:
            raise RuntimeError("Slot APC previous speculative step has not been processed")
        # EngineCore consumes batch-queue outputs in dispatch order. Fail closed
        # if that contract changes; do not release another step's references.
        if not self._pending_steps or self._pending_steps[0].step_id != step_id:
            raise RuntimeError(f"Slot APC received out-of-order completion for step {step_id}")
        step = self._pending_steps.popleft()
        if self.use_eagle:
            # Device completion alone does not establish speculative validity.
            # Wait for Scheduler to append accepted IDs, then publish before
            # that request can be freed. Aborts are drained by on_step_processed.
            self._completing_step = step
            self._completing_snapshots = {s.lifetime.request.request_id: s for s in step.snapshots}
            return
        try:
            for snapshot in step.snapshots:
                if not snapshot.lifetime.cancelled:
                    self._publish_snapshot(snapshot)
        finally:
            self.block_pool.free_blocks(reversed(step.refs))

    def on_request_completed(self, step_id: int | None, request: Request, num_rejected_tokens: int) -> None:
        if not self.use_eagle or step_id is None:
            return
        if self._completing_step is None or self._completing_step.step_id != step_id:
            raise RuntimeError(f"Slot APC received request output outside completed step {step_id}")
        snapshot = self._completing_snapshots.pop(request.request_id, None)
        if snapshot is None or snapshot.lifetime.cancelled:
            return
        lifetime = snapshot.lifetime
        assert lifetime.request is request
        assert num_rejected_tokens >= 0
        lifetime.rejected_tokens += num_rejected_tokens
        # Later batches may have been dispatched before this or earlier rejects
        # were known. Subtract ONLY rejects learned since this snapshot was made.
        end_position = snapshot.num_tokens - (lifetime.rejected_tokens - snapshot.rejected_tokens_at_schedule)
        self._publish_snapshot(snapshot, max(0, end_position))

    def on_step_processed(self, step_id: int | None) -> None:
        if not self.use_eagle or step_id is None:
            return
        if self._completing_step is None or self._completing_step.step_id != step_id:
            raise RuntimeError(f"Slot APC received out-of-order processed step {step_id}")
        step, self._completing_step = self._completing_step, None
        self._completing_snapshots.clear()
        self.block_pool.free_blocks(reversed(step.refs))

    def _publish_snapshot(self, snapshot: _CacheSnapshot, end_position: int | None = None) -> None:
        if not snapshot.blocks:
            return
        request = snapshot.lifetime.request
        # Only accepted, hashed IDs whose KV has executed may be published. A
        # sampled bonus token has an ID but no target KV until the next step.
        # Never use the request's ahead-of-execution computed count.
        num_tokens = min(
            snapshot.num_tokens if end_position is None else end_position,
            len(request.block_hashes) * self.hash_block_size,
        )
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
                use_eagle=manager.use_eagle,
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
        self._pending_external.pop(request_id, None)
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
