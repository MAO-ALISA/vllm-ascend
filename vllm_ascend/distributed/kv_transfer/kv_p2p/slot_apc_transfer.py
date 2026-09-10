# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned, logical-page transfer plans for Mooncake Hybrid slot APC."""

import hashlib
import json
import struct
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig, SlidingWindowSpec

from vllm_ascend.core.slot_apc import SLOT_SIZE

if TYPE_CHECKING:
    import torch
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

SLOT_TRANSFER_VERSION = 1


def _prefix_hash(tokens: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


class SlotTransferLayout:
    def __init__(self, config: KVCacheConfig) -> None:
        self.specs = [group.kv_cache_spec for group in config.kv_cache_groups]
        # Physical addresses and pool capacity differ between peers. Layer
        # order, compression/state layout, dtype and page geometry must not.
        descriptor = [
            (group.layer_names, type(group.kv_cache_spec).__name__, asdict(group.kv_cache_spec))
            for group in config.kv_cache_groups
        ]
        self.fingerprint = hashlib.sha256(json.dumps(descriptor, sort_keys=True, default=str).encode()).hexdigest()

    def page_count(self, group_id: int, end: int) -> int:
        spec = self.specs[group_id]
        ratio = max(1, getattr(spec, "compress_ratio", 1) or 1)
        # Incomplete compression groups have no KV row yet. Do not confuse
        # ceil(end / logical_page_size) with ceil(floor(end / ratio) / B).
        return cdiv(end // ratio, spec.block_size)

    def required_indices(self, group_id: int, end: int) -> range:
        spec = self.specs[group_id]
        start = max(0, end - spec.sliding_window + 1) // spec.block_size if isinstance(spec, SlidingWindowSpec) else 0
        return range(start, self.page_count(group_id, end))

    def export(self, request: "Request", block_ids: tuple[list[int], ...]) -> tuple[tuple[list[int], ...], dict]:
        # P computes the full prompt. D recomputes its last token; compressor
        # state is position-addressable, so exporting N-1 needs no prompt edit
        # after Scheduler has already performed its local prefix lookup.
        end = request.num_prompt_tokens - 1
        if end <= 0 or request.prompt_token_ids is None or len(block_ids) != len(self.specs):
            raise ValueError("Slot APC transfer requires a token prompt and matching KV groups")
        indices = [list(self.required_indices(i, end)) for i in range(len(self.specs))]
        remote = []
        for blocks, positions in zip(block_ids, indices):
            if any(i >= len(blocks) or blocks[i] <= 0 for i in positions):
                raise ValueError("Slot APC export is missing a required KV/state page")
            remote.append([blocks[i] for i in positions])
        return tuple(remote), {
            "version": SLOT_TRANSFER_VERSION,
            "slot_size": SLOT_SIZE,
            "layout": self.fingerprint,
            "num_tokens": end,
            "prefix_hash": _prefix_hash(request.prompt_token_ids[:end]),
            "block_indices": indices,
        }

    def validate(self, request: "Request", params: dict[str, Any]) -> int:
        if (
            any(
                not isinstance(params.get(key), str) or not params[key]
                for key in ("remote_engine_id", "remote_request_id", "remote_host")
            )
            or type(params.get("remote_port")) is not int
            or not 0 < params["remote_port"] < 65536
        ):
            raise ValueError("Mooncake slot APC manifest is missing valid peer routing metadata")
        meta = params.get("slot_apc")
        if not isinstance(meta, dict) or meta.get("version") != SLOT_TRANSFER_VERSION:
            raise ValueError("Mooncake slot APC requires the versioned slot protocol on both peers")
        if meta.get("slot_size") != SLOT_SIZE or meta.get("layout") != self.fingerprint:
            raise ValueError("Mooncake slot APC peer KV layouts differ (including draft layers)")
        end = meta.get("num_tokens")
        tokens = request.prompt_token_ids
        if type(end) is not int or tokens is None or not 0 < end < request.num_prompt_tokens:
            raise ValueError("Invalid Mooncake slot APC token boundary")
        if meta.get("prefix_hash") != _prefix_hash(tokens[:end]):
            raise ValueError("Mooncake slot APC remote prefix does not match the local prompt")
        remote = params.get("remote_block_ids")
        indices = meta.get("block_indices")
        if not isinstance(remote, (list, tuple)) or len(remote) != len(self.specs):
            raise ValueError("Mooncake slot APC remote group count differs")
        expected = [list(self.required_indices(i, end)) for i in range(len(self.specs))]
        if indices != expected:
            raise ValueError("Mooncake slot APC manifest has missing or invalid logical page indices")
        for blocks, positions in zip(remote, expected):
            if (
                not isinstance(blocks, (list, tuple))
                or len(blocks) != len(positions)
                or any(type(block) is not int or block <= 0 for block in blocks)
                or len(set(blocks)) != len(blocks)
            ):
                raise ValueError("Mooncake slot APC manifest has invalid physical page IDs")
        return end

    def load_plan(
        self, request: "Request", blocks: "KVCacheBlocks", params: dict[str, Any], local_end: int
    ) -> tuple[tuple[list[int], ...], tuple[list[int], ...]]:
        end = self.validate(request, params)
        if not 0 <= local_end < end or local_end % SLOT_SIZE:
            raise ValueError("Mooncake slot APC local hit is not a valid slot boundary")
        if len(blocks.blocks) != len(self.specs):
            raise ValueError("Mooncake slot APC local group count differs")
        local_ids, remote_ids = [], []
        for gid, (spec, group) in enumerate(zip(self.specs, blocks.blocks)):
            positions = params["slot_apc"]["block_indices"][gid]
            remote_map = dict(zip(positions, params["remote_block_ids"][gid]))
            local, remote = [], []
            ratio = max(1, getattr(spec, "compress_ratio", 1) or 1)
            first_unshared_page = local_end // (spec.block_size * ratio)
            for index in positions:
                if index >= len(group) or group[index].is_null:
                    raise ValueError("Mooncake slot APC destination is missing a required KV/state page")
                block = group[index]
                if index < first_unshared_page:
                    # Full local hit pages are read-only. No RDMA may touch them.
                    if block.block_hash is None:
                        raise ValueError("Mooncake slot APC local hit contains an uncommitted page")
                    continue
                if block.block_hash is not None or block.ref_cnt != 1:
                    raise ValueError("Mooncake slot APC receive requires private, unhashed destination pages")
                local.append(block.block_id)
                remote.append(remote_map[index])
            local_ids.append(local)
            remote_ids.append(remote)
        return tuple(local_ids), tuple(remote_ids)


def raw_slot_layout(
    pages: dict[int, list["torch.Tensor"]], num_blocks: int
) -> tuple[list[int], list[int], list[int], list[list[int]]]:
    """Describe bounded raw allocations, including scales and page padding."""
    pointers, lengths, page_bytes, groups = [], [], [], []
    seen = {}
    for gid in sorted(pages):
        for page in pages[gid]:
            if not page.is_contiguous() or page.ndim != 2 or page.shape[0] != num_blocks:
                raise ValueError("Mooncake slot APC requires contiguous block-major raw pages")
            pointer, size = page.data_ptr(), page.numel() * page.element_size()
            key = (pointer, size)
            if key in seen:
                index = seen[key]
                if gid not in groups[index]:
                    groups[index].append(gid)
                continue
            seen[key] = len(pointers)
            pointers.append(pointer)
            lengths.append(size)
            page_bytes.append(size // num_blocks)
            groups.append([gid])
    intervals = sorted(zip(pointers, lengths))
    if not intervals or any(a + size > b for (a, size), (b, _) in zip(intervals, intervals[1:])):
        raise ValueError("Mooncake slot APC raw allocations are empty or overlap")
    return pointers, lengths, page_bytes, groups
