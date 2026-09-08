# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

from vllm_ascend.worker.slot_kv_cache_copy import SlotKVCacheCopyPlan


def test_copy_bounded_pages_including_scales_and_shared_layers():
    # Simulate an aligned view inside a larger allocation, with page padding
    # and indexer scale bytes. The prefix/suffix guards must remain untouched.
    allocation = torch.arange(80, dtype=torch.uint8)
    raw = allocation[8:72]
    indexer = torch.arange(48, dtype=torch.uint8)
    other_group = torch.full((64,), 211, dtype=torch.uint8)
    cfg = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["main", "shared_main", "indexer"]),
            SimpleNamespace(layer_names=["other"]),
        ],
    )
    plan = SlotKVCacheCopyPlan(cfg, dict(main=raw, shared_main=raw, indexer=indexer, other=other_group))
    assert len(plan.pages[0]) == 2
    original = allocation.clone()
    original_indexer = indexer.clone()
    plan.copy_blocks([KVCacheBlockCopy(0, 1, 2), KVCacheBlockCopy(0, 1, 3)])
    for dst in [2, 3]:
        assert torch.equal(raw.view(4, -1)[dst], original[8:72].view(4, -1)[1])
        assert torch.equal(indexer.view(4, -1)[dst], original_indexer.view(4, -1)[1])
    assert torch.equal(allocation[:8], original[:8])
    assert torch.equal(allocation[72:], original[72:])
    assert (other_group == 211).all()
    # A divergent continuation cannot modify the cached source's prefix.
    raw.view(4, -1)[2, 4:] = 99
    assert torch.equal(raw.view(4, -1)[1], original[8:72].view(4, -1)[1])


def test_disjoint_views_of_one_storage_are_both_copied():
    storage = torch.arange(96, dtype=torch.uint8)
    first, second = storage[:32], storage[32:]
    original = storage.clone()
    cfg = SimpleNamespace(num_blocks=4, kv_cache_groups=[SimpleNamespace(layer_names=["first", "second"])])
    plan = SlotKVCacheCopyPlan(cfg, dict(first=first, second=second))
    plan.copy_blocks([KVCacheBlockCopy(0, 1, 3)])
    assert torch.equal(first.view(4, -1)[3], original[:32].view(4, -1)[1])
    assert torch.equal(second.view(4, -1)[3], original[32:].view(4, -1)[1])


@pytest.mark.parametrize("has_copies", [False, True])
def test_runner_copies_after_base_zeroing(monkeypatch, has_copies):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.use_async_scheduling = False
    events = []
    result = object()

    def base_update(self, output):
        events.append("zero")
        return result

    monkeypatch.setattr(GPUModelRunner, "_update_states", base_update)
    runner._apply_pp_sampled_tokens_from_scheduler_output = lambda output: events.append("state")
    runner._slot_kv_copy_plan = SimpleNamespace(copy_blocks=lambda copies: events.append("copy"))
    # Missing metadata represents the unchanged, unpatched upstream interface.
    output = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(req_ids=[]))
    if has_copies:
        output.kv_cache_block_copies = [KVCacheBlockCopy(0, 1, 2)]
    assert runner._update_states(output) is result
    assert events == (["state", "zero", "copy"] if has_copies else ["state", "zero"])
