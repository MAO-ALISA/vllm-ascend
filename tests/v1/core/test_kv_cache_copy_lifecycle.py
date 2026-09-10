# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test


def test_output_copy_metadata_is_independent_and_serializable():
    first, second = SchedulerOutput.make_empty(), SchedulerOutput.make_empty()
    first.kv_cache_block_copies.append(KVCacheBlockCopy(2, 7, 19))
    first.kv_cache_step_id = 7
    assert second.kv_cache_block_copies == []
    restored = pickle.loads(pickle.dumps(first))
    assert restored.kv_cache_block_copies == [KVCacheBlockCopy(2, 7, 19)]
    assert restored.kv_cache_step_id == 7
    assert second.kv_cache_step_id is None


def test_scheduler_drains_copies_into_formal_output():
    coordinator = Mock()
    coordinator.take_block_copies.return_value = [KVCacheBlockCopy(1, 3, 8)]
    coordinator.on_step_scheduled.return_value = 17
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.coordinator = coordinator
    scheduler = SimpleNamespace(
        kv_cache_manager=manager, enable_return_routed_experts=False
    )
    output = SchedulerOutput.make_empty()
    Scheduler._update_after_schedule(scheduler, output)
    assert output.kv_cache_block_copies == [KVCacheBlockCopy(1, 3, 8)]
    assert output.kv_cache_step_id == 17
    coordinator.take_block_copies.assert_called_once_with()


def test_completion_hook_runs_before_processing_or_freeing_requests():
    coordinator = Mock()
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.coordinator = coordinator

    class StopBeforeTokenProcessing:
        @property
        def sampled_token_ids(self):
            coordinator.on_step_completed.assert_called_once_with(23)
            raise StopIteration

    scheduler = SimpleNamespace(kv_cache_manager=manager)
    output = SchedulerOutput.make_empty()
    output.kv_cache_step_id = 23
    with pytest.raises(StopIteration):
        Scheduler.update_from_output(scheduler, output, StopBeforeTokenProcessing())


def test_default_coordinator_hooks_have_no_effect():
    state = SimpleNamespace()
    assert KVCacheCoordinator.take_block_copies(state) == []
    assert KVCacheCoordinator.on_step_scheduled(state, iter(())) is None
    assert KVCacheCoordinator.on_step_completed(state) is None
    assert KVCacheCoordinator.on_request_completed(state, None, Mock(), 0) is None
    assert KVCacheCoordinator.on_step_processed(state, None) is None


def test_manager_forwards_speculative_completion_hooks():
    coordinator = Mock()
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.coordinator = coordinator
    request = Mock()
    manager.on_request_completed(17, request, 3)
    manager.on_step_processed(17)
    coordinator.on_request_completed.assert_called_once_with(17, request, 3)
    coordinator.on_step_processed.assert_called_once_with(17)


@pytest.mark.parametrize("num_bonus_tokens", [0, 1])
@pytest.mark.parametrize("aborted", [False, True])
def test_scheduler_accepts_before_publication_and_releases_after_free(
    monkeypatch, num_bonus_tokens, aborted
):
    from vllm.v1.core.sched import scheduler as scheduler_module

    # Execute the real update loop, replacing only output packaging helpers.
    monkeypatch.setattr(scheduler_module, "defaultdict", defaultdict, raising=False)
    monkeypatch.setattr(
        scheduler_module, "EngineCoreOutput", SimpleNamespace, raising=False
    )
    monkeypatch.setattr(
        scheduler_module, "EngineCoreOutputs", SimpleNamespace, raising=False
    )
    monkeypatch.setattr(scheduler_module, "RequestStatus", RequestStatus, raising=False)
    monkeypatch.setattr(
        scheduler_module,
        "remove_all",
        lambda items, removed: [r for r in items if r not in removed],
        raising=False,
    )
    req = Request(
        request_id="decode",
        prompt_token_ids=list(range(127)),
        sampling_params=SamplingParams(max_tokens=2),
        pooling_params=None,
    )
    req.status = RequestStatus.RUNNING
    req.num_computed_tokens = 131
    req.num_output_placeholders = 3 + num_bonus_tokens
    events = []
    coordinator = Mock()
    coordinator.on_step_completed.side_effect = lambda _: events.append("device")

    def append_tokens(request, tokens):
        events.append("accepted")
        request.append_output_token_ids(tokens)
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return tokens, True

    def publish(step_id, request, rejected):
        events.append("publish")
        assert step_id == 23
        assert list(request.output_token_ids) == [127, 128]
        assert rejected == 1 + num_bonus_tokens
        assert request.num_computed_tokens == 130 - num_bonus_tokens

    coordinator.on_request_completed.side_effect = publish
    coordinator.on_step_processed.side_effect = lambda _: events.append("processed")
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.coordinator = coordinator
    manager.take_events = lambda: []
    structured = Mock()
    structured.should_advance.return_value = False
    scheduler = SimpleNamespace(
        kv_cache_manager=manager,
        requests={} if aborted else {req.request_id: req},
        defer_block_free=False,
        perf_metrics=None,
        enable_return_routed_experts=False,
        num_sampled_tokens_per_step=num_bonus_tokens,
        make_spec_decoding_stats=Mock(return_value=None),
        _update_request_with_output=append_tokens,
        structured_output_manager=structured,
        _handle_stopped_request=Mock(return_value=True),
        _free_request=lambda _: events.append("free"),
        running=[req],
        waiting=Mock(),
        connector=None,
        finished_req_ids_dict={},
        make_stats=Mock(return_value=None),
    )
    output = SchedulerOutput.make_empty()
    output.kv_cache_step_id = 23
    output.num_scheduled_tokens = {req.request_id: 4}
    output.total_num_scheduled_tokens = 4
    output.scheduled_spec_decode_tokens = {req.request_id: [1, 2, 3]}
    result = SimpleNamespace(
        sampled_token_ids=[[127, 128]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        routed_experts=None,
        req_id_to_index={req.request_id: 0},
    )
    Scheduler.update_from_output(scheduler, output, result)
    if aborted:
        assert events == ["device", "processed"]
        coordinator.on_request_completed.assert_not_called()
    else:
        assert events == ["device", "accepted", "publish", "free", "processed"]
