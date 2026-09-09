# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

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
