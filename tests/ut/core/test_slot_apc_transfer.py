# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_hybrid_connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
    KVCacheRecvingThread,
    KVCacheTaskTracker,
    MooncakeAgentMetadata,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.slot_apc_transfer import (
    SLOT_TRANSFER_VERSION,
    SlotTransferLayout,
    raw_slot_layout,
)

from .test_slot_apc import (
    config,
    dispatch,
    finish_speculative_step,
    kv_manager,
    request,
    slot_env,  # noqa: F401 -- shared autouse fixture
)


def finish(kv, req, end):
    step, _ = dispatch(kv, (req, end))
    if kv.use_eagle:
        finish_speculative_step(kv, step, req)
    else:
        kv.on_step_completed(step)


def connector(kv):
    result = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    result.slot_apc = True
    result.slot_layout = SlotTransferLayout(kv.coordinator.kv_cache_config)
    result.use_hybrid = True
    result.block_size = kv.coordinator.hash_block_size  # Runtime minimum includes compressor state.
    result.engine_id = "producer"
    result.side_channel_host = "127.0.0.1"
    result.side_channel_port = 12345
    result.tp_size = 1
    result.multi_nodes_meta_mapping = {}
    result._reqs_need_send = {}
    result._reqs_need_recv = {}
    result._reqs_in_batch = set()
    return result


def producer(end, draft_layers=0, *, block_size=128, a5=False):
    kv = kv_manager(num_blocks=4096 * 128 // block_size, draft_layers=draft_layers, block_size=block_size, a5=a5)
    req = request("producer", end + 1, hash_size=block_size // 16)
    req.kv_transfer_params = {"do_remote_decode": True}
    sender = connector(kv)
    original = list(req.prompt_token_ids)
    assert sender.get_num_new_matched_tokens(req, 0) == (0, False)
    assert req.prompt_token_ids == original  # No mutation after local APC lookup.
    assert kv.allocate_slots(req, req.num_tokens) is not None
    finish(kv, req, req.num_tokens)
    req.append_output_token_ids(90000)
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    delayed, params = sender.request_finished_all_groups(req, kv.get_blocks(req.request_id).get_block_ids())
    assert delayed and req.request_id in sender._reqs_need_send
    # Exercise the actual frontend JSON transport of the manifest.
    return kv, req, sender, msgspec.json.decode(msgspec.json.encode(params))


def byte_receiver(source_kv, target_kv):
    count = source_kv.block_pool.num_gpu_blocks
    source, target = {}, {}
    for gid in range(len(source_kv.coordinator.single_type_managers)):
        # Deliberately include non-KV bytes, like scale/padding payloads.
        width = source_kv.coordinator.single_type_managers[gid].block_size * (gid + 1) + 5
        source[gid] = [(torch.arange(count * width) % 251).to(torch.uint8).view(count, width)]
        target[gid] = [torch.full((count, width), 255, dtype=torch.uint8)]
    src_ptrs, src_lengths, src_bytes, src_groups = raw_slot_layout(source, count)
    dst_ptrs, dst_lengths, dst_bytes, dst_groups = raw_slot_layout(target, count)
    assert src_bytes == dst_bytes and src_groups == dst_groups
    allocations = [page.view(-1) for groups in (source, target) for pages in groups.values() for page in pages]

    def view(address, length):
        for allocation in allocations:
            offset = address - allocation.data_ptr()
            if offset >= 0 and offset + length <= allocation.numel():
                return allocation[offset : offset + length]
        raise AssertionError("Transfer escaped a bounded allocation")

    def read(session, local_addresses, remote_addresses, lengths):
        for local, remote, length in zip(local_addresses, remote_addresses, lengths):
            view(local, length).copy_(view(remote, length))
        return 0

    receiver = KVCacheRecvingThread.__new__(KVCacheRecvingThread)
    receiver.slot_apc = True
    receiver.local_engine_id = "consumer"
    receiver.local_handshake_port = 1000
    receiver.tp_rank = 0
    receiver.kv_cache_config = target_kv.coordinator.kv_cache_config
    receiver.kv_cache_specs = [g.kv_cache_spec for g in receiver.kv_cache_config.kv_cache_groups]
    receiver.hma_group_size = len(receiver.kv_cache_specs)
    receiver.addr_group_idx = dst_groups
    receiver.block_len_per_addr = dst_bytes
    receiver.block_stride_per_addr = dst_bytes
    receiver.remote_metadata_lock = threading.Lock()
    receiver.kv_caches_base_addr = {"producer": {12345: src_ptrs}, "consumer": {1000: dst_ptrs}}
    receiver.remote_te_port = {"producer": {12345: 7777}}
    receiver.remote_num_blocks = {("producer", 12345): count}
    receiver.engine = SimpleNamespace(batch_transfer_sync_read=Mock(side_effect=read))
    return receiver, source, target


@pytest.mark.parametrize("end", [1, 3, 4, 7, 8, 31, 32, 127, 128, 129, 255, 256, 511, 512, 513, 16383, 16384, 16385])
@pytest.mark.parametrize("local_hit", [False, True])
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_remote_slot_round_trip_preserves_shared_pages_and_waits_for_completion(end, local_hit, block_size):
    source_kv, source_req, sender, params = producer(end, block_size=block_size)
    target_kv = kv_manager(num_blocks=source_kv.block_pool.num_gpu_blocks, block_size=block_size)
    hash_size = target_kv.coordinator.hash_block_size
    local_end = 128 if local_hit and end > 128 else 0
    if local_end:
        seed = request("local-seed", local_end, hash_size=hash_size)
        target_kv.allocate_slots(seed, local_end)
        finish(target_kv, seed, local_end)
        target_kv.free(seed)
    target = request("target", end + 1, hash_size=hash_size)
    target.kv_transfer_params = params
    blocks, hit = target_kv.get_computed_blocks(target)
    assert hit == local_end
    shared_ids = [[b.block_id for b in group if not b.is_null] for group in blocks.blocks]
    receiver_scheduler = connector(target_kv)
    external, asynchronous = receiver_scheduler.get_num_new_matched_tokens(target, hit)
    assert asynchronous and external == end - hit
    assert (
        target_kv.allocate_slots(
            target,
            0,
            num_new_computed_tokens=hit,
            new_computed_blocks=blocks,
            num_external_computed_tokens=external,
            delay_cache_blocks=True,
        )
        is not None
    )
    receiver_scheduler.update_state_after_alloc(target, target_kv.get_blocks(target.request_id), external)
    metadata = receiver_scheduler.build_connector_meta(SchedulerOutput.make_empty())
    transfer = metadata.requests[target.request_id]
    # RDMA supplies private partial tails; no pending COW may clobber them.
    assert target_kv.take_block_copies() == []
    # Supported DSv4 groups do not queue generic whole-pool zeroing that
    # could race with the receive thread's writes to private destinations.
    assert all(not manager.take_new_block_ids() for manager in target_kv.coordinator.single_type_managers)
    assert target_kv.on_step_scheduled([]) is None
    assert target_kv.coordinator.single_type_managers[0]._num_cached_slots[target.request_id] == hit // 128

    receiver, source_bytes, target_bytes = byte_receiver(source_kv, target_kv)
    receiver._transfer_kv_cache_all_groups(
        dict(
            remote_request_id=source_req.request_id,
            remote_block_ids=transfer.remote_block_ids,
            local_block_ids=transfer.local_block_ids,
            remote_engine_id="producer",
            remote_host="127.0.0.1",
            remote_handshake_port=12345,
        )
    )
    for gid, (local_ids, remote_ids) in enumerate(zip(transfer.local_block_ids, transfer.remote_block_ids)):
        for local, remote in zip(local_ids, remote_ids):
            assert torch.equal(target_bytes[gid][0][local], source_bytes[gid][0][remote])
        for shared in shared_ids[gid]:
            assert torch.all(target_bytes[gid][0][shared] == 255)

    target.num_computed_tokens = end
    target.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    target_kv.cache_blocks(target, end)  # Ordinary callbacks cannot publish RDMA.
    assert target_kv.coordinator.single_type_managers[0]._num_cached_slots[target.request_id] == hit // 128
    scheduler = SimpleNamespace(
        connector=object(),
        kv_cache_manager=target_kv,
        failed_recving_kv_req_ids=set(),
        finished_recving_kv_req_ids={target.request_id},
    )
    Scheduler._update_waiting_for_remote_kv(scheduler, target)
    assert target_kv.coordinator.single_type_managers[0]._num_cached_slots[target.request_id] == end // 128
    assert not scheduler.finished_recving_kv_req_ids
    assert not target_kv.coordinator._pending_external
    if end % 128 == 0:
        assert target_kv.get_computed_blocks(request("probe", end + 1, hash_size=hash_size))[1] == end
    # Only the successful transfer releases the producer's delayed request.
    source_kv.free(source_req)
    target_kv.free(target)
    for kv in (source_kv, target_kv):
        assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


@pytest.mark.parametrize("damage", [None, "version", "layout", "groups", "bytes", "addresses", "capacity"])
def test_peer_handshake_validates_raw_layout_before_caching(monkeypatch, damage):
    receiver = KVCacheRecvingThread.__new__(KVCacheRecvingThread)
    receiver.slot_apc = True
    receiver.slot_layout = SlotTransferLayout(config())
    receiver.local_engine_id = "consumer"
    receiver.addr_group_idx = [[0, 1], [2], [3], [4]]
    receiver.block_len_per_addr = [133, 128, 64, 256]
    receiver.remote_metadata_lock = threading.Lock()
    receiver.kv_caches_base_addr = {"producer": {}}
    receiver.remote_te_port = {"producer": {}}
    receiver.remote_num_blocks = {}
    receiver.encoder = msgspec.msgpack.Encoder()
    receiver.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
    receiver._get_remote_socket = Mock(return_value=object())
    receiver._return_remote_socket = Mock()
    metadata = MooncakeAgentMetadata(
        engine_id="producer",
        te_rpc_port=7777,
        block_size=8,
        kv_caches_base_addr=[1000, 2000, 3000, 4000],
        num_blocks=32,
        block_lens=list(receiver.block_len_per_addr),
        ssm_sizes=(0, 0),
        slot_apc_version=SLOT_TRANSFER_VERSION,
        slot_apc_layout=receiver.slot_layout.fingerprint,
        slot_apc_addr_groups=copy.deepcopy(receiver.addr_group_idx),
    )
    if damage == "version":
        metadata.slot_apc_version = 0
    elif damage == "layout":
        metadata.slot_apc_layout = "different"
    elif damage == "groups":
        metadata.slot_apc_addr_groups[0].reverse()
    elif damage == "bytes":
        metadata.block_lens[0] += 1
    elif damage == "addresses":
        metadata.kv_caches_base_addr.pop()
    elif damage == "capacity":
        metadata.num_blocks = 0
    monkeypatch.setattr(connector_module, "ensure_zmq_send", Mock(), raising=False)
    monkeypatch.setattr(
        connector_module, "ensure_zmq_recv", Mock(return_value=msgspec.msgpack.encode(metadata)), raising=False
    )
    if damage:
        with pytest.raises(ValueError, match="peer raw layouts"):
            receiver._get_remote_metadata("127.0.0.1", 12345)
        assert not receiver.kv_caches_base_addr["producer"] and not receiver.remote_num_blocks
    else:
        receiver._get_remote_metadata("127.0.0.1", 12345)
        assert receiver.remote_num_blocks[("producer", 12345)] == 32
        assert receiver.kv_caches_base_addr["producer"][12345] == metadata.kv_caches_base_addr


def test_worker_registers_bounded_raw_pages_including_padding(monkeypatch):
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.slot_apc = True
    worker.kv_cache_config = config(num_blocks=32)
    worker.slot_layout = SlotTransferLayout(worker.kv_cache_config)
    worker.hma_group_size = len(worker.kv_cache_config.kv_cache_groups)
    worker.slot_pages = {i: [torch.zeros(32, 133 + i, dtype=torch.uint8)] for i in range(worker.hma_group_size)}
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_mla=True, hf_text_config=SimpleNamespace())
    )
    worker.engine_id, worker.te_rpc_port, worker.block_size = "producer", 7777, 8
    worker._mamba_ssm_size = (0, 0)
    worker.kv_role = "kv_producer"
    worker.tp_rank, worker._prefill_tp_size = 0, 1
    worker.side_channel_host, worker.side_channel_port = "127.0.0.1", 12345
    engine = Mock()
    monkeypatch.setattr(connector_module, "global_te", engine, raising=False)

    def make_sender(*args):
        args[7].set()  # ready_event; do not start an actual device/network thread.
        return Mock()

    monkeypatch.setattr(connector_module, "KVCacheSendingThread", make_sender)
    worker.register_kv_caches({})
    pointers, lengths, widths, groups = raw_slot_layout(worker.slot_pages, 32)
    engine.register_buffer.assert_called_once_with(pointers, lengths)
    meta = worker.xfer_handshake_metadata
    assert meta.block_lens == widths and meta.slot_apc_addr_groups == groups
    assert meta.slot_apc_layout == worker.slot_layout.fingerprint


def test_fast_receive_is_tracked_before_enqueue():
    source_kv, source_req, _, params = producer(128)
    target_kv = kv_manager()
    req = request("target", 129)
    req.kv_transfer_params = params
    recv = connector(target_kv)
    recv.get_num_new_matched_tokens(req, 0)
    target_kv.allocate_slots(req, 0, num_external_computed_tokens=128, delay_cache_blocks=True)
    recv.update_state_after_alloc(req, target_kv.get_blocks(req.request_id), 128)
    metadata = recv.build_connector_meta(SchedulerOutput.make_empty())
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.slot_apc = True
    worker.use_mamba = False
    worker.kv_send_thread = None
    tracker = KVCacheTaskTracker()
    worker.kv_recv_thread = SimpleNamespace(
        task_tracker=tracker, add_request=lambda **kwargs: tracker.update_done_task_count(kwargs["request_id"])
    )
    worker._prefill_tp_size = worker._prefill_pp_size = 1
    worker._get_tp_num_need_pulls = lambda _: 1
    worker._get_remote_rank = lambda *_: [0]
    worker.start_load_kv(metadata)
    assert tracker.get_and_clear_finished_requests() == {req.request_id}
    target_kv.free(req)
    source_kv.free(source_req)


def test_prefill_full_local_slot_hit_does_not_truncate_prompt_after_lookup():
    kv = kv_manager()
    seed = request("seed", 128)
    kv.allocate_slots(seed, 128)
    finish(kv, seed, 128)
    kv.free(seed)
    req = request("prefill", 129)
    req.kv_transfer_params = {"do_remote_decode": True}
    blocks, hit = kv.get_computed_blocks(req)
    assert hit == 128
    assert connector(kv).get_num_new_matched_tokens(req, hit) == (0, False)
    assert req.num_tokens - hit == 1
    assert kv.allocate_slots(req, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None
    kv.free(req)


@pytest.mark.parametrize("draft_layers", [1, 3], ids=["mtp", "dspark"])
@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("a5", [False, True], ids=["standard", "a5-state"])
def test_remote_speculative_prefix_then_local_decode(draft_layers, block_size, a5):
    source_kv, source_req, _, params = producer(511, draft_layers, block_size=block_size, a5=a5)
    target_kv = kv_manager(
        num_blocks=source_kv.block_pool.num_gpu_blocks, draft_layers=draft_layers, block_size=block_size, a5=a5
    )
    target = request("target", 512, hash_size=block_size // 16)
    target.kv_transfer_params = params
    recv = connector(target_kv)
    assert recv.get_num_new_matched_tokens(target, 0) == (511, True)
    assert target_kv.allocate_slots(target, 0, num_external_computed_tokens=511, delay_cache_blocks=True) is not None
    recv.update_state_after_alloc(target, target_kv.get_blocks(target.request_id), 511)
    meta = recv.build_connector_meta(SchedulerOutput.make_empty()).requests[target.request_id]
    assert len(meta.remote_block_ids) == 5 + draft_layers
    for gid in range(5, 5 + draft_layers):
        assert meta.local_block_ids[gid]  # Every DSpark layer must be loaded.
    receiver, source_bytes, target_bytes = byte_receiver(source_kv, target_kv)
    receiver._transfer_kv_cache_all_groups(
        dict(
            remote_request_id=source_req.request_id,
            remote_block_ids=meta.remote_block_ids,
            local_block_ids=meta.local_block_ids,
            remote_engine_id="producer",
            remote_host="127.0.0.1",
            remote_handshake_port=12345,
        )
    )
    for gid, (local_ids, remote_ids) in enumerate(zip(meta.local_block_ids, meta.remote_block_ids)):
        for local, remote in zip(local_ids, remote_ids):
            assert torch.equal(target_bytes[gid][0][local], source_bytes[gid][0][remote])
    target_kv.on_remote_cache_ready(target, 511)
    target.num_computed_tokens = 511
    target_kv.allocate_slots(target, 1, num_lookahead_tokens=7)
    step, _ = dispatch(target_kv, (target, 512))
    finish_speculative_step(target_kv, step, target)
    assert target_kv.coordinator.single_type_managers[0]._num_cached_slots[target.request_id] == 4
    source_kv.free(source_req)
    target_kv.free(target)


@pytest.mark.parametrize("damage", ["version", "layout", "tokens", "prefix", "indices", "missing", "null", "routing"])
def test_invalid_remote_manifest_rejected_before_allocation(damage):
    kv, req, _, params = producer(512)
    target = request("target", 513)
    bad = copy.deepcopy(params)
    if damage == "version":
        bad["slot_apc"]["version"] += 1
    elif damage == "layout":
        bad["slot_apc"]["layout"] = "other-layout"
    elif damage == "tokens":
        bad["slot_apc"]["num_tokens"] = 1000000
    elif damage == "prefix":
        target = request("other", 513, prefix=[99999] * 513)
    elif damage == "indices":
        bad["slot_apc"]["block_indices"][0][0] += 1
    elif damage == "missing":
        bad["remote_block_ids"].pop()
    elif damage == "routing":
        bad.pop("remote_request_id")
    else:
        bad["remote_block_ids"][0][0] = 0
    target.kv_transfer_params = bad
    before = kv.block_pool.get_num_free_blocks()
    with pytest.raises(ValueError, match="slot APC"):
        connector(kv).get_num_new_matched_tokens(target, 0)
    assert kv.block_pool.get_num_free_blocks() == before
    kv.free(req)


def test_layout_detects_missing_dspark_layers_and_wire_metadata_round_trips():
    assert (
        SlotTransferLayout(config(draft_layers=1)).fingerprint != SlotTransferLayout(config(draft_layers=3)).fingerprint
    )
    metadata = MooncakeAgentMetadata(
        engine_id="producer",
        te_rpc_port=7777,
        block_size=8,
        kv_caches_base_addr=[1000],
        num_blocks=32,
        block_lens=[133],
        ssm_sizes=(0, 0),
        slot_apc_version=SLOT_TRANSFER_VERSION,
        slot_apc_layout="layout",
        slot_apc_addr_groups=[[0, 1]],
    )
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(metadata), type=MooncakeAgentMetadata) == metadata


def test_shared_raw_views_keep_offsets_padding_and_deduplicate_registration():
    storage = torch.zeros(2 * 133 + 64, dtype=torch.uint8)
    pages = storage[17 : 17 + 2 * 133].view(2, 133)
    pointers, lengths, strides, groups = raw_slot_layout({0: [pages, pages], 1: [pages]}, 2)
    assert pointers == [storage.data_ptr() + 17]
    assert lengths == [266] and strides == [133] and groups == [[0, 1]]


@pytest.mark.parametrize("damage", ["remote_bound", "local_bound", "null", "mapping", "groups"])
def test_receive_rejects_invalid_physical_mappings_before_rdma(damage):
    source, target = kv_manager(), kv_manager()
    receiver, _, _ = byte_receiver(source, target)
    remote_ids = [[1] for _ in receiver.kv_cache_specs]
    local_ids = [[2] for _ in receiver.kv_cache_specs]
    if damage == "remote_bound":
        remote_ids[0][0] = source.block_pool.num_gpu_blocks
    elif damage == "local_bound":
        local_ids[0][0] = target.block_pool.num_gpu_blocks
    elif damage == "null":
        local_ids[0][0] = 0
    elif damage == "mapping":
        remote_ids[0].append(3)
    else:
        remote_ids.pop()
    with pytest.raises(ValueError, match="slot APC"):
        receiver._transfer_kv_cache_all_groups(
            dict(
                remote_request_id="source",
                remote_block_ids=remote_ids,
                local_block_ids=local_ids,
                remote_engine_id="producer",
                remote_host="127.0.0.1",
                remote_handshake_port=12345,
            )
        )
    receiver.engine.batch_transfer_sync_read.assert_not_called()


def test_unaligned_import_cannot_invent_historical_compressor_state():
    source, source_req, _, params = producer(511)
    target = kv_manager()
    req = request("target", 512)
    req.kv_transfer_params = params
    recv = connector(target)
    recv.get_num_new_matched_tokens(req, 0)
    target.allocate_slots(req, 0, num_external_computed_tokens=511, delay_cache_blocks=True)
    recv.update_state_after_alloc(req, target.get_blocks(req.request_id), 511)
    # Mark the exact load complete; no state for boundary 384 was imported.
    target.on_remote_cache_ready(req, 511)
    assert target.coordinator.single_type_managers[0]._num_cached_slots[req.request_id] == 3
    assert target.get_computed_blocks(request("probe", 512))[1] == 0
    target.free(req)
    source.free(source_req)


def test_receive_failure_cannot_be_reported_as_success():
    receiver = KVCacheRecvingThread.__new__(KVCacheRecvingThread)
    receiver.slot_apc = receiver.use_hybrid = True
    receiver._slot_transfer_error = None
    receiver.task_tracker = KVCacheTaskTracker()
    receiver.task_tracker.add_req_to_process("target")
    receiver._transfer_kv_cache_all_groups = Mock(side_effect=RuntimeError("RDMA failed"))
    receiver._send_done_signal_to_free_remote_port = Mock()
    receiver._send_done_recv_signal = Mock()
    receiver._mark_request_task_done = Mock(return_value=True)
    receiver.proc_not_transfer_request_lock = threading.Lock()
    receiver.proc_not_transfer_request = {}
    receiver.request_queue = queue.Queue()
    receiver.request_queue.put(None)
    receiver._handle_request(
        dict(
            request_id="target",
            remote_request_id="source",
            remote_host="127.0.0.1",
            remote_handshake_port=12345,
            remote_port_send_num={},
            all_task_done=True,
            local_block_ids=([7],),
        )
    )
    assert not receiver.task_tracker.finished_requests
    with pytest.raises(RuntimeError, match="RDMA failed"):
        receiver.get_and_clear_finished_requests()
    receiver._send_done_recv_signal.assert_called_once()


def test_late_or_partial_remote_completion_fails_closed():
    kv = kv_manager()
    req = request("target", 513)
    kv.allocate_slots(req, 0, num_external_computed_tokens=512, delay_cache_blocks=True)
    with pytest.raises(RuntimeError, match="unpaired or partial"):
        kv.on_remote_cache_ready(req, 128)
    assert kv.get_computed_blocks(request("probe", 513))[1] == 0
    kv.free(req)
    with pytest.raises(RuntimeError, match="unpaired or partial"):
        kv.on_remote_cache_ready(req, 512)
    assert not kv.coordinator._pending_external


def test_remote_request_cannot_resume_before_receive_completion():
    kv = kv_manager()
    req = request("target", 513)
    req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    kv.allocate_slots(req, 0, num_external_computed_tokens=512, delay_cache_blocks=True)
    req.num_computed_tokens = 512
    scheduler = SimpleNamespace(finished_recving_kv_req_ids=set(), _update_waiting_for_remote_kv=Mock())
    assert not Scheduler._try_promote_blocked_waiting_request(scheduler, req)
    scheduler._update_waiting_for_remote_kv.assert_not_called()
    assert kv.get_computed_blocks(request("probe", 513))[1] == 0
    kv.free(req)


def test_producer_rejects_decode_that_could_recycle_exported_prompt_state():
    kv = kv_manager()
    req = request("producer", 513)
    req.kv_transfer_params = {"do_remote_decode": True}
    req.sampling_params.max_tokens = 16
    with pytest.raises(ValueError, match="max_tokens=1"):
        connector(kv).get_num_new_matched_tokens(req, 0)


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_remote_admission_reserves_private_tail_at_full_compressed_page_boundary(block_size):
    kv = kv_manager(block_size=block_size)
    seed = request("seed", 128, hash_size=block_size // 16)
    kv.allocate_slots(seed, 128)
    finish(kv, seed, 128)
    req = request("target", 513, hash_size=block_size // 16)
    blocks, hit = kv.get_computed_blocks(req)
    assert hit == 128
    required = kv.coordinator.get_num_blocks_to_allocate(req.request_id, 512, blocks.blocks, 0, 512, 512)
    # Keep the seed's request references so pressure cannot evict hit blocks.
    # Leave one fewer free block than admission requires (including COW).
    held = kv.block_pool.get_new_blocks(kv.block_pool.get_num_free_blocks() - required + 1)
    before = kv.block_pool.get_num_free_blocks()
    assert (
        kv.allocate_slots(
            req,
            0,
            num_new_computed_tokens=hit,
            new_computed_blocks=blocks,
            num_external_computed_tokens=512 - hit,
            delay_cache_blocks=True,
        )
        is None
    )
    assert kv.block_pool.get_num_free_blocks() == before
    assert not kv.coordinator._pending_external and not kv.take_block_copies()
    kv.block_pool.free_blocks(held[:1])
    assert (
        kv.allocate_slots(
            req,
            0,
            num_new_computed_tokens=hit,
            new_computed_blocks=blocks,
            num_external_computed_tokens=512 - hit,
            delay_cache_blocks=True,
        )
        is not None
    )
    assert not kv.take_block_copies()
    kv.block_pool.free_blocks(held[1:])
    kv.free(seed)
    kv.free(req)
    assert kv.block_pool.get_num_free_blocks() == kv.block_pool.num_gpu_blocks - 1


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_full_local_hit_still_releases_remote_without_copying(block_size):
    source_kv, source_req, _, params = producer(512, block_size=block_size)
    target_kv = kv_manager(block_size=block_size)
    seed = request("seed", 512, hash_size=block_size // 16)
    target_kv.allocate_slots(seed, 512)
    finish(target_kv, seed, 512)
    target_kv.free(seed)
    req = request("target", 513, hash_size=block_size // 16)
    req.kv_transfer_params = params
    recv = connector(target_kv)
    blocks, hit = target_kv.get_computed_blocks(req)
    assert hit == 512 and recv.get_num_new_matched_tokens(req, hit) == (0, False)
    target_kv.allocate_slots(req, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks)
    recv.update_state_after_alloc(req, target_kv.get_blocks(req.request_id), 0)
    meta = recv.build_connector_meta(SchedulerOutput.make_empty()).requests[req.request_id]
    assert meta.local_block_ids == [] and meta.remote_block_ids == []
    # No remote load: ordinary local C128 partial-hit COW is still required.
    step, copies = dispatch(target_kv, (req, 513))
    assert len(copies) == 1
    target_kv.on_step_completed(step)
    target_kv.free(req)
    source_kv.free(source_req)
