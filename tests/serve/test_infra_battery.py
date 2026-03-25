"""Infrastructure-focused stress tests for the serving stack.

These tests target failure handling, cleanup paths, and invariants that are
easy to miss in happy-path request/response coverage.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from serve.cache.prefix_checkpoint_cache import PrefixCheckpointCache
from serve.engine.request import Request
from serve.engine.sampling import SamplingParams
from serve.engine.scheduler import BatchScheduler
from serve.engine.serving import ServingEngine


class MockPagePool:
    def __init__(self, num_pages=64):
        self.num_pages = num_pages
        self._free = list(range(num_pages))

    def alloc(self, n):
        if n > len(self._free):
            raise RuntimeError("OOM")
        result = self._free[-n:]
        del self._free[-n:]
        return result

    def free(self, page_ids):
        self._free.extend(page_ids)

    @property
    def num_free(self):
        return len(self._free)


def _make_scheduler(num_pages=64, max_running=8, chunk_size=64):
    pool = MockPagePool(num_pages)
    cache = PrefixCheckpointCache(pool)
    return BatchScheduler(
        cache=cache,
        pool=pool,
        captured_bs=[1, 2, 4, 8],
        max_running=max_running,
        max_prefill_tokens=4096,
        chunk_size=chunk_size,
        device="cpu",
    )


def _make_request(rid, prompt_len=64, *, max_new_tokens=8, timeout_s=None):
    return Request(
        rid=rid,
        prompt_ids=list(range(rid * 1000, rid * 1000 + prompt_len)),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        timeout_s=timeout_s,
    )


def _make_minimal_engine(scheduler: BatchScheduler) -> ServingEngine:
    engine = ServingEngine.__new__(ServingEngine)
    engine.scheduler = scheduler
    engine._lock = threading.Lock()
    engine._work_event = threading.Event()
    engine._loop_running = False
    engine._loop_error = None
    engine._next_rid = 0
    engine.world_size = 1
    engine.device = "cpu"
    tokenizer = MagicMock()
    tokenizer.eos_token_id = None
    tokenizer.unk_token_id = -1
    tokenizer.convert_tokens_to_ids.return_value = -1
    engine.tokenizer = tokenizer
    return engine


def _make_running_request_with_visible_output(sched: BatchScheduler, rid: int = 1) -> Request:
    req = _make_request(rid, prompt_len=64, max_new_tokens=8)
    sched.add_request(req)
    while req not in sched.running:
        batch = sched.step()
        assert batch is not None
        assert batch.mode == "prefill"
        if batch.is_last_chunk:
            sched.process_prefill_chunk([42], batch.requests)
        else:
            sched.process_prefill_chunk(None)
    assert req.output_ids == [42]
    assert req in sched.running
    return req


def test_cancelled_running_request_restores_page_budget_without_caching():
    sched = _make_scheduler(num_pages=2)
    req = _make_running_request_with_visible_output(sched)

    req.cancel()
    batch = sched.step()

    assert batch is None
    assert req.finished_reason == "cancelled"
    assert sched.cache.total_cached_pages == 0
    assert sched.pool.num_free == sched.pool.num_pages
    assert sched.num_running == 0


def test_timed_out_prefilling_request_restores_page_budget_without_caching():
    sched = _make_scheduler(num_pages=2, chunk_size=32)
    req = _make_request(1, prompt_len=100, timeout_s=0.001)
    sched.add_request(req)

    batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"
    assert not batch.is_last_chunk
    sched.process_prefill_chunk(None)
    assert req in sched.prefilling
    assert sched.pool.num_free == 0

    time.sleep(0.01)
    batch = sched.step()

    assert batch is None
    assert req.finished_reason == "timeout"
    assert sched.cache.total_cached_pages == 0
    assert sched.pool.num_free == sched.pool.num_pages
    assert sched.num_prefilling == 0


def test_fail_all_releases_mixed_request_states_without_creating_checkpoints():
    sched = _make_scheduler(num_pages=8, chunk_size=32)

    running = _make_running_request_with_visible_output(sched, rid=1)

    prefilling = _make_request(2, prompt_len=100, max_new_tokens=4)
    sched.add_request(prefilling)
    batch = sched.step()
    assert batch is not None
    if batch.mode == "decode":
        sched.process_decode_output([99])
        batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"
    assert not batch.is_last_chunk
    sched.process_prefill_chunk(None)
    assert prefilling in sched.prefilling

    waiting = _make_request(3, prompt_len=16, max_new_tokens=2)
    sched.add_request(waiting)

    sched.fail_all("engine_error")

    for req in (running, prefilling, waiting):
        assert req.finished_reason == "engine_error"
        assert req._done_event.is_set()
    assert sched.cache.total_cached_pages == 0
    assert sched.pool.num_free == sched.pool.num_pages
    assert sched.num_running == 0
    assert sched.num_prefilling == 0
    assert sched.num_waiting == 0


def test_server_loop_crash_marks_engine_unhealthy_and_fails_inflight_work():
    sched = _make_scheduler()
    req = _make_request(1, prompt_len=16, max_new_tokens=2)
    sched.add_request(req)
    engine = _make_minimal_engine(sched)

    crashed = threading.Event()

    def crash_step():
        crashed.set()
        raise RuntimeError("synthetic crash")

    engine._step = crash_step
    engine.start_server_loop()

    assert crashed.wait(timeout=1.0)
    engine._loop_thread.join(timeout=1.0)
    assert not engine._loop_thread.is_alive()

    health = engine.server_loop_health()
    assert health["healthy"] is False
    assert health["alive"] is False
    assert "synthetic crash" in health["last_error"]

    assert req.finished_reason == "engine_error"
    assert req._done_event.is_set()
    assert sched.num_waiting == 0


def test_submit_rejects_new_work_after_server_loop_crash():
    sched = _make_scheduler()
    engine = _make_minimal_engine(sched)
    engine._loop_error = "RuntimeError: synthetic crash"

    with pytest.raises(RuntimeError, match="server loop unhealthy"):
        engine.submit([1, 2, 3], SamplingParams(max_new_tokens=4))

    assert engine._next_rid == 0
    assert sched.num_waiting == 0


def test_abort_paths_do_not_contaminate_subsequent_prefix_cache_reuse():
    sched = _make_scheduler(num_pages=4)
    shared_prompt = list(range(2000, 2128))

    cancelled = _make_running_request_with_visible_output(sched, rid=1)
    cancelled.cancel()
    assert sched.step() is None
    assert sched.cache.total_cached_pages == 0

    good = Request(
        rid=2,
        prompt_ids=list(shared_prompt),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(good)
    while not good.is_finished:
        batch = sched.step()
        assert batch is not None
        assert batch.mode == "prefill"
        if batch.is_last_chunk:
            sched.process_prefill_chunk([77], batch.requests)
        else:
            sched.process_prefill_chunk(None)
    assert good.finished_reason == "length"
    assert sched.cache.total_cached_pages == 2

    follower = Request(
        rid=3,
        prompt_ids=list(shared_prompt),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(follower)
    batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"
    assert follower.checkpoint_len == 128
    assert batch.q_seqlens == [1]
