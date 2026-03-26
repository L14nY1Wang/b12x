"""Top-level serving engine with TP coordination.

The ServingEngine is the single abstraction that both CLI and API use.
It handles model loading, warmup, TP broadcast, chat templates, and
generation using the BatchScheduler + PrefixCheckpointCache.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from transformers import AutoTokenizer

from serve.cache.page_pool import PagePool
from serve.cache.prefix_checkpoint_cache import PrefixCheckpointCache
from serve.engine.request import Request
from serve.engine.sampling import SamplingParams, sample_batch
from serve.engine.scheduler import BatchScheduler
from serve.model.loader import load_model
from serve.tp.group import TPGroup


@dataclass
class GenerationResult:
    """Public result returned to callers."""
    request_id: int
    prompt_ids: list[int]
    generated_ids: list[int]
    finished: bool = False
    finish_reason: str | None = None
    time_to_first_token_ms: float = 0.0
    total_time_ms: float = 0.0


def _should_enable_prefix_cache(
    *,
    is_hybrid: bool,
    world_size: int,
    has_state_snapshot_slots: bool,
) -> bool:
    """Decide whether prefix-cache reuse is safe for the current engine mode."""
    if not is_hybrid:
        return True
    if world_size > 1:
        return False
    return has_state_snapshot_slots


def _should_enable_layer_compile(
    *,
    is_hybrid: bool,
    compile_layers: bool,
) -> bool:
    """Decide whether per-layer torch.compile should run."""
    if not compile_layers:
        return False
    return not is_hybrid


class ServingEngine:
    """Unified serving engine for all ranks."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        tp_group: Optional[TPGroup] = None,
        load_backend: str = "auto",
        kv_dtype: torch.dtype = torch.bfloat16,
        warmup_prefill_lengths: list[int] | None = None,
        graph_batch_sizes: list[int] | None = None,
        prefill_chunk_size: int = 512,
        capture_prefill_graph: bool = False,
        compile_layers: bool = False,
    ):
        torch.set_grad_enabled(False)
        self.tp_group = tp_group
        self.rank = tp_group.rank if tp_group else 0
        self.world_size = tp_group.world_size if tp_group else 1
        self.device = device
        self.model_path = model_path
        self._next_rid = 0
        self._lock = threading.Lock()  # Guards scheduler + _next_rid.
        self._loop_running = False
        self._work_event = threading.Event()  # Wakes the server loop on submit.
        self._loop_error: str | None = None

        if self.rank == 0:
            print(f"Loading model (TP={self.world_size})...")

        self.model = load_model(
            model_path,
            device=device,
            tp_group=tp_group,
            load_backend=load_backend,
        )
        self.cfg = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            import pathlib
            jinja_path = pathlib.Path(model_path) / "chat_template.jinja"
            if jinja_path.exists():
                self.tokenizer.chat_template = jinja_path.read_text()

        # KV cache — only for self-attention layers.
        layer_types = getattr(self.cfg, 'layer_types', None)
        num_kv_layers = self.cfg.num_layers
        if layer_types:
            num_kv_layers = sum(1 for t in layer_types if t == "attention")

        total_mem = torch.cuda.mem_get_info()[1]
        used_mem = total_mem - torch.cuda.mem_get_info()[0]
        # KV cache + model weights together should occupy mem_fraction of total GPU memory.
        mem_fraction = 0.75
        kv_budget = max(0, int(total_mem * mem_fraction) - used_mem)

        # All ranks must agree on num_pages for TP broadcast shapes.
        if self.world_size > 1:
            budget_t = torch.tensor([kv_budget], dtype=torch.long, device=device)
            dist.all_reduce(budget_t, op=dist.ReduceOp.MIN)
            kv_budget = budget_t.item()

        # Linear-attention state arena shared by live requests and cached
        # terminal checkpoints.
        linear_state_arena = None
        live_ssm_slots = 0
        if layer_types and any(t == "linear_attention" for t in layer_types):
            from serve.cache.linear_state_arena import LinearStateArena
            from serve.cache.tensor_arena import TensorArena

            num_linear_layers = sum(1 for t in layer_types if t == "linear_attention")
            live_ssm_slots = max(graph_batch_sizes or [8])
            num_heads = self.cfg.linear_num_v_heads
            head_v_dim = self.cfg.linear_head_v_dim
            head_k_dim = self.cfg.linear_head_k_dim
            conv_dim = (
                self.cfg.linear_num_k_heads * self.cfg.linear_head_k_dim * 2
                + self.cfg.linear_num_v_heads * self.cfg.linear_head_v_dim
            )
            conv_kernel = self.cfg.linear_conv_kernel
            live_ssm_bytes = TensorArena.estimate_memory_bytes(
                num_slots=live_ssm_slots,
                num_linear_layers=num_linear_layers,
                num_heads=num_heads,
                head_v_dim=head_v_dim,
                head_k_dim=head_k_dim,
                conv_dim=conv_dim,
                conv_kernel=conv_kernel,
            )
            kv_budget = max(0, kv_budget - live_ssm_bytes)

            slot_bytes = TensorArena.slot_memory_bytes_for_shape(
                num_linear_layers=num_linear_layers,
                num_heads=num_heads,
                head_v_dim=head_v_dim,
                head_k_dim=head_k_dim,
                conv_dim=conv_dim,
                conv_kernel=conv_kernel,
            )
            max_cached_snapshot_slots = 64
            snapshot_slots = 0
            snapshot_budget = kv_budget // 8
            if slot_bytes > 0 and snapshot_budget >= 2 * slot_bytes:
                snapshot_slots = min(
                    max_cached_snapshot_slots,
                    max(0, snapshot_budget // slot_bytes - 1),
                )

            linear_state_arena = LinearStateArena(
                live_slots=live_ssm_slots,
                snapshot_slots=snapshot_slots,
                num_linear_layers=num_linear_layers,
                num_heads=num_heads,
                head_v_dim=head_v_dim,
                head_k_dim=head_k_dim,
                conv_dim=conv_dim,
                conv_kernel=conv_kernel,
                device=device,
            )
            snapshot_ssm_bytes = (
                TensorArena.estimate_memory_bytes(
                    num_slots=snapshot_slots,
                    num_linear_layers=num_linear_layers,
                    num_heads=num_heads,
                    head_v_dim=head_v_dim,
                    head_k_dim=head_k_dim,
                    conv_dim=conv_dim,
                    conv_kernel=conv_kernel,
                )
                if snapshot_slots > 0
                else 0
            )
            kv_budget = max(0, kv_budget - snapshot_ssm_bytes)
            if self.rank == 0:
                print(
                    f"LinearStateArena: {live_ssm_slots} live + "
                    f"{snapshot_slots} checkpoint slots, "
                    f"{linear_state_arena.memory_bytes() / 1e6:.0f} MB"
                )

        self.pool = PagePool(
            num_pages=PagePool.estimate_num_pages(
                kv_budget,
                num_layers=num_kv_layers,
                kv_heads=self.cfg.num_kv_heads,
                head_dim=self.cfg.head_dim,
                kv_dtype=kv_dtype,
            ),
            num_layers=num_kv_layers,
            kv_heads=self.cfg.num_kv_heads,
            head_dim=self.cfg.head_dim,
            kv_dtype=kv_dtype,
            device=device,
        )
        self.cache = PrefixCheckpointCache(
            self.pool,
            state_arena=linear_state_arena,
        )

        # Model runner.
        from serve.engine.runner import ModelRunner
        from serve.cache.kv_cache import KVCacheManager
        warmup_kv_mgr = KVCacheManager(self.pool)
        self.runner = ModelRunner(self.model, warmup_kv_mgr, device=device,
                                 pool=self.pool, ssm_pool=linear_state_arena)

        if self.rank == 0:
            print(f"KV cache: {self.pool.num_pages} pages ({self.pool.num_pages * 64} tokens)")

        is_hybrid = layer_types is not None and any(t == "linear_attention" for t in layer_types)

        graph_sizes = [1, 2, 4, 8] if graph_batch_sizes is None else graph_batch_sizes
        enable_layer_compile = _should_enable_layer_compile(
            is_hybrid=is_hybrid,
            compile_layers=compile_layers,
        )
        if self.rank == 0 and compile_layers and not enable_layer_compile:
            print("Layer compile disabled for hybrid models.")

        if not is_hybrid:
            # Warmup uses kv_mgr-based path which doesn't support SSM layers.
            if self.rank == 0:
                print("Warming up...")
            prefill_lengths = warmup_prefill_lengths or [4, 64]
            self.runner.warmup(batch_sizes=[1], prefill_lengths=prefill_lengths)

        if enable_layer_compile:
            if self.rank == 0:
                print("Compiling...")
            self.runner.compile_model()

        if enable_layer_compile:
            self.runner.warmup(batch_sizes=[1], prefill_lengths=[4, prefill_chunk_size])

        if graph_sizes:
            if self.rank == 0:
                print("Capturing CUDA graphs...")
            self.runner.capture_decode_graphs(
                batch_sizes=graph_sizes,
                prefill_chunk_size=prefill_chunk_size if capture_prefill_graph else None,
            )

        # Warmup/compile/graph capture all touch shared runtime state. Reset it
        # before the scheduler starts serving real requests.
        if linear_state_arena is not None:
            linear_state_arena.zero_all()
        for i in range(len(self.pool.k_cache)):
            self.pool.k_cache[i].zero_()
            self.pool.v_cache[i].zero_()

        # Scheduler.
        hybrid_prefix_cache = _should_enable_prefix_cache(
            is_hybrid=is_hybrid,
            world_size=self.world_size,
            has_state_snapshot_slots=(
                linear_state_arena is not None
                and linear_state_arena.num_snapshot_slots > 0
            ),
        )
        if self.rank == 0 and is_hybrid and self.world_size > 1:
            print("Hybrid prefix cache disabled under TP until SSM snapshots are mirrored across ranks.")

        self.scheduler = BatchScheduler(
            cache=self.cache,
            pool=self.pool,
            ssm_pool=linear_state_arena,
            enable_prefix_cache=hybrid_prefix_cache,
            captured_bs=graph_sizes,
            max_running=max(graph_sizes) if graph_sizes else 8,
            chunk_size=prefill_chunk_size,
            device=device,
        )

        if self.world_size > 1:
            dist.barrier()

        if self.rank == 0:
            print("Ready.")

    # -- public API (rank 0) -----------------------------------------------

    def submit(
        self,
        input_ids: list[int],
        params: SamplingParams | None = None,
        timeout_s: float | None = None,
    ) -> Request:
        """Submit a request without blocking. Returns the Request object.

        Does NOT broadcast to TP followers — use generate() or
        generate_batch() for TP-safe generation. submit() is intended
        for single-GPU or server-mode use.
        """
        if params is None:
            params = SamplingParams()
        self._ensure_stop_ids(params)
        self._raise_if_loop_unhealthy()

        req = Request(
            rid=self._next_rid,
            prompt_ids=input_ids,
            sampling_params=params,
            timeout_s=timeout_s,
        )
        with self._lock:
            self._next_rid += 1
            self.scheduler.add_request(req)
        self._work_event.set()
        return req

    def chat(
        self,
        messages: list[dict[str, str]],
        params: SamplingParams | None = None,
        **template_kwargs,
    ) -> GenerationResult:
        if params is None:
            params = SamplingParams()
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
        )
        input_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        return self.generate(input_ids, params)

    def complete(
        self,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> GenerationResult:
        if params is None:
            params = SamplingParams()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        return self.generate(input_ids, params)

    def generate(
        self,
        input_ids: list[int],
        params: SamplingParams | None = None,
    ) -> GenerationResult:
        req = self.submit(input_ids, params)

        if self._has_server_loop():
            req._done_event.wait()
        else:
            while not req.is_finished:
                self._step()

        return self._to_result(req)

    def generate_batch(
        self,
        prompts: list[list[int]],
        params: SamplingParams | None = None,
    ) -> list[GenerationResult]:
        """Generate for multiple prompts concurrently."""
        if params is None:
            params = SamplingParams()
        self._ensure_stop_ids(params)

        reqs = [self.submit(ids, params) for ids in prompts]

        if self._has_server_loop():
            for r in reqs:
                r._done_event.wait()
        else:
            while not all(r.is_finished for r in reqs):
                self._step()

        return [self._to_result(r) for r in reqs]

    def generate_stream(
        self,
        input_ids: list[int],
        params: SamplingParams | None = None,
    ):
        req = self.submit(input_ids, params)

        prev_len = 0
        if self._has_server_loop():
            # Server loop is stepping — wait on token events.
            while not req.is_finished:
                req._token_event.wait()
                req._token_event.clear()
                for tok_id in req.output_ids[prev_len:]:
                    text = self.tokenizer.decode([tok_id], skip_special_tokens=True)
                    yield tok_id, text, self._to_result(req)
                prev_len = len(req.output_ids)
        else:
            # CLI mode — drive the step loop ourselves.
            while not req.is_finished:
                self._step()
                if len(req.output_ids) > prev_len:
                    for tok_id in req.output_ids[prev_len:]:
                        text = self.tokenizer.decode([tok_id], skip_special_tokens=True)
                        yield tok_id, text, self._to_result(req)
                    prev_len = len(req.output_ids)

    def chat_stream(self, messages, params=None, **template_kwargs):
        if params is None:
            params = SamplingParams()
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
        )
        input_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        yield from self.generate_stream(input_ids, params)

    # -- server mode (background step loop) --------------------------------

    def start_server_loop(self) -> None:
        """Start a background thread that continuously steps the scheduler.

        Used by the API server so multiple requests can be submitted
        concurrently and the step loop processes them all.
        """
        if hasattr(self, '_loop_thread') and self._loop_thread.is_alive():
            return
        self._loop_error = None
        self._loop_running = True
        self._loop_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._loop_thread.start()

    def stop_server_loop(self) -> None:
        self._loop_running = False
        if hasattr(self, '_loop_thread'):
            self._loop_thread.join(timeout=5.0)

    def _has_server_loop(self) -> bool:
        return (
            self._loop_running
            and hasattr(self, '_loop_thread')
            and self._loop_thread.is_alive()
        )

    def _server_loop(self) -> None:
        """Background loop: step while there's work, wait for signal otherwise."""
        torch.set_grad_enabled(False)
        try:
            while self._loop_running:
                with self._lock:
                    has_work = self.scheduler.has_work
                if has_work:
                    self._step()
                else:
                    self._work_event.wait(timeout=0.1)
                    self._work_event.clear()
        except Exception as exc:
            self._loop_error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.scheduler.fail_all("engine_error")
        finally:
            self._loop_running = False
            self._work_event.set()

    # -- core step ---------------------------------------------------------

    def _step(self) -> None:
        """One scheduling + forward + sample step.

        In TP mode, broadcasts step info to followers before the forward
        pass so they can mirror it.
        """
        with self._lock:
            batch = self.scheduler.step()
        if batch is None:
            if self.world_size > 1:
                self._broadcast_step_idle()
            return

        # Broadcast batch to followers.
        if self.world_size > 1:
            self._broadcast_step(batch)

        # Build SSM cache indices if model has SSM layers.
        ssm_indices = None
        if self.scheduler.ssm_pool is not None:
            ssm_indices = torch.tensor(
                [r.ssm_slot for r in batch.requests],
                dtype=torch.int64, device=self.device,
            )

        logits = self.runner.forward_batch(
            batch.token_ids, batch.q_seqlens,
            batch.page_table, batch.cache_seqlens,
            mode=batch.mode,
            graph_bs=batch.graph_bs,
            ssm_cache_indices=ssm_indices,
        )

        step_log_path = os.environ.get("B12X_STEP_LOG_PATH")
        if step_log_path and batch.mode == "prefill":
            try:
                topk = torch.topk(logits[0].float(), k=10)
                payload = {
                    "mode": batch.mode,
                    "q_seqlens": batch.q_seqlens,
                    "topk_ids": topk.indices.tolist(),
                    "topk_tokens": [
                        self.tokenizer.decode([tok], skip_special_tokens=True)
                        for tok in topk.indices.tolist()
                    ],
                    "topk_logits": [float(x) for x in topk.values.tolist()],
                }
                with open(step_log_path, "w") as f:
                    json.dump(payload, f, indent=2)
            except Exception:
                pass

        if batch.mode == "prefill" and not batch.is_last_chunk:
            with self._lock:
                self.scheduler.process_prefill_chunk(None, batch.requests)
            return

        params_list = [r.sampling_params for r in batch.requests]
        gen_ids = [list(r.output_ids) for r in batch.requests]
        next_tokens = sample_batch(logits, params_list, gen_ids)
        token_list = next_tokens.tolist()

        with self._lock:
            if batch.mode == "prefill":
                self.scheduler.process_prefill_chunk(token_list, batch.requests)
            else:
                self.scheduler.process_decode_output(token_list)

    # -- helpers -----------------------------------------------------------

    def _ensure_stop_ids(self, params: SamplingParams) -> None:
        ids = set(params.stop_token_ids or [])
        # Add all EOS-like tokens.
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        # Also check generation_config for additional EOS tokens.
        gen_cfg = getattr(self.tokenizer, '_tokenizer', None)
        if hasattr(self, '_extra_stop_ids'):
            ids.update(self._extra_stop_ids)
        elif not hasattr(self, '_extra_stop_ids'):
            extra = set()
            # Add <|endoftext|> and <|im_end|> if present.
            for name in ['<|endoftext|>', '<|im_end|>']:
                tok_id = self.tokenizer.convert_tokens_to_ids(name)
                if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                    extra.add(tok_id)
            self._extra_stop_ids = extra
            ids.update(extra)
        params.stop_token_ids = list(ids)

    def _raise_if_loop_unhealthy(self) -> None:
        if self._loop_error is not None:
            raise RuntimeError(f"server loop unhealthy: {self._loop_error}")

    def server_loop_health(self) -> dict:
        loop_alive = hasattr(self, '_loop_thread') and self._loop_thread.is_alive()
        healthy = self._loop_error is None
        return {
            "running": self._loop_running,
            "alive": loop_alive,
            "healthy": healthy,
            "last_error": self._loop_error,
        }

    def _to_result(self, req: Request) -> GenerationResult:
        return GenerationResult(
            request_id=req.rid,
            prompt_ids=req.prompt_ids,
            generated_ids=list(req.output_ids),
            finished=req.is_finished,
            finish_reason=req.finished_reason,
            time_to_first_token_ms=req.ttft_ms or 0.0,
            total_time_ms=req.total_time_ms or 0.0,
        )

    # -- follower loop (non-rank-0) ----------------------------------------

    def run_follower(self) -> None:
        """Mirror rank 0's forward passes. No scheduler or sampling needed."""
        assert self.rank != 0
        while True:
            result = self._receive_step()
            if result == "shutdown":
                break
            if result is None:
                continue  # Idle step — loop back.
            mode, token_ids, q_seqlens, page_table, cache_seqlens, graph_bs, ssm_indices = result
            self.runner.forward_batch(
                token_ids, q_seqlens, page_table, cache_seqlens,
                mode=mode, graph_bs=graph_bs,
                ssm_cache_indices=ssm_indices,
            )

    def shutdown(self) -> None:
        self.stop_server_loop()
        if self.rank == 0 and self.world_size > 1:
            # Send shutdown signal.
            header = torch.tensor([255, 0, 0, 0], dtype=torch.long, device=self.device)
            dist.broadcast(header, src=0)

    # -- TP step-level broadcast -------------------------------------------

    # Header: [mode_code, total_q, bs, graph_bs, page_table_width]
    # mode_code: 0=idle, 1=prefill, 2=decode, 255=shutdown.

    _MODE_IDLE = 0
    _MODE_PREFILL = 1
    _MODE_DECODE = 2
    _MODE_SHUTDOWN = 255

    def _broadcast_step_idle(self) -> None:
        """Tell followers there's no work this step."""
        header = torch.tensor([self._MODE_IDLE, 0, 0, 0, 0],
                              dtype=torch.long, device=self.device)
        dist.broadcast(header, src=0)

    def _broadcast_step(self, batch) -> None:
        """Broadcast batch tensors to followers before forward pass."""
        mode_code = self._MODE_PREFILL if batch.mode == "prefill" else self._MODE_DECODE
        total_q = batch.token_ids.shape[0]
        bs = batch.cache_seqlens.shape[0]
        graph_bs = batch.graph_bs or 0
        page_table_width = batch.page_table.shape[1]

        header = torch.tensor([mode_code, total_q, bs, graph_bs, page_table_width],
                              dtype=torch.long, device=self.device)
        dist.broadcast(header, src=0)

        # Broadcast batch tensors.
        dist.broadcast(batch.token_ids, src=0)
        q_seqlens_t = torch.tensor(batch.q_seqlens, dtype=torch.int32, device=self.device)
        dist.broadcast(q_seqlens_t, src=0)
        dist.broadcast(batch.page_table, src=0)
        dist.broadcast(batch.cache_seqlens, src=0)

        # Broadcast SSM cache indices if hybrid model.
        if self.scheduler.ssm_pool is not None:
            ssm_idx = torch.tensor(
                [r.ssm_slot for r in batch.requests],
                dtype=torch.int64, device=self.device,
            )
            dist.broadcast(ssm_idx, src=0)

    def _receive_step(self):
        """Receive a step broadcast from rank 0.

        Returns "shutdown", None (idle), or (mode, token_ids, ...) tuple.
        """
        header = torch.empty(5, dtype=torch.long, device=self.device)
        dist.broadcast(header, src=0)

        mode_code = header[0].item()
        if mode_code == self._MODE_SHUTDOWN:
            return "shutdown"
        if mode_code == self._MODE_IDLE:
            return None

        total_q = header[1].item()
        bs = header[2].item()
        graph_bs = header[3].item() or None
        page_table_width = header[4].item()
        mode = "prefill" if mode_code == self._MODE_PREFILL else "decode"

        token_ids = torch.empty(total_q, dtype=torch.long, device=self.device)
        dist.broadcast(token_ids, src=0)
        q_seqlens_t = torch.empty(bs, dtype=torch.int32, device=self.device)
        dist.broadcast(q_seqlens_t, src=0)
        q_seqlens = q_seqlens_t.tolist()

        page_table = torch.empty(bs, page_table_width, dtype=torch.int32, device=self.device)
        dist.broadcast(page_table, src=0)
        cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
        dist.broadcast(cache_seqlens, src=0)

        # Receive SSM indices if hybrid model.
        ssm_indices = None
        if self.runner.ssm_pool is not None:
            ssm_indices = torch.empty(bs, dtype=torch.int64, device=self.device)
            dist.broadcast(ssm_indices, src=0)

        return mode, token_ids, q_seqlens, page_table, cache_seqlens, graph_bs, ssm_indices
