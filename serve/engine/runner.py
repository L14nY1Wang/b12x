"""Model runner — executes forward passes through the transformer stack.

Manages workspace pools, pre-allocates static buffers for CUDA graph
capture, and provides prefill/decode entry points.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F


from serve.cache.kv_cache import KVCacheManager
from serve.cache.page_pool import PagePool
from serve.model.attention import B12xPagedAttention
from serve.model.loader import LoadedModel
from serve.model.ops import rms_norm


class ModelRunner:
    """Runs forward passes through the full model stack."""

    def __init__(
        self,
        model: LoadedModel,
        kv_mgr=None,
        device: torch.device | str = "cuda",
        max_batch_size: int = 128,
        max_total_tokens: int = 4096,
        pool: PagePool | None = None,
        ssm_pool=None,
    ):
        self.model = model
        self.kv_mgr = kv_mgr
        self.pool = pool or (kv_mgr.pool if kv_mgr is not None else None)
        self.ssm_pool = ssm_pool
        self.device = torch.device(device)
        self.cfg = model.config
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens

        # Inject shared workspace pools and bind per-layer cache refs.
        from b12x.integration.attention import allocate_paged_attention_workspace_pool
        from b12x.integration.tp_moe import allocate_tp_moe_workspace_pool
        attn_workspace = allocate_paged_attention_workspace_pool()
        moe_workspace = allocate_tp_moe_workspace_pool()

        kv_layer_idx = 0
        ssm_layer_idx = 0
        layer_types = getattr(model.config, 'layer_types', None)

        for i, layer in enumerate(model.layers):
            attn = getattr(layer, 'attn', None)
            if isinstance(attn, B12xPagedAttention):
                attn.set_workspace(attn_workspace)
            layer.set_moe_workspace(moe_workspace)

            # Bind cache refs so layers own their slice.
            lt = layer_types[i] if layer_types else None
            if lt == "linear_attention" and ssm_pool is not None:
                layer.bind_cache(
                    ssm_state=ssm_pool.ssm_state_for_layer(ssm_layer_idx),
                    conv_state=ssm_pool.conv_state_for_layer(ssm_layer_idx),
                )
                ssm_layer_idx += 1
            else:
                layer.bind_cache(
                    k_cache=self.pool.k_cache[kv_layer_idx],
                    v_cache=self.pool.v_cache[kv_layer_idx],
                )
                kv_layer_idx += 1

        # CUDA graph pool (lazily populated).
        self._graph_pool = None

    def prefill(
        self,
        token_ids: torch.Tensor,
        request_ids: list[int],
        q_seqlens: list[int],
    ) -> torch.Tensor:
        """Run prefill for one or more requests. Returns next-token logits.

        token_ids: [total_tokens] int64, concatenated prompt tokens.
        request_ids: which requests these tokens belong to.
        q_seqlens: number of tokens per request.
        """
        # KV management: either scheduler does it (kv_mgr=None) or we do.
        if self.kv_mgr is not None:
            for rid, qlen in zip(request_ids, q_seqlens):
                if rid not in self.kv_mgr._requests:
                    self.kv_mgr.allocate_request(rid)
                self.kv_mgr.extend_request(rid, qlen)

        return self._forward(token_ids, request_ids, q_seqlens, mode="extend")

    def compile_model(self, mode: str = "max-autotune-no-cudagraphs") -> None:
        """Apply torch.compile to each layer's forward pass.

        b12x kernel calls within each layer are wrapped with
        @torch.compiler.disable so dynamo doesn't trace into them.
        Compiling per-layer means CapturedGraph._run_forward() calls
        the compiled layers directly.

        """
        for i in range(len(self.model.layers)):
            self.model.layers[i] = torch.compile(self.model.layers[i], mode=mode)

    def capture_decode_graphs(
        self,
        batch_sizes: list[int] | None = None,
        prefill_chunk_size: int | None = None,
    ) -> None:
        """Capture CUDA graphs for decode and optionally prefill."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        from serve.engine.cuda_graph import GraphPool
        self._graph_pool = GraphPool(
            self.model, self.pool, self.device,
            batch_sizes=batch_sizes,
            prefill_chunk_size=prefill_chunk_size,
        )

    def warmup(
        self,
        batch_sizes: list[int] | None = None,
        prefill_lengths: list[int] | None = None,
    ) -> None:
        """Pre-compile kernels for common shapes.

        Runs dummy prefill + decode at each (batch_size, prefill_length)
        combination to trigger CuTe DSL kernel compilation upfront.
        """
        if batch_sizes is None:
            batch_sizes = [1]
        if prefill_lengths is None:
            # Cover common prompt lengths: short, medium (chat template), long.
            prefill_lengths = [4, 64, 256]

        import time
        t0 = time.time()

        for plen in prefill_lengths:
            # Prefill at various lengths to compile attention for those shapes.
            dummy_ids = torch.ones(plen, dtype=torch.long, device=self.device)
            self.prefill(dummy_ids, request_ids=[10000], q_seqlens=[plen])
            # One decode step.
            dummy_tok = torch.ones(1, dtype=torch.long, device=self.device)
            self.decode(dummy_tok, request_ids=[10000])
            self.kv_mgr.free_request(10000)

        for bs in batch_sizes:
            if bs == 1:
                continue  # Already covered above.
            for i in range(bs):
                dummy_ids = torch.ones(4, dtype=torch.long, device=self.device)
                self.prefill(dummy_ids, request_ids=[10000 + i], q_seqlens=[4])
            rids = [10000 + i for i in range(bs)]
            for _ in range(2):
                dummy_tok = torch.ones(bs, dtype=torch.long, device=self.device)
                self.decode(dummy_tok, request_ids=rids)
            for i in range(bs):
                self.kv_mgr.free_request(10000 + i)

        torch.cuda.synchronize()
        print(f"Warmup complete ({time.time() - t0:.1f}s)")

    def decode(
        self,
        token_ids: torch.Tensor,
        request_ids: list[int],
    ) -> torch.Tensor:
        """Run one decode step for all active requests. Returns next-token logits.

        token_ids: [batch] int64, one token per request.
        """
        # Allocate 1 new KV slot per request.
        for rid in request_ids:
            self.kv_mgr.extend_request(rid, 1)

        bs = len(request_ids)
        if self._graph_pool is not None:
            graph = self._graph_pool.get(bs)
            if graph is not None:
                return self._replay_graph(graph, token_ids, request_ids)

        q_seqlens = [1] * bs
        return self._forward(token_ids, request_ids, q_seqlens, mode="decode")

    def _replay_graph(self, graph, token_ids, request_ids):
        """Replay a captured CUDA graph."""
        device = self.device
        page_table = self.kv_mgr.build_page_table(request_ids, device=device)
        cache_seqlens = self.kv_mgr.build_cache_seqlens(request_ids, device=device)
        pre_write = cache_seqlens - 1
        positions = pre_write.long()
        return graph.replay(token_ids, page_table, cache_seqlens, pre_write, positions)

    def forward_batch(
        self,
        token_ids: torch.Tensor,
        q_seqlens: list[int],
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        mode: str,
        graph_bs: int | None = None,
        ssm_cache_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with pre-built batch tensors (scheduler-driven)."""
        device = self.device
        bs = cache_seqlens.shape[0]

        if graph_bs is not None and self._graph_pool is not None:
            if mode == "decode":
                graph = self._graph_pool.get(graph_bs)
            else:
                graph = self._graph_pool.get_prefill(graph_bs)
            if graph is not None:
                q_lens_t = torch.tensor(q_seqlens, dtype=torch.int32, device=device)
                pre_write = cache_seqlens - q_lens_t
                positions = _build_positions(q_seqlens, pre_write, device)
                return graph.replay(token_ids, page_table, cache_seqlens, pre_write, positions,
                                    ssm_cache_indices=ssm_cache_indices)

        # Build cu_seqlens_q only for eager path.
        q_lens = torch.tensor(q_seqlens, dtype=torch.int32, device=device)
        cu_seqlens_q = torch.zeros(len(q_seqlens) + 1, dtype=torch.int32, device=device)
        cu_seqlens_q[1:] = q_lens.cumsum(0)

        return self._forward_inner(token_ids, q_seqlens, page_table, cache_seqlens,
                                   cu_seqlens_q, mode, ssm_cache_indices)

    def _forward(
        self,
        token_ids: torch.Tensor,
        request_ids: list[int],
        q_seqlens: list[int],
        mode: str,
    ) -> torch.Tensor:
        """Forward pass with KVCacheManager-built tensors (warmup/legacy)."""
        cfg = self.cfg
        device = self.device

        page_table = self.kv_mgr.build_page_table(request_ids, device=device)
        cache_seqlens = self.kv_mgr.build_cache_seqlens(request_ids, device=device)
        cu_seqlens_q = self.kv_mgr.build_cu_seqlens_q(q_seqlens, device=device)
        return self._forward_inner(token_ids, q_seqlens, page_table, cache_seqlens, cu_seqlens_q, mode)

    def _forward_inner(self, token_ids, q_seqlens, page_table, cache_seqlens,
                        cu_seqlens_q, mode, ssm_cache_indices=None):
        """Core forward pass — unified for all model types."""
        from serve.engine.step_state import StepState
        cfg = self.cfg
        device = self.device

        pre_write_seqlens = cache_seqlens - torch.tensor(
            q_seqlens, dtype=torch.int32, device=device
        )
        positions = _build_positions(q_seqlens, pre_write_seqlens, device)
        hidden = self.model.embed_tokens(token_ids.to(device))

        layer_trace_path = os.environ.get("B12X_LAYER_STDS_PATH")
        layer_stds = None
        if layer_trace_path and mode != "decode":
            layer_stds = [float(hidden.float().std().item())]

        # Build MambaForwardMetadata if SSM indices are provided.
        mamba_meta = None
        if ssm_cache_indices is not None:
            from serve.engine.mamba_metadata import MambaForwardMetadata
            is_decode = (mode == "decode")
            mamba_meta = MambaForwardMetadata(
                cache_indices=ssm_cache_indices,
                has_initial_states=torch.ones(ssm_cache_indices.shape[0], dtype=torch.bool, device=device)
                    if is_decode
                    else (pre_write_seqlens[:ssm_cache_indices.shape[0]] > 0),
                cu_seqlens=cu_seqlens_q if not is_decode else None,
                seq_lens=q_seqlens if not is_decode else None,
            )

        state = StepState(
            cos=self.model.cos,
            sin=self.model.sin,
            positions=positions,
            page_table=page_table,
            cache_seqlens=pre_write_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            mamba=mamba_meta,
            is_decode=(mode == "decode"),
        )

        for layer in self.model.layers:
            hidden = layer(hidden, state)
            if layer_stds is not None:
                layer_stds.append(float(hidden.float().std().item()))

        hidden = rms_norm(hidden, self.model.final_norm_weight, cfg.rms_norm_eps,
                          gemma_style=getattr(cfg, 'gemma_norm', False))
        logits = F.linear(hidden, self.model.lm_head_weight)

        if layer_stds is not None:
            try:
                import torch.distributed as dist
                if (not dist.is_initialized()) or dist.get_rank() == 0:
                    with open(layer_trace_path, "w") as f:
                        json.dump(layer_stds, f)
            except Exception:
                pass

        # Return only the last token's logits per request.
        if all(s == 1 for s in q_seqlens):
            return logits  # Decode: one token per request, all are "last".
        cu = cu_seqlens_q[1:] - 1  # Last index per request.
        return logits[cu.long()]


def _build_positions(
    q_seqlens: list[int],
    pre_write_seqlens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build position tensor for RoPE.

    For each request, positions are [cache_len, cache_len+1, ..., cache_len+q_len-1].
    """
    total_q = sum(q_seqlens)
    if len(q_seqlens) == 1:
        # Single request — simple arange. Graph-safe.
        return (pre_write_seqlens[0] + torch.arange(total_q, device=device)).long()

    q_lens_t = torch.tensor(q_seqlens, device=device, dtype=torch.int32)
    batch_ids = torch.repeat_interleave(
        torch.arange(len(q_seqlens), device=device), q_lens_t.long()
    )
    cu = torch.zeros(len(q_seqlens) + 1, device=device, dtype=torch.int32)
    cu[1:] = q_lens_t.cumsum(0)
    offsets = torch.arange(total_q, device=device) - cu[batch_ids]
    return (pre_write_seqlens[batch_ids] + offsets).long()
