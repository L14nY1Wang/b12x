# Qwen3.5 Decode Graph Benchmark Plan

Status as of 2026-04-02:

- `decode-graph-buckets` mode is now implemented in
  `benchmarks/benchmark_paged_attention.py`
- the benchmark defaults to one graph per batch bucket and reuses that graph
  across the runtime context ladder
- capture defaults to the registered decode graph tuning policy when available
- `--check` now reports reference-relative metrics instead of only raw backend
  deltas

## Goal

Make `benchmarks/benchmark_paged_attention.py` model the way Qwen3.5 decode is
actually used in serving:

- one captured graph per decode batch bucket
- one workspace set per decode batch bucket
- graph and workspace reused across many runtime context lengths
- timed region is pure `graph.replay()`

The target batch buckets for this session are:

- `1,2,4,8,12,16`

Batch `32` is intentionally out of the default session scope for now because
its BF16 capture policy is still synthetic and not yet trustworthy enough for
the default benchmark path.

The initial realistic context ladder is:

- `128,16384,32768,65536,131072` tokens
- equivalently `2,256,512,1024,2048` pages at `page_size=64`

The Qwen3.5 geometry is:

- `q_seqlen=1`
- `page_size=64`
- `q_heads=8`
- `kv_heads=1`
- `head_dim=256`
- `q_dtype=bf16`
- `kv_dtype=bf16` and `fp8_e4m3fn`

## Current Mismatch

Today the benchmark does not match the serving graph contract closely enough:

1. It captures one graph per `(batch, q_seqlen, cache_seqlen)` case.
2. It does not own a long-lived decode graph bucket object that is reused
   across multiple runtime context lengths.
3. Its default decode matrix is short-context synthetic coverage
   (`64,512,2048,8192`), not the long-context Qwen3.5 ladder.
4. It reports generic per-case geomeans instead of per-bucket serving tables.

## Constraints

- Keep the comparison fair: FlashInfer FA2 must use the same batch buckets,
  capture strategy, and replay-only timing discipline.
- Do not recapture inside the timed loop.
- Do not allocate a new workspace inside the timed loop.
- Do not change the benchmark to make `b12x` look better by relaxing the
  serving contract.
- Preserve `--check` correctness validation.

## Work Plan

### 1. Introduce decode graph buckets

Refactor the benchmark around an explicit decode bucket abstraction, for
example `DecodeGraphBucket`.

Each bucket should own:

- `batch`
- `capture_context_tokens`
- static `q`, `k_cache`, `v_cache`
- static metadata buffers used during replay
- one `PagedAttentionWorkspace`
- one captured `torch.cuda.CUDAGraph`
- the comparable FA2 graph state for the same bucket

The bucket should expose:

- `prepare_replay(context_tokens=...)`
- `bench_replay(replays=...)`

### 2. Capture once, replay many

For each batch bucket:

1. Allocate the max-context K/V cache for that bucket.
2. Capture one graph at the bucket's capture context.
3. Reuse the captured graph for each runtime context in the ladder.
4. Update only runtime metadata between measurements.
5. Time only `graph.replay()`.

This is the key realism change. The benchmark should stop treating each runtime
context as a fresh capture problem.

### 3. Match the current serving path first

The benchmark should mirror the decode graph behavior used by the serving
stack today:

- batch buckets come from the serving graph pool model in
  `serve/engine/cuda_graph.py`
- per-layer paged attention workspaces are long-lived per graph
- runtime metadata changes between replays

Do not invent a benchmark-only graph contract first. Start by matching the
current serving path, then optimize it.

### 4. Add a realistic CLI

Add options along these lines:

- `--batch-buckets 1,2,4,8,12,16`
- `--decode-contexts 128,16384,32768,65536,131072`
- `--capture-context 0` to use the tuned per-bucket capture contract
- `--mode decode-graph-buckets`

Optional later extension:

- per-bucket capture contexts if memory or tuning requires it

The important contract is still one capture per batch bucket.

### 5. Keep Decode Contexts Positive

The default ladder starts at `128`, and decode-bucket mode should stay scoped
to strictly positive contexts.

Do not add benchmark-only empty-KV handling. If someone wants to study true
empty-KV decode, that should be a separate targeted investigation with its own
correctness contract.

### 6. Keep FA2 on the same contract

For FlashInfer FA2:

- capture once per batch bucket at the bucket capture context
- reuse the same graph across runtime contexts in that bucket
- re-plan or refresh metadata only outside the timed region
- time only replay

The comparison is only meaningful if both backends follow the same capture and
replay semantics.

### 7. Improve reporting

Make the benchmark print one row per serving-relevant point:

- `kv_dtype`
- `batch`
- `context_tokens`
- `capture_tokens`
- b12x mean/min/CI
- FA2 mean/min/CI
- `fa2/b12x`
- plan description

Keep geomean as a secondary summary, not the primary output.

The primary artifact for this session should be a per-bucket decode table.

### 8. Add replay-specific tests

Add tests that prove the new benchmark contract:

- one bucket can replay multiple context lengths without recapture
- the b12x bucket and FA2 bucket both update metadata correctly
- batch buckets `1,2,4,8,12,16` are accepted by default
- zero and negative decode contexts are rejected with a clear error

Existing correctness tests stay in place.

## Suggested Phase Order

### Phase 1

Land the bucket abstraction and prove it at:

- batch `1`
- contexts `128,16384,32768,65536,131072`

This is the minimal end-to-end proof that the benchmark matches the intended
serving model.

### Phase 2

Extend the same harness to:

- `2,4,8,12,16,32`

Do not fork the design by batch size. The same code path should handle every
requested bucket.

### Phase 3

Once the bucket benchmark exists, decide whether to:

- retune decode policies for `1,2,4,8,12,16`
- update the registered tuning modules
- collect a trace-driven context distribution from real SGLang step logs

## Success Criteria

This effort is successful when:

1. The benchmark captures one decode graph per batch bucket and no longer
   recaptures per runtime context.
2. The benchmark reports Qwen3.5-relevant decode tables for the requested
   batch buckets.
3. The timed region is replay-only.
4. FA2 follows the same contract.
5. The benchmark either supports `0` context correctly or reports it as a real
   blocker instead of hiding it.
