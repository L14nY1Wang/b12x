# Attention Tuning Saga

This note captures the main learnings from the `rs-4` FP8 paged-attention tuning line.

The source of record for benchmark rows is `results.attention.tsv`. This doc is the
compressed narrative: what actually moved the needle, what looked promising but
failed, and what remains unresolved.

## Goal

- Primary metric: FP8 paged-attention `fa2/b12x` geomean on the serving-style matrix.
- Hard guardrail: no material BF16 regression.
- Correctness is non-negotiable: benchmark `--check`, contiguous guardrail, and
  attention tests must stay clean.
- Runtime benchmarking was done on `CUDA_VISIBLE_DEVICES=2`.

## High-Level Result

The line started with a weak FP8 paged baseline around `0.427x` and eventually
reached a refreshed merged-best baseline at `1f934cc` with FP8 paged `0.827x`
and BF16 paged `1.062x`.

That improvement did not come from one kernel rewrite. It came from stacking a
small number of real wins:

- more outer parallelism through split policy changes
- correct packed-load / packed-fragment serving paths
- direct scheduling for uniform paged Q
- a stronger split-combine reducer
- a long-uniform FP8 extend planner rule that promoted the `48x64`, `3`-warp tile only where it helped
- full FP8 paged K/V TMA

## Milestones That Held

These are the commits that materially changed the floor.

- `137ea0c`
  - Raised the split ceiling to `16` and delayed extend’s jump to `16`.
  - Moved FP8 paged from `0.427x` to `0.492x`.
  - Takeaway: occupancy and outer parallelism were real bottlenecks early.

- `d44547a`
  - Promoted FP8 decode to `16` splits at `32+` pages.
  - Reached FP8 paged `0.508x`.
  - Takeaway: decode needed more CTAs sooner than extend.

- `889fcab`
  - Landed the shared-layout packed-load fix in the FP8 forward path.
  - Reached FP8 paged `0.634x`.
  - Takeaway: the raw-load optimization was real, but only after fixing the page-view / shared-layout contract.

- `9a57847`
  - Decode-direct scheduling plus softmax-row-state follow-up.
  - Reached FP8 paged `0.644x`, BF16 `1.024x`.
  - Takeaway: control-path cleanup mattered, not just memory traffic.

- `977ac07`
  - Long FP8 decode micro promotion by occupancy.
  - Reached FP8 paged `0.656x`, BF16 `1.057x`.
  - Takeaway: the long-context decode family choice was under-optimized.

- `62f0c14`
  - Full FP8 paged K/V TMA to all page counts.
  - Reached FP8 paged `0.709x`, BF16 `1.039x`.
  - Takeaway: this was a real structural win, not noise.

- `a1fcae2`
  - Uniform-paged-Q direct scheduler transplant.
  - Reached FP8 paged `0.721x`, BF16 `1.041x`.
  - Takeaway: avoiding unnecessary varlen scheduler overhead helped once the memory path was stronger.

- `a137636`
  - Multi-warp split-combine reducer.
  - Reached FP8 paged `0.743x`, BF16 `1.075x`.
  - Takeaway: combine was not free once split counts got large enough.

- `5b82076`
  - Limited the `48x64`, `3`-warp direct tile to very long uniform FP8 extend.
  - Reached FP8 paged `0.791x`, BF16 `1.061x`.
  - Biggest lift was `extend k=32768`.
  - Takeaway: planner selectivity mattered more than globally promoting the tile.

- `1f934cc`
  - Refreshed merged-best baseline on `cuda2`.
  - FP8 paged `0.827x`, BF16 paged `1.062x`.

## Ideas That Repeatedly Failed

These categories were explored many times and mostly stayed flat or regressed.

- Simple tile / warp / register-budget tweaks
  - Usually moved results by noise or made long contexts worse.
  - Takeaway: once the kernel was reasonably balanced, static shape retuning alone stopped helping.

- Raising split counts indiscriminately
  - Useful early, then saturated quickly.
  - `32`-way decode and more aggressive extend splits were mostly regressions or noise.
  - Takeaway: split depth only helps until combine overhead and per-CTA work imbalance take over.

- Partial combine redesigns
  - Stripe-across-head-dim and related combine experiments sometimes nudged FP8 up but usually regressed BF16.
  - Takeaway: combine changes had to be real architectural improvements, not local threading tweaks.

- Naive packed PV raw-word paths
  - Many variants reached `0.86x` to `0.93x` but collapsed correctness.
  - The best wrong variants were:
    - `9e68e2b` `0.901x`
    - `18e6140` `0.923x`
    - `9f1253d` `0.930x`
  - Takeaway: the value-path byte/word layout is extremely sensitive. “Fast and wrong” was easy here.

- Partial rs-5 PV transpose transplants
  - They often improved throughput but kept landing in the same “small cosine miss” or full layout corruption regime.
  - Takeaway: that line was not drop-in compatible with the live raw/shared layout contract.

## Important Learnings From the PV Path

The value path produced the most confusing failures. These were the durable findings:

- The old `Uint32 copy_B` raw-load path is fundamentally wrong.
  - It changes the tensor shape and therefore changes which shared-memory addresses the copy path touches.

- The working packed-fragment donor logic was not “use wider raw loads”.
  - The useful part was preserving the correct `Uint8` load path and only changing how bytes were packed later.

- The copy-view `Uint32` recast on the correct `Uint8` path already matches the correct flattened packed-word order.
  - That means some “pack optimization” ideas were already effectively optimized by the compiler.

- The `sV` BF16 shared-memory destination layout is column-pair-major.
  - In probe form:
    - `flat_idx = col_pair * tile_n + row`
  - This is why naive row-major `Uint32` stores to `sV` were wrong.

- A direct FP8-stage-store probe showed that the base packed BF16 shared-store is exact in isolation.
  - `benchmarks/probe_fp8_stage_store.py --store-threads 32` returns `mismatch_count: 0`.
  - All nearby local permutations are fully wrong:
    - `swap_words`
    - `swap_halves`
    - `swap_words + swap_halves`

## Native FP8 MMA: Why It Was Not the Immediate Fix

Native SM120 warp-level FP8 MMA does exist in CUTLASS headers
(`mma.sync.aligned.kind::f8f6f4.m16n8k32...`), but it did not turn into an
immediate keep path here for two reasons:

- The live attention math is mixed-type.
  - QK is effectively `BF16 x FP8`
  - PV is effectively `BF16 x FP8`
  - The native FP8 MMA route wants low-precision inputs on both sides, which is not the current algorithm.

- The CUTLASS Python DSL path for this still appears incomplete for our use.
  - Inline PTX is technically possible in this repo via `dsl_user_op` and `llvm.inline_asm`.
  - But wiring a full FP8 MMA replacement into the live kernel would be a bigger algorithm/runtime change, not a quick dequant deletion.

## The Latest Overlap Investigation

The most recent line tried to hide PV dequant by moving FP8->BF16 V conversion to
the producer side and feeding the consumer with BF16 `sV`.

What happened:

- A scalar producer-side dequant proved the barrier/control idea was correct.
  - It was numerically clean.
  - It was catastrophically slow.

- A packed producer-side BF16 shared-store restored the speed profile.
  - But the full attention kernel still failed correctness.

- The isolated probes then showed:
  - the packed store itself is exact
  - the `32`-thread producer version is also exact
  - the destination layout formula is correct

That narrowed the remaining bug substantially.

The problem is not:

- the local packed BF16 store permutation
- the choice of `32` producer threads vs a wider threaded store
- the obvious local word/half reorderings

The problem is likely one of:

- a later consumer-fragment / transpose / copy-path interaction that the scalar-vs-logical probe does not capture
- a runtime lifetime / synchronization issue that only appears once the faster packed producer path is wired into the full kernel

## Practical Next Step

The highest-signal next probe is:

- compare the actual consumer `tOrVt` fragment contents for
  - scalar producer-side dequant into `sV`
  - packed producer-side dequant into `sV`

If those fragments differ, the remaining bug is in the `sV -> sVt -> tiled_copy_B -> MMA fragment`
path, not in the local BF16 packed store.

If those fragments match, the remaining bug is a control/lifetime issue in the
live overlap path rather than a data-layout issue.

## Files Added For This Investigation

- `benchmarks/probe_bf16_smem_layout.py`
  - Dumps the BF16 shared-memory packed-word order for `sV`.

- `benchmarks/probe_fp8_stage_store.py`
  - Compares scalar vs packed FP8->BF16 shared stores in isolation.

These probes are meant to be disposable research tools, not runtime code.
