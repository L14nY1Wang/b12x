# CUTLASS 4.4.2 Runtime `__cache_key__` and `b12x`

## Summary

CUTLASS 4.4.2 adds `__cache_key__` support to the CuTe DSL runtime
representations for:

- runtime pointers
- runtime tensors
- fake tensors

Upstream describes this as a stable, hashable representation intended to improve
compiled-function caching. For `b12x`, this matters because the MoE and dense
GEMM compile paths build large `cute.compile(...)` signatures out of fake
tensors and runtime pointer placeholders.

The important detail is that `b12x` does not use the upstream runtime pointer
class directly in all places. It carries its own compatibility wrapper in
`b12x/cute/utils.py`, so upgrading `nvidia-cutlass-dsl` to 4.4.2 does not
automatically give `b12x` the upstream pointer `__cache_key__` behavior.

## Why This Is Relevant

`b12x` compiles kernels from Python-side signatures rather than only from
static source text. The compile-time argument objects are part of the identity
of the generated function.

Current high-impact call sites:

- `b12x/integration/tp_moe.py`
  - `_get_static_kernel(...)`
  - `_get_dynamic_kernel(...)`
- `b12x/gemm/dense.py`
  - `_get_compiled_dense_gemm(...)`

The MoE paths construct many fake tensors before calling `cute.compile(...)`.
Examples include activation tensors, scale-storage buffers, token maps, and
other scheduler/control buffers. The dense GEMM path constructs runtime pointer
placeholders and passes those directly to `cute.compile(...)`.

That means `b12x` is directly exposed to any change in how CuTe DSL identifies
runtime arguments for compilation and caching.

## Upstream 4.4.2 Change

In CUTLASS 4.4.2:

- runtime `_Pointer` has a `__cache_key__` property
- runtime `_Tensor` has a `__cache_key__` property
- runtime `_FakeTensor` has a `__cache_key__` property

The intent is straightforward:

- cache keys should be stable
- cache keys should be hashable
- equivalent runtime signatures should resolve to the same compiled artifact

For fake tensors, the cache key is based on dtype, memory space, alignment,
shape, and stride. For runtime pointers, the cache key is structural rather than
address-based: dtype, address space, and assumed alignment.

That structural choice is important. It avoids tying compilation identity to a
specific pointer value, which would defeat most caching.

## Current `b12x` State

### Fake tensors

The MoE compile paths already benefit from the upstream fake-tensor change once
the project upgrades to CUTLASS 4.4.2, because those call sites use
`cute.runtime.make_fake_compact_tensor(...)` directly.

This covers a large fraction of the compile signature in:

- `b12x/integration/tp_moe.py`

### Runtime tensors

`b12x` does not appear to rely heavily on runtime tensor objects as compile-time
arguments in the hot paths discussed here. This is lower priority than pointers
and fake tensors.

### Runtime pointers

This is the gap.

`b12x` defines a local `_Pointer` wrapper in `b12x/cute/utils.py` and exposes it
through `make_ptr(...)`. That local wrapper exists for compatibility reasons and
is used in:

- `b12x/gemm/dense.py`
- `b12x/integration/tp_moe.py`

The local wrapper currently provides:

- MLIR type lowering
- C-pointer marshaling
- type verification for the DSL

It does **not** provide `__cache_key__`.

So after a CUTLASS 4.4.2 upgrade, `b12x` would have:

- upstream fake tensors with `__cache_key__`
- upstream runtime tensors with `__cache_key__`
- local runtime pointers without `__cache_key__`

That is an inconsistent state.

## Practical Impact

This is not expected to change kernel correctness.

The main impact is on compile identity and caching behavior:

- cache reuse may be weaker than intended for signatures containing local
  pointer placeholders
- `b12x` would fail to pick up the full benefit of the 4.4.2 caching change
- debugging cache behavior becomes harder because some argument kinds participate
  in structural cache keys while local pointers do not

The dense GEMM compile path is the clearest example because
`_get_compiled_dense_gemm(...)` constructs placeholder pointers and passes them
to `cute.compile(...)` directly.

The MoE compile paths also use `make_ptr(...)` for scale-factor placeholders:

- `sfa_fake`
- `sfb_w13_fake`
- `sfb_down_fake`

Those pointers are part of the compile-time signature as well.

## Recommended Action

When upgrading `nvidia-cutlass-dsl` from 4.4.1 to 4.4.2, add the upstream-style
`__cache_key__` property to `b12x`'s local pointer wrapper.

The intended shape is:

```python
@property
def __cache_key__(self) -> tuple:
    return (self.dtype, self._addr_space, self._assumed_align)
```

This matches the upstream structural behavior:

- include dtype
- include address space
- include assumed alignment
- do not include the raw pointer value

Not including the raw pointer value is deliberate. Compilation should depend on
the pointer type contract, not on the specific allocation address.

## Why Not Remove the Local Wrapper Immediately

Replacing `b12x.cute.utils.make_ptr(...)` with upstream `cute.runtime.make_ptr`
may become possible later, but it should not be treated as a no-risk cleanup.

The local wrapper currently carries behavior that `b12x` depends on:

- pointer verification behavior
- local compatibility assumptions
- existing call-site expectations

The lowest-risk change is to add `__cache_key__` first and keep the wrapper
otherwise unchanged.

## Suggested Validation

After the CUTLASS upgrade and local pointer update:

1. Run a dense GEMM compile smoke test.
2. Run static and dynamic MoE compile smoke tests.
3. Repeat representative compile calls in the same process and verify that the
   second call reuses cached compilation rather than regenerating code.
4. Keep the existing correctness and CUDA-graph replay tests unchanged; this
   note is about compile identity and caching, not kernel math.

## Non-Goals

This note does not claim that the 4.4.2 `__cache_key__` change alone will
improve runtime performance of the generated kernels.

The expected benefit is narrower:

- cleaner compile-time identity
- better cache stability
- fewer unnecessary recompiles in repeated compile scenarios

## References

- CUTLASS `v4.4.2` release commit: `da5e086dab31d63815acafdac9a9c5893b1c69e2`
- Upstream change areas:
  - `python/CuTeDSL/cutlass/cute/runtime.py`
  - `python/CuTeDSL/cutlass/cute/typing.py`
  - `python/CuTeDSL/cutlass/cute/tensor.py`
- `b12x` local integration points:
  - `b12x/cute/utils.py`
  - `b12x/integration/tp_moe.py`
  - `b12x/gemm/dense.py`
