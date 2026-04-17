from __future__ import annotations

import warnings

import b12x
import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.dsl import BaseDSL

import b12x.cute.runtime_patches as runtime_patches
from b12x.cute.runtime_patches import _build_compile_disk_cache_key, _structural_cache_key
from b12x.cute.utils import make_ptr


def test_compile_only_cache_warning_is_suppressed() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        BaseDSL.print_warning(object(), "Cache is disabled as user wants to compile only.")

    assert captured == []


def test_other_cutlass_warnings_still_emit() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        BaseDSL.print_warning(object(), "some other warning")

    assert len(captured) == 1
    assert str(captured[0].message) == "some other warning"


def test_b12x_pointer_cache_key_is_structural() -> None:
    ptr_a = make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16)
    ptr_b = make_ptr(cutlass.Int32, 32, cute.AddressSpace.gmem, assumed_align=16)

    assert ptr_a.__cache_key__ == ptr_b.__cache_key__


def test_compile_disk_cache_key_ignores_pointer_address_and_stream_value() -> None:
    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    ptr_a = make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16)
    ptr_b = make_ptr(cutlass.Int32, 32, cute.AddressSpace.gmem, assumed_align=16)

    compile_callable = cute.compile

    key_a = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_ignores_pointer_address_and_stream_value,
        (fake, ptr_a, 0),
        {},
    )
    key_b = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_ignores_pointer_address_and_stream_value,
        (fake, ptr_b, 0),
        {},
    )

    assert key_a == key_b


def test_compile_disk_cache_key_changes_with_compile_env(monkeypatch) -> None:
    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    compile_callable = cute.compile

    monkeypatch.delenv("NVCC_PREPEND_FLAGS", raising=False)
    key_a = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_changes_with_compile_env,
        (fake, 0),
        {},
    )

    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "--use_fast_math")
    key_b = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_changes_with_compile_env,
        (fake, 0),
        {},
    )

    assert key_a != key_b


def test_compile_disk_cache_key_changes_with_toolchain_key(monkeypatch) -> None:
    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    compile_callable = cute.compile

    monkeypatch.setattr(
        runtime_patches,
        "_runtime_toolchain_key",
        lambda: (("cutlass_dsl", "4.4.1"),),
    )
    key_a = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_changes_with_toolchain_key,
        (fake, 0),
        {},
    )

    monkeypatch.setattr(
        runtime_patches,
        "_runtime_toolchain_key",
        lambda: (("cutlass_dsl", "4.4.2"),),
    )
    key_b = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_changes_with_toolchain_key,
        (fake, 0),
        {},
    )

    assert key_a != key_b


def test_structural_cache_key_handles_symbolic_fake_compact_tensor_dims() -> None:
    class FakeSymInt:
        def __init__(self, name: str) -> None:
            self.name = name

        def __int__(self) -> int:
            raise TypeError("symbolic dim")

        def __str__(self) -> str:
            return self.name

    FakeCompactTensor = type("_FakeCompactTensor", (), {})
    FakeCompactTensor.__module__ = "cutlass.cute.runtime"
    fake = FakeCompactTensor()
    fake._dtype = cutlass.Int32
    fake._shape = (FakeSymInt("s0"), 8)
    fake._stride_order = (1, 0)
    fake._memspace = cute.AddressSpace.gmem
    fake._assumed_align = 4
    fake._use_32bit_stride = True

    key = _structural_cache_key(fake)

    assert key[0] == "fake_compact_tensor"
    assert key[2][0] == ("symbolic_dim", FakeSymInt.__module__, FakeSymInt.__qualname__, "s0")


def test_structural_cache_key_distinguishes_unnamed_cutlass_symbolic_dims() -> None:
    FakeTensor = type("_FakeTensor", (), {})
    FakeTensor.__module__ = "cutlass.cute.runtime"

    fake_a = FakeTensor()
    fake_a._dtype = cutlass.Int32
    fake_a._shape = (cute.sym_int32(divisibility=8), 8)
    fake_a._stride = (8, 1)
    fake_a._memspace = cute.AddressSpace.gmem
    fake_a._assumed_align = 4

    fake_b = FakeTensor()
    fake_b._dtype = cutlass.Int32
    fake_b._shape = (cute.sym_int32(divisibility=8), 8)
    fake_b._stride = (8, 1)
    fake_b._memspace = cute.AddressSpace.gmem
    fake_b._assumed_align = 4

    assert _structural_cache_key(fake_a) != _structural_cache_key(fake_b)


def test_structural_cache_key_skips_warninging_fake_tensor_cache_key() -> None:
    class FakeSymInt:
        def __int__(self) -> int:
            raise TypeError("symbolic dim")

        def __str__(self) -> str:
            return "?{i32 div=8}"

    class FakeTensor:
        __module__ = "some.fake.runtime"

        def __init__(self) -> None:
            self.dtype = cutlass.Int32
            self.shape = (FakeSymInt(), 8)
            self._stride = (8, 1)

        def stride(self):
            return self._stride

        @property
        def __cache_key__(self):
            warnings.warn(
                "FakeTensor cache_key contains unnamed symbolic dimensions. "
                "Different variables with the same shape/stride pattern will have identical cache keys, "
                "which may cause incorrect cache hits.",
                UserWarning,
                stacklevel=2,
            )
            return ("should_not_be_used",)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        key = _structural_cache_key(FakeTensor())

    assert captured == []
    assert key[0] == "fake_tensor"
