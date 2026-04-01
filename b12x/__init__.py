"""Public b12x package surface."""

from .cute.runtime_patches import apply_cutlass_runtime_patches

apply_cutlass_runtime_patches()
