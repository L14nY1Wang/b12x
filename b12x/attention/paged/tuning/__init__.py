from __future__ import annotations

import importlib
import importlib.util
import pathlib
import pkgutil
import sys

from .registry import (
    DECODE_GRAPH_POLICY,
    DecodeGraphPolicy,
    get_decode_graph_policy,
    lookup_decode_graph_chunk_pages,
    register_decode_graph_policy,
    normalize_kv_dtype_key,
)

for _module_info in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
    if _module_info.name.rsplit(".", 1)[-1] == "registry":
        continue
    importlib.import_module(_module_info.name)

_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
for _policy_path in sorted(_PACKAGE_DIR.glob("*.*.bs*.py")):
    if _policy_path.name in {"__init__.py", "registry.py"}:
        continue
    _module_name = f"{__name__}._generated_{_policy_path.stem.replace('.', '_')}"
    if _module_name in sys.modules:
        continue
    _spec = importlib.util.spec_from_file_location(_module_name, _policy_path)
    if _spec is None or _spec.loader is None:
        continue
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_module_name] = _module
    _spec.loader.exec_module(_module)

__all__ = [
    "DECODE_GRAPH_POLICY",
    "DecodeGraphPolicy",
    "get_decode_graph_policy",
    "lookup_decode_graph_chunk_pages",
    "register_decode_graph_policy",
    "normalize_kv_dtype_key",
]
