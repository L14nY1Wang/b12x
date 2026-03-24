"""HuggingFace model loading with recipe-based layer surgery.

Instantiates the HF model on the meta device (zero memory), then loads
safetensor shards directly to GPU. The recipe walks the HF module tree
for name resolution and packs weights into b12x structures.
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import safetensors.torch as sf
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from serve.model.ops import precompute_rope_freqs
from serve.tp.group import TPGroup


# -- recipe registry -------------------------------------------------------

_RECIPES = {}


def register_recipe(model_type: str):
    """Decorator to register a surgery recipe for a model_type."""
    def wrapper(fn):
        _RECIPES[model_type] = fn
        return fn
    return wrapper


# -- safetensor loading ----------------------------------------------------


class ShardedLoader:
    """Loads individual tensors from sharded safetensor files directly to GPU.

    Uses safetensors.safe_open to load one tensor at a time, avoiding
    the need to hold entire shards in memory. This is critical for
    models where shards span multiple layers and packed expert weights
    would otherwise double memory usage.
    """

    def __init__(self, model_path: str | pathlib.Path, device: str = "cuda"):
        self.model_path = pathlib.Path(model_path)
        self.device = device

        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            self.weight_map: dict[str, str] = dict(index["weight_map"])
        else:
            single = self.model_path / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"no safetensors found in {self.model_path}")
            self.weight_map = {k: "model.safetensors" for k in sf.load_file(str(single), device="meta").keys()}

        self._open_files: dict[str, object] = {}
        self._all_keys = set(self.weight_map.keys())

    def get(self, key: str) -> torch.Tensor:
        """Load a single tensor by name, directly to GPU."""
        shard_file = self.weight_map[key]
        if shard_file not in self._open_files:
            import safetensors
            path = str(self.model_path / shard_file)
            self._open_files[shard_file] = safetensors.safe_open(path, framework="pt", device=self.device)
        return self._open_files[shard_file].get_tensor(key)

    def evict_all(self) -> None:
        """Close all open file handles."""
        self._open_files.clear()

    def keys(self) -> set[str]:
        return self._all_keys


# -- public API ------------------------------------------------------------


class LoadedModel:
    """The result of loading and surgery: layers + embeddings + head."""

    def __init__(
        self,
        layers: nn.ModuleList,
        embed_tokens: nn.Embedding,
        final_norm_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        config: object,
    ):
        self.layers = layers
        self.embed_tokens = embed_tokens
        self.final_norm_weight = final_norm_weight
        self.lm_head_weight = lm_head_weight
        self.cos = cos
        self.sin = sin
        self.config = config


def load_model(
    model_path: str,
    device: torch.device | str = "cuda",
    tp_group: Optional[TPGroup] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> LoadedModel:
    """Load a HF model and apply b12x layer surgery.

    1. Instantiate the HF model on the meta device (zero memory).
    2. Build a ShardedLoader for direct-to-GPU safetensor loading.
    3. Apply the registered recipe, which walks the meta model for
       structure and loads weights from the ShardedLoader.
    """
    device_str = str(torch.device(device))
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    model_type = hf_config.model_type

    if model_type not in _RECIPES:
        # Lazy-load recipe modules to avoid circular imports.
        import serve.model.recipe_qwen3_5  # noqa: F401

    if model_type not in _RECIPES:
        raise ValueError(
            f"no surgery recipe registered for model_type={model_type!r}. "
            f"available: {list(_RECIPES.keys())}"
        )

    # Meta-device model: module tree with zero memory.
    # For multimodal models, use text_config to avoid instantiating vision encoder.
    instantiate_config = hf_config
    if hasattr(hf_config, 'text_config'):
        instantiate_config = hf_config.text_config
        instantiate_config.model_type = instantiate_config.model_type or model_type
    print(f"Instantiating {model_type} on meta device...")
    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(instantiate_config)

    # Direct-to-GPU tensor loader.
    loader = ShardedLoader(model_path, device=device_str)

    recipe_fn = _RECIPES[model_type]
    result = recipe_fn(hf_model, hf_config, loader, device_str, tp_group)

    # Free everything.
    del hf_model
    loader.evict_all()
    torch.cuda.empty_cache()

    return result


# -- register known recipes ------------------------------------------------

@register_recipe("minimax_m2")
def _apply_minimax_m2(hf_model, hf_config, loader, device, tp_group):
    from serve.model.recipe_minimax_m2 import build_config, extract_layer

    world_size = tp_group.world_size if tp_group is not None else 1
    cfg = build_config(hf_config, tp_world_size=world_size)

    # Precompute RoPE tables.
    max_seq_len = getattr(hf_config, "max_position_embeddings", 32768)
    cos, sin = precompute_rope_freqs(
        cfg.head_dim, cfg.rotary_dim, max_seq_len, base=cfg.rope_base, device=device
    )

    # Extract layers.
    layers = nn.ModuleList()
    for i in range(cfg.num_layers):
        layer = extract_layer(
            hf_model.model.layers[i], i, cfg, tp_group, device, loader
        )
        layers.append(layer)

    # Embedding and head.
    embed_weight = loader.get("model.embed_tokens.weight").to(torch.bfloat16)
    embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, _weight=embed_weight)
    final_norm_weight = loader.get("model.norm.weight")
    lm_head_weight = loader.get("lm_head.weight").to(torch.bfloat16)

    return LoadedModel(
        layers=layers,
        embed_tokens=embed_tokens,
        final_norm_weight=final_norm_weight,
        lm_head_weight=lm_head_weight,
        cos=cos,
        sin=sin,
        config=cfg,
    )



