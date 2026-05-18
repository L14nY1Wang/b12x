"""Public NSA indexer integration surface."""

from __future__ import annotations

from b12x.attention.nsa_indexer import (
    NSAIndexerExtendLogitsMetadata,
    NSAIndexerPagedDecodeMetadata,
    clear_nsa_indexer_caches,
    get_paged_mqa_logits_metadata,
    make_nsa_indexer_contract_phantoms,
    pack_nsa_index_k_cache_reference,
    persistent_topk2048_workspace_nbytes,
    resolve_sparse_nsa_extend_prefill_block_k,
    run_persistent_topk2048,
    sparse_nsa_extend_logits_reference,
    sparse_nsa_index_decode_logits_paged,
    sparse_nsa_index_extend_logits,
    sparse_nsa_index_extend_tiled_topk,
    sparse_nsa_paged_logits_reference,
    supports_persistent_topk2048,
    unpack_nsa_index_k_cache_reference,
    uses_paged_mqa_schedule_metadata,
)

__all__ = [
    "NSAIndexerExtendLogitsMetadata",
    "NSAIndexerPagedDecodeMetadata",
    "clear_nsa_indexer_caches",
    "get_paged_mqa_logits_metadata",
    "make_nsa_indexer_contract_phantoms",
    "pack_nsa_index_k_cache_reference",
    "persistent_topk2048_workspace_nbytes",
    "resolve_sparse_nsa_extend_prefill_block_k",
    "run_persistent_topk2048",
    "sparse_nsa_extend_logits_reference",
    "sparse_nsa_index_decode_logits_paged",
    "sparse_nsa_index_extend_logits",
    "sparse_nsa_index_extend_tiled_topk",
    "sparse_nsa_paged_logits_reference",
    "supports_persistent_topk2048",
    "unpack_nsa_index_k_cache_reference",
    "uses_paged_mqa_schedule_metadata",
]
