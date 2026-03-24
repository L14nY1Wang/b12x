"""Public paged-attention integration surface for the primary backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from b12x.attention.paged.api import clear_paged_caches, paged_attention_forward as _paged_attention_forward
from b12x.attention.paged.planner import (
    PagedPlan as PagedAttentionPlan,
    PagedPlanKey as PagedAttentionPlanKey,
    create_paged_plan as _create_paged_plan,
    infer_paged_mode as infer_paged_attention_mode,
)
from b12x.attention.paged.workspace import (
    PagedWorkspace as PagedAttentionWorkspace,
    allocate_paged_workspace_for_plan as _allocate_paged_workspace_for_plan,
)


@dataclass
class PagedAttentionWorkspacePool:
    """Caller-owned workspace cache partitioned by CUDA stream and plan."""

    workspaces: dict[tuple[int, PagedAttentionPlanKey], PagedAttentionWorkspace] = field(
        default_factory=dict
    )

    def clear(self) -> None:
        self.workspaces.clear()


def _resolve_paged_workspace(
    workspace: PagedAttentionWorkspace | PagedAttentionWorkspacePool,
    *,
    plan: PagedAttentionPlan,
) -> PagedAttentionWorkspace:
    if isinstance(workspace, PagedAttentionWorkspace):
        if workspace.plan_key is not None and workspace.plan_key != plan.key:
            raise ValueError(
                "paged workspace plan mismatch: "
                f"expected {workspace.plan_key}, got {plan.key}"
            )
        return workspace
    if not isinstance(workspace, PagedAttentionWorkspacePool):
        raise TypeError(
            "workspace must be a PagedAttentionWorkspace or PagedAttentionWorkspacePool"
        )
    stream_key = int(torch.cuda.current_stream(plan.device).stream_id)
    key = (stream_key, plan.key)
    resolved = workspace.workspaces.get(key)
    if resolved is None:
        resolved = _allocate_paged_workspace_for_plan(plan)
        workspace.workspaces[key] = resolved
    return resolved


def allocate_paged_attention_workspace_for_plan(
    plan: PagedAttentionPlan,
    total_q: int | None = None,
    batch: int | None = None,
) -> PagedAttentionWorkspace:
    if total_q is not None and int(total_q) != int(plan.total_q):
        raise ValueError(
            f"workspace total_q must match the plan total_q={plan.total_q}, got {total_q}"
        )
    if batch is not None and int(batch) != int(plan.page_table_shape[0]):
        raise ValueError(
            "workspace batch must match the plan page-table batch "
            f"{plan.page_table_shape[0]}, got {batch}"
        )
    return _allocate_paged_workspace_for_plan(plan)


def allocate_paged_attention_workspace_pool() -> PagedAttentionWorkspacePool:
    return PagedAttentionWorkspacePool()


def create_paged_attention_plan(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    causal: bool = True,
    mode: Literal["decode", "extend"] | None = None,
    fixed_split_size: int | None = None,
) -> PagedAttentionPlan:
    if not causal:
        raise ValueError("b12x paged attention currently supports causal mode only")
    return _create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode=mode,
        fixed_split_size=-1 if fixed_split_size is None else int(fixed_split_size),
    )


def allocate_paged_attention_workspace(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    causal: bool = True,
    mode: Literal["decode", "extend"] | None = None,
    fixed_split_size: int | None = None,
) -> PagedAttentionWorkspace:
    plan = create_paged_attention_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=causal,
        mode=mode,
        fixed_split_size=fixed_split_size,
    )
    return allocate_paged_attention_workspace_for_plan(plan)


def b12x_paged_attention_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    workspace: PagedAttentionWorkspace | PagedAttentionWorkspacePool,
    plan: PagedAttentionPlan | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if plan is None:
        raise TypeError("an explicit PagedAttentionPlan is required")
    if softmax_scale is not None:
        raise ValueError("softmax_scale overrides are unsupported on the primary paged backend")
    resolved_workspace = _resolve_paged_workspace(workspace, plan=plan)
    out, lse_base2 = _paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=resolved_workspace,
        plan=plan,
        k_descale=k_descale,
        v_descale=v_descale,
        output=output,
    )
    return out, lse_base2.transpose(0, 1)


def b12x_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    workspace: PagedAttentionWorkspace | PagedAttentionWorkspacePool,
    plan: PagedAttentionPlan,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if plan.mode != "decode":
        raise ValueError(f"expected a decode plan, got {plan.mode}")
    return b12x_paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=workspace,
        plan=plan,
        k_descale=k_descale,
        v_descale=v_descale,
        output=output,
    )


def b12x_paged_extend(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    workspace: PagedAttentionWorkspace | PagedAttentionWorkspacePool,
    plan: PagedAttentionPlan,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if plan.mode != "extend":
        raise ValueError(f"expected an extend plan, got {plan.mode}")
    return b12x_paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=workspace,
        plan=plan,
        k_descale=k_descale,
        v_descale=v_descale,
        output=output,
    )


def clear_attention_caches() -> None:
    clear_paged_caches()


__all__ = [
    "PagedAttentionPlan",
    "PagedAttentionPlanKey",
    "PagedAttentionWorkspace",
    "PagedAttentionWorkspacePool",
    "allocate_paged_attention_workspace",
    "allocate_paged_attention_workspace_pool",
    "allocate_paged_attention_workspace_for_plan",
    "b12x_paged_decode",
    "b12x_paged_attention_forward",
    "b12x_paged_extend",
    "clear_attention_caches",
    "create_paged_attention_plan",
    "infer_paged_attention_mode",
]
