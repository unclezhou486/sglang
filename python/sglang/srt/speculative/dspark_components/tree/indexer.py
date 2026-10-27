"""Tree-aware c4 lightning indexer adapter.

Mirrors vllm-ascend ``ops/triton/spec_decode/tree/native_indexer.py``. The tree
op needs per-node K (``nodes``) and its scale; those are produced by the c4
indexer compressor, which is a separate integration step.
"""

import torch

from .torch_ops import ensure_tree_ops


def tree_indexer_ops_available() -> bool:
    if not ensure_tree_ops():
        return False
    namespace = getattr(torch.ops, "_C_ascend", None)
    return namespace is not None and hasattr(
        namespace, "quant_lightning_indexer_tree"
    )


def tree_indexer_select_all(plan, topk: int, device) -> torch.Tensor:
    """Host-capacity shortcut: when history/4 <= topk every block is selected."""
    lengths = (plan.start.long() + plan.depths + 1) // 4
    positions = torch.arange(topk, dtype=torch.int32, device=device).expand(
        plan.n, -1
    )
    return torch.where(
        positions < lengths.unsqueeze(1).to(torch.int32),
        positions,
        torch.full_like(positions, -1),
    ).unsqueeze(1)


def tree_indexer_metadata(*, query: torch.Tensor, plan, topk: int):
    seq = (plan.start.long() + plan.depths + 1).int()
    lengths, residual = seq // 4, seq % 4
    max_kv = int(plan.start.max().item() + plan.depths.max().item() + 1)
    schedule = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=query.shape[1],
        num_heads_k=1,
        head_dim=query.shape[2],
        topk=topk,
        quant_mode=2,
        cu_seqlens_q=plan.query_cu,
        seqused_k=lengths,
        cmp_residual_k=residual,
        batch_size=plan.n,
        max_seqlen_q=1,
        max_seqlen_k=max(1, max_kv // 4),
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        device=str(query.device),
    )
    return lengths, residual, schedule


def tree_indexer(
    *,
    query: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    history: torch.Tensor,
    history_scale: torch.Tensor,
    nodes: torch.Tensor,
    node_scale: torch.Tensor,
    table: torch.Tensor,
    plan,
    topk: int,
    lengths: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    schedule: torch.Tensor | None = None,
) -> torch.Tensor:
    if not tree_indexer_ops_available():
        raise NotImplementedError(
            "DSpark tree indexer requires "
            "torch.ops._C_ascend.quant_lightning_indexer_tree; the installed "
            "vendor/binding does not expose it."
        )
    if lengths is None or residual is None or schedule is None:
        lengths, residual, schedule = tree_indexer_metadata(
            query=query, plan=plan, topk=topk
        )
    return torch.ops._C_ascend.quant_lightning_indexer_tree(
        query,
        history,
        weights.reshape(query.shape[:2]).half(),
        query_scale.reshape(query.shape[:2]).half(),
        history_scale,
        nodes.reshape(query.shape[0], query.shape[2]),
        node_scale,
        plan.ancestors,
        plan.start,
        plan.req_ids,
        table,
        plan.query_cu,
        lengths,
        residual,
        schedule,
        topk,
    )
