"""Tree-aware c4 lightning indexer adapter.

Mirrors vllm-ascend ``ops/triton/spec_decode/tree/native_indexer.py``. The tree
op needs per-node K (``nodes``) and its scale; those are produced by the c4
indexer compressor, which is a separate integration step.
"""

import logging

import torch

from .backend import tree_backend
from .torch_ops import ensure_tree_ops

logger = logging.getLogger(__name__)

_shadow_failure_logged = False
_shadow_mismatch_logged = False


def tree_indexer_backend() -> str:
    """Indexer follows the global tree backend switch.

    ``shadow`` runs the vendor op for the real output and the Triton op on the
    same inputs, logging the first mismatch.  It is the safe SGLang bring-up
    mode: a Triton failure never changes the serving result.
    """
    return tree_backend()


def _load_sgl_kernel_indexer():
    try:
        from sgl_kernel_npu.tree.quant_lightning_indexer_tree import (
            quant_lightning_indexer_tree,
        )
    except Exception as exc:
        raise RuntimeError(
            "SGLANG_DSPARK_TREE_INDEXER_BACKEND=sgl_kernel_npu requires the "
            "sgl_kernel_npu tree package in this process"
        ) from exc
    return quant_lightning_indexer_tree


def _sgl_kernel_indexer_available() -> bool:
    try:
        _load_sgl_kernel_indexer()
        return True
    except Exception:
        return False


def _vendor_indexer_available() -> bool:
    if not ensure_tree_ops():
        return False
    namespace = getattr(torch.ops, "_C_ascend", None)
    return namespace is not None and hasattr(
        namespace, "quant_lightning_indexer_tree"
    )


def tree_indexer_ops_available() -> bool:
    backend = tree_indexer_backend()
    if backend == "sgl_kernel_npu":
        return _sgl_kernel_indexer_available()
    # shadow is fail-open: the vendor result is the real output, so a missing
    # Triton package must not disable tree serving.
    return _vendor_indexer_available()


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


def quantize_indexer_nodes(node_kv: torch.Tensor):
    """Quantize bf16 per-node indexer K to INT8 + FP16 per-node scale.

    Mirrors the indexer's own per-token dynamic quant. The tree op requires
    ``nodes`` INT8 [n, 128] and ``node_scale`` FP16 [n].
    """
    import torch_npu

    flat = node_kv.reshape(-1, node_kv.shape[-1])
    nodes, scale = torch_npu.npu_dynamic_quant(flat)
    return nodes.view(node_kv.shape), scale.reshape(-1).to(torch.float16)



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


def _tree_indexer_lengths(plan):
    seq = (plan.start.long() + plan.depths + 1).int()
    return seq // 4, seq % 4


def _vendor_tree_indexer(
    *,
    query,
    weights,
    query_scale,
    history,
    history_scale,
    nodes,
    node_scale,
    table,
    plan,
    topk,
    lengths,
    residual,
    schedule,
):
    if not _vendor_indexer_available():
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
        plan.ancestors.to(torch.int64),
        plan.start.to(torch.int32),
        plan.req_ids.to(torch.int64),
        table.to(torch.int32),
        plan.query_cu.to(torch.int32),
        lengths,
        residual,
        schedule,
        topk,
    )


def _sgl_kernel_tree_indexer(
    *,
    query,
    weights,
    query_scale,
    history,
    history_scale,
    nodes,
    node_scale,
    table,
    plan,
    topk,
    lengths,
    residual,
    schedule,
):
    op = _load_sgl_kernel_indexer()
    if lengths is None or residual is None:
        lengths, residual = _tree_indexer_lengths(plan)
    if schedule is None:
        # The Triton op validates the metadata ABI and ignores its contents.
        schedule = torch.zeros(1024, dtype=torch.int32, device=query.device)
    return op(
        query,
        history,
        weights.reshape(query.shape[:2]).half(),
        query_scale.reshape(query.shape[:2]).half(),
        history_scale,
        nodes.reshape(query.shape[0], query.shape[2]),
        node_scale,
        plan.ancestors.to(torch.int64),
        plan.start.to(torch.int32),
        plan.req_ids.to(torch.int64),
        table.to(torch.int32),
        plan.query_cu.to(torch.int32),
        lengths,
        residual,
        schedule,
        topk,
    )


def _shadow_outputs_match(vendor_out: torch.Tensor, triton_out: torch.Tensor) -> bool:
    if vendor_out.shape != triton_out.shape:
        return False
    # Top-k order is not contractual (D3): compare the sorted index multiset.
    rows = int(vendor_out.shape[0])
    vendor_sorted = torch.sort(
        vendor_out.reshape(rows, -1).to(torch.int64), dim=-1
    ).values
    triton_sorted = torch.sort(
        triton_out.reshape(rows, -1).to(torch.int64), dim=-1
    ).values
    return bool(torch.equal(vendor_sorted, triton_sorted))


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
    global _shadow_failure_logged, _shadow_mismatch_logged

    backend = tree_indexer_backend()
    common = dict(
        query=query,
        weights=weights,
        query_scale=query_scale,
        history=history,
        history_scale=history_scale,
        nodes=nodes,
        node_scale=node_scale,
        table=table,
        plan=plan,
        topk=topk,
        lengths=lengths,
        residual=residual,
        schedule=schedule,
    )
    if backend == "sgl_kernel_npu":
        return _sgl_kernel_tree_indexer(**common)
    if backend == "vendor":
        return _vendor_tree_indexer(**common)

    # shadow: vendor is authoritative; Triton is observational only.
    vendor_out = _vendor_tree_indexer(**common)
    try:
        triton_out = _sgl_kernel_tree_indexer(**common)
    except Exception as exc:
        if not _shadow_failure_logged:
            logger.warning(
                "DSpark tree indexer shadow Triton failed; vendor output is kept: %s",
                exc,
            )
            _shadow_failure_logged = True
        return vendor_out

    if not _shadow_outputs_match(vendor_out, triton_out):
        if not _shadow_mismatch_logged:
            logger.warning(
                "DSpark tree indexer shadow mismatch: vendor and Triton top-k "
                "multisets differ; vendor output is kept (backend=%s).",
                backend,
            )
            _shadow_mismatch_logged = True
    return vendor_out
