"""DSV4 tree node KV pipeline (sglang port of vllm-ascend's node_pool.py).

Per layer:
  hidden[T, H] -> compressor_tree_project -> nodes[n+1, ...]
               -> compressor_tree          -> node_kv[n, head]
               -> sparse_attn_sharedkv_tree(cmp_tree_kv=node_kv)

Semantics (from vllm-ascend):
- Every tree node uses its *own* hidden row (Hidden[node] = node), not a
  depth-shared row. The target verify already produces one hidden row per
  verify node, so `hidden` is passed through unchanged.
- Node KV is ephemeral for the verify forward. The compressor epilogue still
  writes the accepted path's compressed KV to the verify locations, so commit
  reuses the existing verify -> history move.
"""

from dataclasses import dataclass

import torch

from tree.attention import tree_attention, tree_attn_op_available


@dataclass
class NodePlan:
    n: int
    start: torch.Tensor  # [n] int32 prefix base per node
    depths: torch.Tensor  # [n] int64 depth per node (root = 0)
    ancestors: torch.Tensor  # [n, A] int64 root-first node ids
    req_ids: torch.Tensor  # [n] int64 request index per node
    query_cu: torch.Tensor  # [n + 1] int32
    hidden_ids: torch.Tensor  # [n] int64
    token_index: torch.Tensor  # [n] int64


def build_node_plan(pre) -> NodePlan:
    """Derive the node plan from a TreePrePlan (no device readback)."""
    meta = pre.metadata
    n = meta.verify_ids.numel()
    device = meta.verify_ids.device
    return NodePlan(
        n=n,
        start=meta.tree_prefix.to(torch.int32),
        depths=(meta.positions - meta.tree_prefix.to(meta.positions.dtype)).to(
            torch.int64
        ),
        ancestors=meta.tree_ancestors.to(torch.int64),
        req_ids=meta.node_req_ids.to(torch.int64),
        query_cu=meta.node_cu_seqlens_q.to(torch.int32),
        hidden_ids=torch.arange(n, dtype=torch.int64, device=device),
        token_index=torch.arange(n, dtype=torch.int64, device=device),
    )


def tree_compressor_ops_available() -> bool:
    try:
        namespace = getattr(torch.ops, "_C_ascend", None)
        return namespace is not None and hasattr(namespace, "compressor_tree")
    except Exception:
        return False


def compress_nodes(
    module,
    hidden: torch.Tensor,
    state: torch.Tensor,
    table: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    plan: NodePlan,
):
    """Project and aggregate per-node compressed KV.

    ``module`` must expose the sglang DSV4 compressor attributes:
    ``ratio``, ``overlap``, ``head_dim``, ``ape``, ``rope_head_dim``,
    ``norm``, ``_fused_wkv_w``, ``_fused_wgate_w``, ``_fused_norm_weight_fp32``.
    Returns ``(node_kv [n, head] bf16, nodes [(n+1), 1, 1, 2*coff*head] fp32)``.
    """
    if not tree_compressor_ops_available():
        raise NotImplementedError(
            "DSpark tree mode requires torch.ops._C_ascend.compressor_tree / "
            "compressor_tree_project; the installed vendor does not expose them."
        )
    for name in ("_fused_wkv_w", "_fused_wgate_w", "_fused_norm_weight_fp32"):
        if getattr(module, name, None) is None:
            raise RuntimeError(
                f"compressor is missing {name}; call the backend's "
                "_ensure_fused_caches() before tree compression."
            )
    if hidden.shape[0] != plan.n:
        raise ValueError(
            f"tree node hidden rows {hidden.shape[0]} != plan.n {plan.n}; "
            "the target verify must produce one hidden row per tree node."
        )
    if plan.ancestors.shape[0] != plan.n:
        raise ValueError("ancestors row count must equal plan.n")
    coff = 1 + int(module.overlap)
    head = int(module.head_dim)
    device = hidden.device
    nodes = torch.zeros(
        (plan.n + 1, 1, 1, 2 * coff * head), dtype=torch.float32, device=device
    )
    residue = ((plan.start.long() + plan.depths) % int(module.ratio)).to(torch.int32)
    torch.ops._C_ascend.compressor_tree_project(
        hidden.index_select(0, plan.hidden_ids),
        module._fused_wkv_w,
        module._fused_wgate_w,
        nodes,
        module.ape,
        residue,
        plan.query_cu,
    )
    node_kv = torch.ops._C_ascend.compressor_tree(
        state,
        nodes,
        module._fused_norm_weight_fp32,
        cos,
        sin,
        table,
        plan.start,
        plan.ancestors,
        plan.depths,
        plan.req_ids,
        int(module.ratio),
        int(module.rope_head_dim),
        float(module.norm.variance_epsilon),
    )
    return node_kv, nodes


def attend_nodes(
    *,
    query: torch.Tensor,
    original: torch.Tensor,
    compressed: torch.Tensor,
    node_kv: torch.Tensor,
    metadata,
    plan: NodePlan,
    ori_sparse_indices: torch.Tensor,
    cmp_sparse_indices,
    ori_block_table,
    cmp_block_table,
    sinks,
    ratio: int,
    window: int,
    scale: float,
):
    """Tree attention over shared history + per-node compressed KV."""
    if not tree_attn_op_available():
        raise NotImplementedError(
            "DSpark tree mode requires "
            "torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree; "
            "the installed vendor does not expose it."
        )
    return tree_attention(
        q=query,
        ori_kv=original,
        cmp_kv=compressed,
        cmp_tree_kv=node_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=plan.query_cu,
        seqused_kv=metadata.node_seqused_kv,
        tree_ancestors=plan.ancestors,
        tree_prefix=plan.start,
        sinks=sinks,
        cmp_ratio=ratio,
        softmax_scale=scale,
        ori_win_left=window - 1,
        ori_win_right=0,
    )
