"""DSV4 tree attention: CPU reference + adapter for the native Ascend op."""

import os

import torch


def tree_attn_op_available() -> bool:
    """True only when the native tree attention op is registered."""
    try:
        namespace = getattr(torch.ops, "_C_ascend", None)
        return namespace is not None and hasattr(
            namespace, "npu_sparse_attn_sharedkv_tree"
        )
    except Exception:
        return False


def _allow_cpu_reference() -> bool:
    return os.environ.get("SGLANG_DSPARK_TREE_ALLOW_CPU_REFERENCE") == "1"


def _ancestors(layout, req: int, node_id: int) -> list[int]:
    if node_id <= 0:
        return [0]
    row = []
    cur = node_id
    while cur > 0:
        row.append(int(cur))
        cur = int(layout.parents[req, cur - 1])
    row.append(0)
    row.reverse()
    return row


def tree_attention_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    layout,
    *,
    prefix_lens: torch.Tensor,
    scale: float,
    window: int,
) -> torch.Tensor:
    """Reference attention where a row sees prefix + its strict ancestors + self.

    ``q`` is [T, H, D]; ``kv`` is [K, H, D] with rows
    ``[0, prefix)`` = committed prefix and ``prefix + node_id`` = node row.
    This is test-only and never used in serving.
    """
    if q.dim() != 3 or kv.dim() != 3:
        raise ValueError("q and kv must be [tokens, heads, dim]")
    batch, budget = layout.tokens.shape
    query_width = budget + 1
    total = batch * query_width
    if q.shape[0] != total:
        raise ValueError(f"q rows {q.shape[0]} != batch*query_width {total}")
    device = q.device
    mask = torch.zeros((total, kv.shape[0]), dtype=torch.bool, device=device)
    for req in range(batch):
        prefix = int(prefix_lens[req])
        for node_id in range(query_width):
            depth = 0 if node_id == 0 else int(layout.depths[req, node_id - 1])
            start = max(0, prefix + depth - window)
            for pos in range(start, prefix):
                mask[req * query_width + node_id, pos] = True
            for ancestor in _ancestors(layout, req, node_id):
                mask[req * query_width + node_id, prefix + ancestor] = True

    scores = torch.einsum("thd,khd->thk", q.float(), kv.float()) * float(scale)
    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("thk,khd->thd", probs, kv.float())


def build_tree_attn_kwargs(
    *,
    q,
    ori_kv,
    ori_sparse_indices,
    cu_seqlens_q,
    seqused_kv,
    tree_ancestors,
    tree_prefix,
    cmp_ratio,
    softmax_scale,
    **kwargs,
):
    """Assemble the native op kwargs, validating the required contract."""
    required = {
        "q": q,
        "ori_kv": ori_kv,
        "ori_sparse_indices": ori_sparse_indices,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_kv": seqused_kv,
        "tree_ancestors": tree_ancestors,
        "tree_prefix": tree_prefix,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing tree attention inputs: {missing}")
    out = dict(required)
    out.update(
        cmp_kv=kwargs.pop("cmp_kv", None),
        cmp_sparse_indices=kwargs.pop("cmp_sparse_indices", None),
        ori_block_table=kwargs.pop("ori_block_table", None),
        cmp_block_table=kwargs.pop("cmp_block_table", None),
        sinks=kwargs.pop("sinks", None),
        metadata=kwargs.pop("metadata", None),
        cmp_tree_kv=kwargs.pop("cmp_tree_kv", None),
        cu_seqlens_ori_kv=kwargs.pop("cu_seqlens_ori_kv", None),
        cu_seqlens_cmp_kv=kwargs.pop("cu_seqlens_cmp_kv", None),
        cmp_ratio=int(cmp_ratio),
        softmax_scale=float(softmax_scale),
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=int(kwargs.pop("ori_win_left", 127)),
        ori_win_right=int(kwargs.pop("ori_win_right", 0)),
        layout_q="TND",
        layout_kv="PA_ND",
    )
    out.update(kwargs)
    return out


def tree_attention(**kwargs) -> torch.Tensor:
    """Call the native op, or the test-only reference when explicitly enabled.

    A device failure is never swallowed: the reference is only reachable when
    ``SGLANG_DSPARK_TREE_ALLOW_CPU_REFERENCE=1`` and a CPU-style ``kv`` input is
    supplied.
    """
    if "kv" in kwargs:
        if _allow_cpu_reference():
            return tree_attention_reference(**kwargs)
        raise NotImplementedError(
            "DSpark tree mode requires the Ascend tree attention op "
            "torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree; it is unavailable."
        )
    if not tree_attn_op_available():
        raise NotImplementedError(
            "DSpark tree mode requires the Ascend tree attention op "
            "torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree; it is unavailable."
        )
    op_kwargs = build_tree_attn_kwargs(**kwargs)
    q = op_kwargs.pop("q")
    return torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **op_kwargs)[0]
