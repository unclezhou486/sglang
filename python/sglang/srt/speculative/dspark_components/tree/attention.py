"""DSV4 tree attention: CPU reference + adapter for the native Ascend op."""

import os

import torch


def tree_attn_op_available() -> bool:
    """True only when the native tree attention op is registered."""
    from .torch_ops import ensure_tree_ops

    if not ensure_tree_ops():
        return False
    try:
        namespace = getattr(torch.ops, "_C_ascend", None)
        return namespace is not None and hasattr(
            namespace, "npu_sparse_attn_sharedkv_tree"
        )
    except Exception:
        return False


def build_tree_attn_metadata(*, query, metadata, plan, ratio, window, cmp_sparse_indices, compressed):
    """Build the tiling metadata the tree attention op requires.

    Mirrors vllm-ascend's ``native_attention``: the metadata is generated from
    per-node seqused lengths, and the same ``cu_seqlens_*`` args are passed to
    the tree op.
    """
    seq = metadata.node_seqused_kv.to(torch.int32)
    cu_ori = torch.cat((seq.new_zeros(1), seq.cumsum(0, dtype=torch.int32)))
    cu_cmp = torch.cat(
        (seq.new_zeros(1), (seq // int(ratio)).cumsum(0, dtype=torch.int32))
    )
    cmp_topk = (
        int(cmp_sparse_indices.shape[-1])
        if cmp_sparse_indices is not None and int(ratio) == 4
        else 0
    )
    kwargs = dict(
        num_heads_q=query.shape[1],
        num_heads_kv=1,
        head_dim=query.shape[2],
        seqused_q=seq.new_empty(0),
        batch_size=plan.n,
        max_seqlen_q=1,
        max_seqlen_kv=max(1, int(seq.max().item()) if seq.numel() else 0),
        ori_topk=0,
        cmp_topk=cmp_topk,
        has_ori_kv=True,
        has_cmp_kv=compressed is not None,
        device=str(query.device),
        cu_seqlens_q=plan.query_cu,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        seqused_kv=seq,
        cmp_ratio=int(ratio),
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=int(window) + int(metadata.query_width) - 2,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )
    custom = getattr(torch.ops, "custom", None)
    op = getattr(custom, "npu_sparse_attn_sharedkv_metadata", None)
    if op is None:
        ascend = getattr(torch.ops, "_C_ascend", None)
        op = getattr(ascend, "npu_sparse_attn_sharedkv_metadata", None)
    if op is None:
        raise RuntimeError(
            "DSpark tree attention requires "
            "torch.ops.custom.npu_sparse_attn_sharedkv_metadata (or the "
            "_C_ascend equivalent); it is unavailable."
        )
    return op(**kwargs), cu_ori, cu_cmp


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
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing tree attention inputs: {missing}")
    cmp_tree_kv = kwargs.pop("cmp_tree_kv", None)
    tree_ancestors = kwargs.pop("tree_ancestors", None)
    tree_prefix = kwargs.pop("tree_prefix", None)
    if cmp_tree_kv is not None:
        if tree_ancestors is None or tree_prefix is None:
            raise ValueError(
                "tree attention requires tree_ancestors and tree_prefix when "
                "cmp_tree_kv is supplied"
            )
    else:
        # SWA-only layers have no compressed tree KV. The op's CheckFeature
        # requires cmp_tree_kv / ancestors / prefixes to be all-or-none.
        tree_ancestors = None
        tree_prefix = None
    out = dict(required)
    out.update(
        cmp_kv=kwargs.pop("cmp_kv", None),
        cmp_sparse_indices=kwargs.pop("cmp_sparse_indices", None),
        ori_block_table=kwargs.pop("ori_block_table", None),
        cmp_block_table=kwargs.pop("cmp_block_table", None),
        sinks=kwargs.pop("sinks", None),
        metadata=kwargs.pop("metadata", None),
        cmp_tree_kv=cmp_tree_kv,
        tree_ancestors=tree_ancestors,
        tree_prefix=tree_prefix,
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
