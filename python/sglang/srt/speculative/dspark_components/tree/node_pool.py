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

from .attention import (
    build_tree_attn_metadata,
    tree_attention,
    tree_attn_op_available,
)


@dataclass
class NodePlan:
    n: int
    max_q: int
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
    ancestors = meta.tree_ancestors.to(torch.int64)
    offsets = meta.node_req_ids.to(torch.int64) * meta.query_width
    ancestors = torch.where(
        ancestors >= 0,
        ancestors + offsets[:, None],
        ancestors,
    )
    return NodePlan(
        n=n,
        max_q=meta.query_width,
        start=meta.tree_prefix.to(torch.int32),
        depths=(meta.positions - meta.tree_prefix.to(meta.positions.dtype)).to(
            torch.int64
        ),
        ancestors=ancestors,
        req_ids=meta.node_req_ids.to(torch.int64),
        query_cu=meta.node_cu_seqlens_q.to(torch.int32),
        hidden_ids=torch.arange(n, dtype=torch.int64, device=device),
        token_index=torch.arange(n, dtype=torch.int64, device=device),
    )


def tree_compressor_ops_available() -> bool:
    from .torch_ops import ensure_tree_ops

    if not ensure_tree_ops():
        return False
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
    *,
    record_sink: list | None = None,
    kv_buffer: torch.Tensor | None = None,
    kv_table: torch.Tensor | None = None,
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
    if record_sink is not None:
        # ratio=1 records the projected state row (one per node); the commit
        # kernel treats ratio=1 as state and writes it back to the state cache.
        record_sink.append(
            record_compressed_nodes(
                plan,
                original=state,
                node_data=nodes[1:],
                table=table,
                ratio=1,
            )
        )
        if kv_buffer is not None and kv_table is not None:
            record_sink.append(
                record_compressed_nodes(
                    plan,
                    original=kv_buffer,
                    node_data=node_kv,
                    table=kv_table,
                    ratio=int(module.ratio),
                )
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
    metadata_tensor, cu_ori, cu_cmp = build_tree_attn_metadata(
        query=query,
        metadata=metadata,
        plan=plan,
        ratio=ratio,
        window=window,
        cmp_sparse_indices=cmp_sparse_indices,
        compressed=compressed,
    )
    ori_table = (
        ori_block_table.index_select(0, plan.req_ids)
        if ori_block_table is not None
        else None
    )
    cmp_table = (
        cmp_block_table.index_select(0, plan.req_ids)
        if cmp_block_table is not None
        else None
    )
    return tree_attention(
        q=query,
        ori_kv=original,
        cmp_kv=compressed,
        cmp_tree_kv=node_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_table,
        cmp_block_table=cmp_table,
        cu_seqlens_q=plan.query_cu,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        seqused_kv=metadata.node_seqused_kv,
        tree_ancestors=plan.ancestors,
        tree_prefix=plan.start,
        sinks=sinks,
        metadata=metadata_tensor,
        cmp_ratio=ratio,
        softmax_scale=scale,
        ori_win_left=window - 1,
        ori_win_right=0,
    )


@dataclass
class NodeRecord:
    """One layer's compressed KV buffer plus its per-node rows."""

    original: torch.Tensor  # [pages, rows, ..., dim]
    node_data: torch.Tensor  # [n, ..., dim] contiguous
    table: torch.Tensor  # [reqs, width] int32 page table
    pages: tuple[int, ...]
    ratio: int


def _window_pages(plan: NodePlan, table: torch.Tensor, ratio: int, rows: int):
    req_ids = plan.req_ids.tolist()
    starts = plan.start.tolist()
    depth_limit = int(plan.max_q)
    width = int(table.shape[1])
    table_host = table.tolist()
    pages: set[int] = set()
    n_req = max(req_ids) + 1 if req_ids else 0
    for req in range(n_req):
        first = req_ids.index(req)
        prefix = int(starts[first])
        for pos in range(prefix, prefix + depth_limit):
            cpos = pos // ratio
            col = cpos // rows
            if col >= width:
                continue
            page = int(table_host[req][col])
            if page > 0:
                pages.add(page)
    return tuple(sorted(pages))


def record_compressed_nodes(
    plan: NodePlan,
    *,
    original: torch.Tensor,
    node_data: torch.Tensor,
    table: torch.Tensor,
    ratio: int,
) -> NodeRecord:
    """Record the pages and node rows for the post-accept commit kernels."""
    if ratio not in (1, 4, 128):
        raise ValueError(f"tree node commit ratio must be 1/4/128, got {ratio}")
    if original.dim() < 3:
        raise ValueError("original compressed buffer must be [pages, rows, ...]")
    if node_data.shape[0] != plan.n:
        raise ValueError(
            f"node rows {node_data.shape[0]} != plan.n {plan.n}"
        )
    if not node_data.is_contiguous():
        raise ValueError("node_data must be contiguous for the commit kernel")
    rows = int(original.shape[1])
    row_bytes = 1
    for dim in original.shape[2:]:
        row_bytes *= int(dim)
    row_bytes *= original.element_size()
    node_row_bytes = 1
    for dim in node_data.shape[1:]:
        node_row_bytes *= int(dim)
    node_row_bytes *= node_data.element_size()
    if node_row_bytes != row_bytes:
        raise ValueError(
            f"node row bytes {node_row_bytes} != cache row bytes {row_bytes}"
        )
    return NodeRecord(
        original=original,
        node_data=node_data,
        table=table,
        pages=_window_pages(plan, table, ratio, rows),
        ratio=int(ratio),
    )


def build_node_descriptors(records: list[NodeRecord]):
    """Pure descriptor list for freeze_node_pages; also returns staging size."""
    descriptors: list[tuple[int, ...]] = []
    total = 0
    max_rows = 1
    for record in records:
        original = record.original
        element = original.element_size()
        rows = int(original.shape[1])
        row_bytes = 1
        for dim in original.shape[2:]:
            row_bytes *= int(dim)
        row_bytes *= element
        page_stride = int(original.stride(0)) * element
        row_stride = int(original.stride(1)) * element
        for page in record.pages:
            descriptors.append(
                (
                    original.data_ptr() + page * page_stride,
                    row_stride,
                    row_bytes,
                    rows,
                    record.node_data.data_ptr(),
                    record.table.data_ptr(),
                    int(record.table.shape[1]),
                    int(page),
                    total,
                    int(record.ratio),
                )
            )
            total += rows * row_bytes
            max_rows = max(max_rows, rows)
    return descriptors, total, max_rows


def globalize_winners(plan: NodePlan, winners: torch.Tensor) -> torch.Tensor:
    """Map per-request endpoint node ids to flat verify-row ids.

    ``acceptance.endpoint_node`` is request-local (0..query_width-1). The
    commit kernel indexes ``Depths`` / ``Ancestors`` by flat row, so every
    request after the first needs its row offset added.
    """
    offsets = torch.arange(
        winners.numel(), dtype=winners.dtype, device=winners.device
    ) * int(plan.max_q)
    return winners + offsets


def prepare_nodes(plan: NodePlan, records: list[NodeRecord], winners, active, device):
    """Freeze accepted node rows into staging before touching the live cache."""
    winners = globalize_winners(plan, winners)
    try:
        from .node_commit import freeze_node_pages
    except ModuleNotFoundError as exc:
        raise NotImplementedError(
            "DSpark tree node commit requires Triton (freeze_node_pages); "
            "install triton to commit node pages."
        ) from exc
    descriptors, total, max_rows = build_node_descriptors(records)
    if not descriptors:
        return (
            torch.empty((0, 10), dtype=torch.int64, device=device),
            torch.empty((0,), dtype=torch.uint8, device=device),
            torch.empty((0,), dtype=torch.bool, device=device),
            1,
        )
    desc = torch.tensor(descriptors, dtype=torch.int64, device=device)
    staging = torch.empty(total, dtype=torch.uint8, device=device)
    enabled = torch.empty(len(descriptors), dtype=torch.bool, device=device)
    bn = 1 << max(int(plan.n) - 1, 0).bit_length()
    freeze_node_pages[(len(descriptors), max_rows)](
        desc,
        plan.ancestors,
        plan.depths,
        plan.start,
        plan.req_ids,
        winners,
        active,
        staging,
        enabled,
        plan.n,
        # MAX_Q is the Ancestors row stride and the depth-priority base. It is
        # max_depth + 1, not the target query width (which is budget + 1 and
        # can be larger when budget > depth).
        int(plan.ancestors.shape[1]),
        bn,
        256,
    )
    return desc, staging, enabled, max_rows


def commit_nodes(prepared) -> None:
    """Scatter the frozen staging rows into the live compressed cache pages."""
    try:
        from .node_commit import commit_node_pages
    except ModuleNotFoundError as exc:
        raise NotImplementedError(
            "DSpark tree node commit requires Triton (commit_node_pages); "
            "install triton to commit node pages."
        ) from exc
    desc, staging, enabled, max_rows = prepared
    if desc.shape[0]:
        commit_node_pages[(len(desc), max_rows)](desc, staging, enabled, 256)
