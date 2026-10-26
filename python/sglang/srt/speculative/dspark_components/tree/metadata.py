"""Pure DSV4 per-node verify geometry for the DSpark tree path.

All functions are torch-only and host-side; they never read device scalars.
"""

from dataclasses import dataclass

import torch


@dataclass
class TreeMetadata:
    positions: torch.Tensor  # [T] logical position per query node
    node_seqused_kv: torch.Tensor  # [T] visible KV length per node
    verify_ids: torch.Tensor  # [T] query token ids (anchor + nodes)
    tree_prefix: torch.Tensor  # [T] per-node prefix base
    tree_ancestors: torch.Tensor  # [T, A] root-first node ids, -1 padded
    node_req_ids: torch.Tensor  # [T] request index per node
    node_cu_seqlens_q: torch.Tensor  # [T + 1]
    request_cu_seqlens_q: torch.Tensor  # [bs + 1]
    request_seqused_kv: torch.Tensor  # [bs]
    query_width: int
    num_nodes: int


def _ancestor_row(layout, req: int, node_id: int, max_depth: int) -> list[int]:
    if node_id <= 0:
        row = [0]
    else:
        row = []
        cur = node_id
        while cur > 0:
            row.append(int(cur))
            cur = int(layout.parents[req, cur - 1])
        row.append(0)
        row.reverse()
    if len(row) > max_depth + 1:
        raise ValueError(
            f"node {node_id} has {len(row)} ancestors but only {max_depth + 1} slots"
        )
    row.extend([-1] * (max_depth + 1 - len(row)))
    return row


def build_tree_metadata(
    layout,
    *,
    prefix_lens: torch.Tensor,
    anchor_tokens: torch.Tensor,
) -> TreeMetadata:
    """Build per-node verify geometry for a finalized TreeLayout."""
    batch, budget = layout.tokens.shape
    if prefix_lens.shape[0] != batch or anchor_tokens.shape[0] != batch:
        raise ValueError("prefix_lens/anchor_tokens batch mismatch")
    device = layout.tokens.device
    query_width = budget + 1
    num_nodes = int(layout.num_nodes.max().item()) if batch else 0
    max_depth = int(layout.depths.max().item()) if budget else 0

    depths = torch.cat(
        (
            torch.zeros((batch, 1), dtype=torch.long, device=device),
            layout.depths.long(),
        ),
        dim=1,
    )
    verify_ids_2d = torch.cat(
        (anchor_tokens.long().unsqueeze(1), layout.tokens.long()), dim=1
    )
    prefix_2d = prefix_lens.long().unsqueeze(1).expand(batch, query_width)
    positions_2d = prefix_2d + depths
    seqused_2d = positions_2d + 1

    ancestors = torch.full(
        (batch, query_width, max_depth + 1),
        -1,
        dtype=torch.long,
        device=device,
    )
    for req in range(batch):
        for node_id in range(query_width):
            ancestors[req, node_id] = torch.tensor(
                _ancestor_row(layout, req, node_id, max_depth),
                dtype=torch.long,
                device=device,
            )

    node_req_ids = torch.arange(batch, device=device).repeat_interleave(query_width)
    node_cu = torch.arange(
        batch * query_width + 1, dtype=torch.int32, device=device
    )
    request_cu = (
        torch.arange(batch + 1, dtype=torch.int32, device=device) * query_width
    )
    request_seqused = prefix_lens.to(torch.int32) + query_width
    return TreeMetadata(
        positions=positions_2d.reshape(-1),
        node_seqused_kv=seqused_2d.reshape(-1).to(torch.int32),
        verify_ids=verify_ids_2d.reshape(-1),
        tree_prefix=prefix_2d.reshape(-1).to(torch.int32),
        tree_ancestors=ancestors.reshape(batch * query_width, max_depth + 1),
        node_req_ids=node_req_ids.to(torch.int32),
        node_cu_seqlens_q=node_cu,
        request_cu_seqlens_q=request_cu,
        request_seqused_kv=request_seqused,
        query_width=query_width,
        num_nodes=num_nodes,
    )


def build_chain_metadata(
    *,
    prefix_lens: torch.Tensor,
    depth: int,
    anchor_tokens: torch.Tensor,
) -> TreeMetadata:
    """Chain geometry (topk=1); verify_ids are only the anchor."""
    batch = prefix_lens.shape[0]
    device = prefix_lens.device
    query_width = depth + 1
    prefix_2d = prefix_lens.long().unsqueeze(1).expand(batch, query_width)
    positions = prefix_2d + torch.arange(query_width, device=device)
    ancestors = torch.full(
        (batch, query_width, depth + 1), -1, dtype=torch.long, device=device
    )
    for node_id in range(query_width):
        ancestors[:, node_id, : node_id + 1] = torch.arange(
            node_id + 1, device=device
        )
    verify_ids = torch.full(
        (batch, query_width), -1, dtype=torch.long, device=device
    )
    verify_ids[:, 0] = anchor_tokens.long()
    return TreeMetadata(
        positions=positions.reshape(-1),
        node_seqused_kv=(positions + 1).reshape(-1).to(torch.int32),
        verify_ids=verify_ids.reshape(-1),
        tree_prefix=prefix_2d.reshape(-1).to(torch.int32),
        tree_ancestors=ancestors.reshape(batch * query_width, depth + 1),
        node_req_ids=torch.arange(batch, device=device)
        .repeat_interleave(query_width)
        .to(torch.int32),
        node_cu_seqlens_q=torch.arange(
            batch * query_width + 1, dtype=torch.int32, device=device
        ),
        request_cu_seqlens_q=(
            torch.arange(batch + 1, dtype=torch.int32, device=device) * query_width
        ),
        request_seqused_kv=prefix_lens.to(torch.int32) + query_width,
        query_width=query_width,
        num_nodes=depth,
    )
