"""Flattened draft-tree layout and torch-only link/visibility finalization.

Adapted from vllm-ascend's tree/layout.py (Apache-2.0). Only the torch path is
kept; the NPU Triton kernel is a later optimization.
"""

from dataclasses import dataclass

import torch


@dataclass
class TreeLayout:
    """Flattened draft tree for a batch of parallel-draft logits.

    Node ids include the root (root=0, draft nodes 1..N). ``tokens``,
    ``depths`` and ``parents`` are slot-indexed for non-root nodes.
    ``first_child`` and ``next_sibling`` are node-id-indexed and include the
    root. ``visibility[i, j]`` means non-root node i+1 may see non-root node
    j+1; seeing the root is implicit.
    """

    tokens: torch.Tensor
    depths: torch.Tensor
    parents: torch.Tensor
    num_nodes: torch.Tensor
    visibility: torch.Tensor
    first_child: torch.Tensor
    next_sibling: torch.Tensor


def _empty_layout(batch_size: int, budget: int, device) -> TreeLayout:
    return TreeLayout(
        tokens=torch.full((batch_size, budget), -1, dtype=torch.long, device=device),
        depths=torch.zeros((batch_size, budget), dtype=torch.long, device=device),
        parents=torch.full((batch_size, budget), -1, dtype=torch.long, device=device),
        num_nodes=torch.zeros(batch_size, dtype=torch.int32, device=device),
        visibility=torch.zeros(
            (batch_size, budget, budget), dtype=torch.bool, device=device
        ),
        first_child=torch.full(
            (batch_size, budget + 1), -1, dtype=torch.long, device=device
        ),
        next_sibling=torch.full(
            (batch_size, budget + 1), -1, dtype=torch.long, device=device
        ),
    )


def finalize_tree_layout_torch(
    out: TreeLayout,
    tokens: torch.Tensor,
    depths: torch.Tensor,
    parent_ids: torch.Tensor,
    num_nodes: int,
) -> TreeLayout:
    """Fill ``out`` with selected nodes and derive sibling links + visibility."""
    if num_nodes < 0 or num_nodes > out.tokens.shape[1]:
        raise ValueError(f"num_nodes {num_nodes} outside budget {out.tokens.shape[1]}")
    num_reqs = tokens.shape[0]
    out.tokens.fill_(-1)
    out.depths.zero_()
    out.parents.fill_(-1)
    out.visibility.zero_()
    out.first_child.fill_(-1)
    out.next_sibling.fill_(-1)

    if num_nodes:
        out.tokens[:, :num_nodes] = tokens[:, :num_nodes].to(out.tokens.dtype)
        out.depths[:, :num_nodes] = depths[:, :num_nodes].to(out.depths.dtype)
        out.parents[:, :num_nodes] = parent_ids[:, :num_nodes].to(out.parents.dtype)
    out.num_nodes[:] = num_nodes

    rows = torch.arange(num_reqs, device=parent_ids.device)
    for slot in range(num_nodes):
        node_id = slot + 1
        parent_id = parent_ids[:, slot].long()
        # first_child holds the most recently inserted child; next_sibling
        # chains back to the previously inserted one.
        previous_first = out.first_child.gather(1, parent_id.unsqueeze(1)).squeeze(1)
        out.next_sibling[:, node_id] = previous_first
        out.first_child[rows, parent_id] = node_id

        if slot > 0:
            src_idx = (parent_id - 1).clamp(min=0)
            parent_vis = out.visibility[:, :num_nodes, :slot].gather(
                1,
                src_idx.view(-1, 1, 1).expand(-1, 1, slot),
            ).squeeze(1)
            out.visibility[:, slot, :slot] = torch.where(
                (parent_id > 0).unsqueeze(-1),
                parent_vis,
                out.visibility[:, slot, :slot],
            )
        out.visibility[:, slot, slot] = True

    return out


def build_chain_layout(tokens: torch.Tensor) -> TreeLayout:
    """Build the topk=1 chain layout: node i+1's parent is node i."""
    if tokens.dim() != 2 or tokens.shape[1] < 1:
        raise ValueError("chain tokens must be [batch, depth] with depth >= 1")
    batch_size, depth = tokens.shape
    device = tokens.device
    depths = torch.arange(1, depth + 1, device=device).expand(batch_size, -1)
    parents = torch.cat(
        (
            torch.zeros((batch_size, 1), dtype=torch.long, device=device),
            torch.arange(1, depth, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1),
        ),
        dim=1,
    )
    out = _empty_layout(batch_size, depth, device)
    return finalize_tree_layout_torch(out, tokens, depths, parents, depth)
