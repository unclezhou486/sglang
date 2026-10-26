"""Path gather and accept-index construction for tree verify outputs."""

from dataclasses import dataclass

import torch


@dataclass
class TreeAcceptOuts:
    out_tokens: torch.Tensor  # [r, query_width]
    accept_index: torch.Tensor  # [r, query_width], flat verify-row indices
    commit_lens: torch.Tensor  # [r], root + accepted drafts
    new_seq_lens: torch.Tensor  # [r], prefix + commit_lens
    num_accepted: torch.Tensor  # [r]


def build_tree_accept_outs(
    verify_ids: torch.Tensor,
    path_node_ids: torch.Tensor,
    bonus: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> TreeAcceptOuts:
    """Gather accepted node tokens in depth order and append the bonus token.

    ``verify_ids`` is [r, 1 + budget] with column 0 the accepted root/anchor.
    ``path_node_ids`` is [r, budget] with node ids in 1..budget or -1.
    """
    if verify_ids.dim() != 2 or path_node_ids.dim() != 2:
        raise ValueError("verify_ids and path_node_ids must be 2-D")
    batch, query_width = verify_ids.shape
    if path_node_ids.shape[0] != batch:
        raise ValueError("path_node_ids batch mismatch")
    if bonus.shape[0] != batch or prefix_lens.shape[0] != batch:
        raise ValueError("bonus/prefix_lens batch mismatch")
    if query_width != path_node_ids.shape[1] + 1:
        raise ValueError(
            "verify_ids width must be 1 + budget "
            f"(got {query_width} vs budget {path_node_ids.shape[1]})"
        )

    device = verify_ids.device
    rows = torch.arange(batch, device=device)
    valid = path_node_ids >= 0
    num_accepted = valid.sum(dim=1).to(torch.int32)

    out_tokens = torch.zeros((batch, query_width), dtype=verify_ids.dtype, device=device)
    accept_index = torch.full(
        (batch, query_width), -1, dtype=torch.long, device=device
    )
    for depth in range(path_node_ids.shape[1]):
        mask = valid[:, depth]
        if not bool(mask.any()):
            continue
        selected_rows = rows[mask]
        nodes = path_node_ids[mask, depth].long()
        out_tokens[selected_rows, depth] = verify_ids[selected_rows, nodes]
        accept_index[selected_rows, depth] = selected_rows * query_width + nodes

    out_tokens[rows, num_accepted.long()] = bonus
    commit_lens = num_accepted + 1
    new_seq_lens = prefix_lens.to(torch.int64) + commit_lens.to(torch.int64)
    return TreeAcceptOuts(
        out_tokens=out_tokens,
        accept_index=accept_index,
        commit_lens=commit_lens,
        new_seq_lens=new_seq_lens,
        num_accepted=num_accepted,
    )
