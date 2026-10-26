"""Greedy tree acceptance over a finalized TreeLayout.

Device tensors only; no host readback and no data-dependent Python loops over
requests. The walk follows first_child/next_sibling and compares each child's
token with the parent's target argmax, matching EAGLE/DFlash tree semantics.
"""

from dataclasses import dataclass

import torch


@dataclass
class TreeAcceptance:
    path_node_ids: torch.Tensor  # [r, budget], -1 padded
    num_sampled: torch.Tensor  # [r], includes bonus
    num_accepted: torch.Tensor  # [r], draft nodes only
    endpoint_node: torch.Tensor  # [r], last accepted node or 0
    bonus: torch.Tensor  # [r], target token after the endpoint
    active: torch.Tensor  # [r] bool


def greedy_tree_accept(
    tree,
    target_predict: torch.Tensor,
    *,
    active: torch.Tensor | None = None,
) -> TreeAcceptance:
    """Walk the draft tree against per-node target argmax tokens.

    ``target_predict`` is [batch, budget + 1]; column 0 is the root's target
    argmax and column i+1 is node i's target argmax.
    """
    if tree.tokens.dim() != 2:
        raise ValueError("tree.tokens must be [batch, budget]")
    batch, budget = tree.tokens.shape
    if target_predict.shape != (batch, budget + 1):
        raise ValueError(
            f"target_predict must be [{batch}, {budget + 1}], "
            f"got {tuple(target_predict.shape)}"
        )
    device = tree.tokens.device
    rows = torch.arange(batch, device=device)
    if active is None:
        active = torch.ones(batch, dtype=torch.bool, device=device)
    alive = active.to(torch.bool).clone()
    current = torch.zeros(batch, dtype=torch.long, device=device)
    path = torch.full((batch, budget), -1, dtype=torch.long, device=device)
    num_accepted = torch.zeros(batch, dtype=torch.int32, device=device)
    num_sampled = active.to(torch.int32).clone()

    for step in range(budget):
        target = target_predict[rows, current.clamp(0, budget)]
        child = tree.first_child[rows, current.clamp(0, budget)].long()
        found = torch.full_like(current, -1)
        for _ in range(budget):
            linked = alive & (found < 0) & (child > 0) & (child <= budget)
            token = tree.tokens[rows, (child - 1).clamp(0, budget - 1)]
            match = linked & (token == target)
            found = torch.where(match, child, found)
            nxt = tree.next_sibling[rows, child.clamp(0, budget)].long()
            child = torch.where(linked & ~match, nxt, child)
        matched = alive & (found > 0)
        path[:, step] = torch.where(matched, found, -1)
        current = torch.where(matched, found, current)
        num_accepted += matched.to(torch.int32)
        num_sampled += matched.to(torch.int32)
        alive = matched

    bonus = target_predict[rows, current.clamp(0, budget)]
    bonus = torch.where(active, bonus, torch.zeros_like(bonus))
    endpoint = torch.where(active, current, torch.zeros_like(current))
    return TreeAcceptance(
        path_node_ids=path,
        num_sampled=num_sampled,
        num_accepted=num_accepted,
        endpoint_node=endpoint,
        bonus=bonus,
        active=active.to(torch.bool),
    )
