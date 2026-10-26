"""Level-wise tree builder with ancestor-closed pruning.

Adapted from vllm-ascend's tree/builder.py (Apache-2.0), torch path only.
"""

from dataclasses import dataclass

import torch

from tree.layout import TreeLayout, finalize_tree_layout_torch
from tree.scoring import IndependentScorer


@dataclass
class TreeBuildState:
    tokens: torch.Tensor  # [r, capacity] non-root node tokens
    depths: torch.Tensor  # [r, capacity] non-root node depths
    parents: torch.Tensor  # [r, capacity + 1] node-id indexed, root at 0
    path_scores: torch.Tensor  # [r, capacity + 1] cumulative log prob
    width: int
    num_nodes: int = 0


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


class TreeBuilder:
    """Build a tree of at most ``budget`` non-root nodes.

    At each draft depth the builder expands every frontier node by up to
    ``topk`` candidates and keeps the best ``width = min(topk, vocab, budget)``
    candidates globally. After all depths it keeps the best ``budget`` nodes by
    cumulative score. Because a child's cumulative score can never exceed its
    parent's, the kept set is ancestor-closed.
    """

    def __init__(self, budget: int, topk: int, depth: int, scorer=None):
        if budget < 1 or topk < 1 or depth < 1:
            raise ValueError("Tree budget, topk and depth must be positive")
        if budget < depth:
            raise ValueError("Tree budget must be >= draft depth")
        self.budget = int(budget)
        self.topk = int(topk)
        self.depth = int(depth)
        self.scorer = scorer if scorer is not None else IndependentScorer(topk)
        self.num_selected_nodes = 0

    def build(
        self,
        base_logits: torch.Tensor,
        root_token_ids: torch.Tensor,
        *,
        out: TreeLayout | None = None,
    ) -> TreeLayout:
        if base_logits.dim() != 3:
            raise ValueError("base_logits must be [batch, depth, vocab]")
        r, depth, vocab = base_logits.shape
        if depth != self.depth:
            raise ValueError(f"base_logits depth {depth} != builder depth {self.depth}")
        if root_token_ids.shape[0] != r:
            raise ValueError("root_token_ids batch mismatch")
        width = min(self.topk, vocab, self.budget)
        capacity = width * depth
        device = base_logits.device

        state = TreeBuildState(
            tokens=torch.full((r, capacity), -1, dtype=torch.long, device=device),
            depths=torch.zeros((r, capacity), dtype=torch.long, device=device),
            parents=torch.full(
                (r, capacity + 1), -1, dtype=torch.long, device=device
            ),
            path_scores=torch.zeros((r, capacity + 1), dtype=torch.float32, device=device),
            width=width,
        )
        state.parents[:, 0] = 0
        frontier = torch.zeros((r, 1), dtype=torch.long, device=device)

        for depth_slot in range(depth):
            if depth_slot == 0:
                frontier_tokens = root_token_ids.long().unsqueeze(1)
            else:
                frontier_tokens = state.tokens.gather(1, frontier - 1)
            layer = self.scorer.score(
                base_logits[:, depth_slot], frontier_tokens, width
            )
            parent_scores = state.path_scores.gather(1, frontier)
            flat = (parent_scores.unsqueeze(-1) + layer.scores).reshape(r, -1)
            selected = flat.argsort(dim=-1, descending=True, stable=True)[:, :width]

            start = state.num_nodes
            node_ids = torch.arange(
                start + 1, start + width + 1, device=device, dtype=torch.long
            )
            parent_slots = selected // width
            state.tokens[:, start : start + width] = layer.token_ids.reshape(
                r, -1
            ).gather(1, selected)
            state.depths[:, start : start + width] = depth_slot + 1
            state.parents[:, start + 1 : start + width + 1] = frontier.gather(
                1, parent_slots
            )
            state.path_scores[:, start + 1 : start + width + 1] = flat.gather(
                1, selected
            )
            state.num_nodes += width
            frontier = node_ids.unsqueeze(0).expand(r, -1)

        keep_count = min(self.budget, state.num_nodes)
        priority = state.path_scores[:, 1 : state.num_nodes + 1]
        kept = priority.argsort(dim=-1, descending=True, stable=True)[
            :, :keep_count
        ]
        kept_ids = (kept + 1).sort(dim=-1).values
        old_to_new = torch.zeros(
            (r, state.num_nodes + 1), dtype=torch.long, device=device
        )
        old_to_new.scatter_(
            1,
            kept_ids,
            torch.arange(1, keep_count + 1, device=device)
            .unsqueeze(0)
            .expand(r, -1),
        )
        new_parents = old_to_new.gather(1, state.parents.gather(1, kept_ids))
        new_tokens = state.tokens.gather(1, kept_ids - 1)
        new_depths = state.depths.gather(1, kept_ids - 1)

        self.num_selected_nodes = keep_count
        if out is None:
            out = _empty_layout(r, keep_count, device)
        return finalize_tree_layout_torch(
            out, new_tokens, new_depths, new_parents, keep_count
        )
