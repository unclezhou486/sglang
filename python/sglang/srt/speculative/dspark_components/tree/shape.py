"""Host-only tree dimensions for DSpark parallel drafting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TreeShape:
    """Draft depth, non-root node budget and per-level branching factor.

    ``depth`` is the DSpark draft block size (gamma); ``budget`` counts
    non-root nodes; ``topk`` bounds each retained level. Node 0 is the
    accepted root. This mirrors vllm-ascend's TreeShape contract.
    """

    depth: int
    budget: int
    topk: int
    dsv4_dspark: bool = True

    def __post_init__(self) -> None:
        if self.depth < 1 or self.budget < self.depth or self.topk < 1:
            raise ValueError(
                "Tree dimensions require depth >= 1, budget >= depth and topk >= 1"
            )

    @property
    def target_query_len(self) -> int:
        """Maximum target verify rows per request, including the root."""
        if self.topk == 1:
            return 1 + self.depth
        return 1 + self.budget

    def validate_selected_nodes(self, count: int) -> int:
        if not 0 <= count < self.target_query_len:
            raise ValueError(
                f"Selected tree nodes {count} exceed capacity "
                f"{self.target_query_len - 1}"
            )
        return count
