"""Pure config resolution for the DSpark tree path."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TreeConfig:
    enabled: bool
    depth: int
    budget: int
    topk: int
    verify_width: int


def resolve_tree_config(
    *,
    enable: bool,
    topk: int | None,
    budget: int | None,
    depth: int,
    ragged_mode: str = "static",
) -> TreeConfig | None:
    """Resolve and validate the tree knobs. Returns None when disabled.

    ``topk`` is the per-level branching factor; ``budget`` counts non-root
    nodes. topk=1 is the chain: it requires budget == depth and does not use
    the tree attention path.
    """
    if not enable:
        return None
    if budget is None:
        raise ValueError("DSpark tree mode requires --speculative-dspark-tree-budget")
    if topk is None:
        topk = 2
    topk = int(topk)
    budget = int(budget)
    depth = int(depth)
    if topk < 1:
        raise ValueError(f"tree topk must be >= 1, got {topk}")
    if depth < 1:
        raise ValueError(f"tree depth must be >= 1, got {depth}")
    if budget < depth:
        raise ValueError(f"tree budget {budget} must be >= draft depth {depth}")
    if topk == 1 and budget != depth:
        raise ValueError(
            f"topk=1 is the chain and requires budget == depth ({depth}), got {budget}"
        )
    if ragged_mode != "static":
        raise ValueError(
            f"DSpark tree v1 requires SGLANG_RAGGED_VERIFY_MODE=static, got {ragged_mode!r}"
        )
    verify_width = 1 + (depth if topk == 1 else budget)
    return TreeConfig(
        enabled=True,
        depth=depth,
        budget=budget,
        topk=topk,
        verify_width=verify_width,
    )
