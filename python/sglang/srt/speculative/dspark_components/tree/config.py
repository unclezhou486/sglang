"""Pure config resolution for the DSpark tree path."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TreeConfig:
    enabled: bool
    depth: int
    budget: int
    topk: int
    verify_width: int


@dataclass(frozen=True)
class DSparkWidths:
    """Draft depth and target-verify width are two different quantities.

    ``gamma`` is the draft horizon (``--speculative-dspark-block-size``).
    ``verify_width`` is the number of target-verify rows per request; it is
    ``1 + budget`` for a tree and ``gamma + 1`` for a chain.
    """

    gamma: int
    verify_width: int
    tree: TreeConfig | None


def resolve_dspark_widths(
    *,
    explicit_block_size: int | None,
    checkpoint_gamma: int | None,
    num_draft_tokens: int | None,
    tree_enable: bool,
    tree_topk: int | None,
    tree_budget: int | None,
    ragged_mode: str = "static",
    default_gamma: int = 7,
) -> DSparkWidths:
    """Resolve gamma and the target-verify width together.

    A tree with ``topk > 1`` may have ``budget > gamma``; its verify width is
    then ``1 + budget`` while the draft model still proposes ``gamma`` tokens.
    A chain (``topk == 1``) keeps ``budget == gamma`` and ``gamma + 1`` rows.
    Non-tree DSpark keeps ``gamma + 1`` rows.
    """
    gamma = explicit_block_size if explicit_block_size is not None else checkpoint_gamma
    if tree_enable:
        if gamma is None:
            raise ValueError(
                "DSpark tree mode requires a known draft depth (gamma): set "
                "--speculative-dspark-block-size or use a draft checkpoint with "
                "a block_size."
            )
        tree = resolve_tree_config(
            enable=True,
            topk=tree_topk,
            budget=tree_budget,
            depth=int(gamma),
            ragged_mode=ragged_mode,
        )
        assert tree is not None
        verify_width = tree.verify_width
    else:
        if gamma is None:
            if num_draft_tokens is not None:
                gamma = int(num_draft_tokens) - 1
            else:
                gamma = int(default_gamma)
        gamma = int(gamma)
        if gamma < 1:
            raise ValueError(
                "DSpark gamma must be >= 1, got "
                f"{gamma} (num_draft_tokens={num_draft_tokens})."
            )
        tree = None
        verify_width = gamma + 1

    if num_draft_tokens is not None and int(num_draft_tokens) != verify_width:
        width_src = (
            f"1 + tree budget (={verify_width})"
            if tree is not None and tree.topk > 1
            else f"gamma + 1 (={verify_width})"
        )
        raise ValueError(
            "DSpark speculative_num_draft_tokens must equal "
            f"{width_src}, but got {num_draft_tokens}."
        )

    return DSparkWidths(gamma=int(gamma), verify_width=int(verify_width), tree=tree)


def resolve_draft_query_width(*, gamma: int, sample_from_anchor: bool) -> int:
    """Draft-block query rows: anchor sampling uses gamma; otherwise gamma + 1."""
    return int(gamma) if sample_from_anchor else int(gamma) + 1


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
    the tree attention path. For topk>1, budget may exceed depth, up to the
    builder capacity ``topk * depth``; the target-verify width is ``1 + budget``.
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
    if topk > 1 and budget > topk * depth:
        raise ValueError(
            f"tree budget {budget} exceeds builder capacity topk * depth = "
            f"{topk * depth}; increase --speculative-eagle-topk or lower the budget"
        )
    if ragged_mode != "static":
        raise ValueError(
            f"DSpark tree requires SGLANG_RAGGED_VERIFY_MODE=static, got {ragged_mode!r}"
        )
    verify_width = 1 + (depth if topk == 1 else budget)
    return TreeConfig(
        enabled=True,
        depth=depth,
        budget=budget,
        topk=topk,
        verify_width=verify_width,
    )


def resolve_tree_for_server(server_args, *, gamma: int, ragged_mode: str = "static"):
    """Resolve tree knobs from a ServerArgs-like object (getattr-safe)."""
    return resolve_tree_config(
        enable=bool(getattr(server_args, "speculative_dspark_tree_enable", False)),
        topk=getattr(server_args, "speculative_eagle_topk", None),
        budget=getattr(server_args, "speculative_dspark_tree_budget", None),
        depth=gamma,
        ragged_mode=ragged_mode,
    )
