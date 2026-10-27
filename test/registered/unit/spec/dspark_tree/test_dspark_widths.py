"""CPU tests for DSpark gamma / verify-width resolution (budget > gamma)."""

import pytest

from tree.config import (
    resolve_draft_query_width,
    resolve_dspark_widths,
    resolve_tree_config,
)


def test_tree_budget_above_gamma_uses_budget_width():
    widths = resolve_dspark_widths(
        explicit_block_size=5,
        checkpoint_gamma=None,
        num_draft_tokens=9,
        tree_enable=True,
        tree_topk=2,
        tree_budget=8,
    )
    assert widths.gamma == 5
    assert widths.verify_width == 9
    assert widths.tree is not None
    assert widths.tree.depth == 5
    assert widths.tree.budget == 8
    assert widths.tree.verify_width == 9


def test_tree_width_mismatch_is_rejected():
    with pytest.raises(ValueError, match=r"1 \+ tree budget"):
        resolve_dspark_widths(
            explicit_block_size=5,
            checkpoint_gamma=None,
            num_draft_tokens=6,
            tree_enable=True,
            tree_topk=2,
            tree_budget=8,
        )


def test_tree_requires_known_gamma():
    with pytest.raises(ValueError, match="requires a known draft depth"):
        resolve_dspark_widths(
            explicit_block_size=None,
            checkpoint_gamma=None,
            num_draft_tokens=9,
            tree_enable=True,
            tree_topk=2,
            tree_budget=8,
        )


def test_tree_capacity_is_topk_times_depth():
    with pytest.raises(ValueError, match="builder capacity"):
        resolve_dspark_widths(
            explicit_block_size=5,
            checkpoint_gamma=None,
            num_draft_tokens=11,
            tree_enable=True,
            tree_topk=2,
            tree_budget=11,
        )


def test_tree_budget_equal_gamma_is_still_supported():
    cfg = resolve_tree_config(enable=True, topk=2, budget=5, depth=5)
    assert cfg is not None
    assert cfg.verify_width == 6


def test_chain_requires_budget_equal_depth():
    cfg = resolve_tree_config(enable=True, topk=1, budget=5, depth=5)
    assert cfg is not None
    assert cfg.verify_width == 6
    with pytest.raises(ValueError, match="budget == depth"):
        resolve_tree_config(enable=True, topk=1, budget=6, depth=5)


def test_non_tree_keeps_gamma_plus_one():
    widths = resolve_dspark_widths(
        explicit_block_size=5,
        checkpoint_gamma=None,
        num_draft_tokens=6,
        tree_enable=False,
        tree_topk=None,
        tree_budget=None,
    )
    assert widths.gamma == 5
    assert widths.verify_width == 6
    assert widths.tree is None

    legacy = resolve_dspark_widths(
        explicit_block_size=None,
        checkpoint_gamma=None,
        num_draft_tokens=9,
        tree_enable=False,
        tree_topk=None,
        tree_budget=None,
    )
    assert legacy.gamma == 8
    assert legacy.verify_width == 9


def test_draft_query_width_anchor_rule():
    assert resolve_draft_query_width(gamma=5, sample_from_anchor=True) == 5
    assert resolve_draft_query_width(gamma=5, sample_from_anchor=False) == 6
