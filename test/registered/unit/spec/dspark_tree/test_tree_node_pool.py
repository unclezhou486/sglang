"""CPU tests for the DSV4 tree node plan and device-op boundaries."""

import pytest
import torch

from tree.config import resolve_tree_config
from tree.node_pool import build_node_plan, compress_nodes, tree_compressor_ops_available
from tree.round import plan_tree_pre


def _pre():
    cfg = resolve_tree_config(enable=True, topk=2, budget=4, depth=3)
    return plan_tree_pre(
        base_logits=torch.zeros(1, 3, 8),
        root_token_ids=torch.tensor([1]),
        prefix_lens=torch.tensor([3]),
        block_table=torch.tensor([[0, 1, 2, 3]]),
        block_size=4,
        window=4,
        config=cfg,
    )


def test_node_plan_matches_metadata():
    pre = _pre()
    plan = build_node_plan(pre)
    meta = pre.metadata
    assert plan.n == meta.verify_ids.numel()
    assert plan.start.tolist() == meta.tree_prefix.tolist()
    assert plan.depths.tolist() == [
        int(p) - int(s) for p, s in zip(meta.positions, meta.tree_prefix)
    ]
    assert torch.equal(plan.ancestors, meta.tree_ancestors.long())
    assert plan.query_cu.tolist() == list(range(plan.n + 1))
    # Every node uses its own hidden row (vllm Hidden[node] = node).
    assert plan.hidden_ids.tolist() == list(range(plan.n))


def test_device_boundary_is_explicit():
    pre = _pre()
    plan = build_node_plan(pre)
    if tree_compressor_ops_available():
        pytest.skip("vendor tree compressor ops are installed")
    with pytest.raises(NotImplementedError, match="compressor_tree"):
        compress_nodes(
            module=None,
            hidden=torch.zeros(plan.n, 8),
            state=torch.zeros(4, 2, 16),
            table=torch.zeros(1, 4, dtype=torch.int32),
            cos=torch.zeros(8, 64),
            sin=torch.zeros(8, 64),
            plan=plan,
        )
