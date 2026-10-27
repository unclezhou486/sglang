"""CPU tests for the tree c4 indexer adapter."""

import pytest
import torch

from tree.config import resolve_tree_config
from tree.indexer import (
    tree_indexer,
    tree_indexer_ops_available,
    tree_indexer_select_all,
)
from tree.node_pool import build_node_plan
from tree.round import plan_tree_pre


def _plan():
    cfg = resolve_tree_config(enable=True, topk=2, budget=4, depth=3)
    pre = plan_tree_pre(
        base_logits=torch.zeros(1, 3, 8),
        root_token_ids=torch.tensor([1]),
        prefix_lens=torch.tensor([3]),
        block_table=torch.tensor([[0, 1, 2, 3]]),
        block_size=4,
        window=4,
        config=cfg,
    )
    return build_node_plan(pre)


def test_select_all_short_history():
    plan = _plan()
    out = tree_indexer_select_all(plan, topk=4, device=torch.device("cpu"))
    assert out.shape == (plan.n, 1, 4)
    # start=3, depths=0/1/2 -> lengths = 1 for all rows, so only index 0 valid
    assert out[0, 0, 0].item() == 0
    assert out[0, 0, 1].item() == -1


def test_adapter_raises_without_vendor_op(monkeypatch):
    monkeypatch.delenv("SGLANG_DSPARK_TREE_OPS", raising=False)
    if tree_indexer_ops_available():
        pytest.skip("vendor tree indexer ops are installed")
    plan = _plan()
    with pytest.raises(NotImplementedError, match="quant_lightning_indexer_tree"):
        tree_indexer(
            query=torch.zeros(plan.n, 1, 8),
            weights=torch.zeros(plan.n, 1),
            query_scale=torch.zeros(plan.n, 1),
            history=torch.zeros(4, 8),
            history_scale=torch.zeros(4, 1),
            nodes=torch.zeros(plan.n, 8),
            node_scale=torch.zeros(plan.n, 1),
            table=torch.zeros(1, 4, dtype=torch.int32),
            plan=plan,
            topk=4,
        )
