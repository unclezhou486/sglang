"""CPU tests for the DSV4 tree node plan and device-op boundaries."""

import pytest
import torch

from tree.config import resolve_tree_config
from tree.node_pool import (
    build_node_descriptors,
    build_node_plan,
    compress_nodes,
    prepare_nodes,
    record_compressed_nodes,
    tree_compressor_ops_available,
)
from tree.round import plan_tree_pre
from tree.torch_ops import ensure_tree_ops, load_tree_ops


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


def test_node_plan_globalizes_ancestors_per_request():
    cfg = resolve_tree_config(enable=True, topk=2, budget=4, depth=3)
    pre = plan_tree_pre(
        base_logits=torch.zeros(2, 3, 8),
        root_token_ids=torch.tensor([1, 2]),
        prefix_lens=torch.tensor([3, 5]),
        block_table=torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        block_size=4,
        window=4,
        config=cfg,
    )
    plan = build_node_plan(pre)
    meta = pre.metadata
    offsets = meta.node_req_ids.to(torch.int64) * meta.query_width
    expected = torch.where(
        meta.tree_ancestors >= 0,
        meta.tree_ancestors + offsets[:, None],
        meta.tree_ancestors,
    )
    assert torch.equal(plan.ancestors, expected)


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


def test_record_and_descriptors():
    pre = _pre()
    plan = build_node_plan(pre)
    assert plan.max_q == pre.metadata.query_width
    original = torch.zeros(4, 4, 1, 8, dtype=torch.float32)
    node_data = torch.zeros(plan.n, 1, 8, dtype=torch.float32)
    table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    record = record_compressed_nodes(
        plan, original=original, node_data=node_data, table=table, ratio=4
    )
    assert record.pages == (1,)
    descriptors, total, max_rows = build_node_descriptors([record])
    assert len(descriptors) == 1
    page_ptr, row_stride, row_bytes, rows, node_ptr, table_ptr, width, page, off, ratio = (
        descriptors[0]
    )
    assert row_bytes == 8 * 4
    assert rows == 4
    assert width == 4
    assert page == 1
    assert ratio == 4
    assert total == 4 * (8 * 4)
    assert max_rows == 4


def test_record_state_ratio_one():
    pre = _pre()
    plan = build_node_plan(pre)
    state = torch.zeros(4, 4, 8, dtype=torch.float32)
    nodes = torch.zeros(plan.n, 1, 1, 8, dtype=torch.float32)
    table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    record = record_compressed_nodes(
        plan, original=state, node_data=nodes, table=table, ratio=1
    )
    assert record.ratio == 1
    assert record.pages == (1, 2)
    descriptors, total, max_rows = build_node_descriptors([record])
    assert descriptors[0][2] == 8 * 4
    assert descriptors[0][3] == 4
    assert total == 2 * 4 * (8 * 4)
    assert max_rows == 4


def test_prepare_requires_triton_on_cpu():
    try:
        import triton  # noqa: F401
    except ModuleNotFoundError:
        triton_installed = False
    else:
        triton_installed = True
    if triton_installed:
        pytest.skip("triton is installed")
    pre = _pre()
    plan = build_node_plan(pre)
    record = record_compressed_nodes(
        plan,
        original=torch.zeros(4, 4, 1, 8, dtype=torch.float32),
        node_data=torch.zeros(plan.n, 1, 8, dtype=torch.float32),
        table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        ratio=4,
    )
    with pytest.raises(NotImplementedError, match="Triton"):
        prepare_nodes(
            plan,
            [record],
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.bool),
            torch.device("cpu"),
        )


def test_torch_ops_no_env(monkeypatch):
    monkeypatch.delenv("SGLANG_DSPARK_TREE_OPS", raising=False)
    assert load_tree_ops("") == ()
    assert ensure_tree_ops() is False
