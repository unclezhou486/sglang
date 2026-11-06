"""CPU tests for the tree attention argument contract."""

from unittest import mock

import pytest
import torch

from tree.attention import build_tree_attn_kwargs, build_tree_attn_metadata
from tree.config import resolve_tree_config
from tree.node_pool import build_node_plan
from tree.round import plan_tree_pre


def _base_kwargs():
    return dict(
        q=torch.zeros(9, 1, 8),
        ori_kv=torch.zeros(4, 1, 8),
        ori_sparse_indices=torch.zeros(9, 1, 128, dtype=torch.int32),
        cu_seqlens_q=torch.arange(10, dtype=torch.int32),
        seqused_kv=torch.full((9,), 3, dtype=torch.int32),
        tree_ancestors=torch.zeros(9, 2, dtype=torch.long),
        tree_prefix=torch.zeros(9, dtype=torch.int32),
        cmp_ratio=1,
        softmax_scale=0.1,
    )


def test_swa_only_drops_tree_fields_without_cmp_tree_kv():
    out = build_tree_attn_kwargs(**_base_kwargs())
    assert out["cmp_tree_kv"] is None
    assert out["tree_ancestors"] is None
    assert out["tree_prefix"] is None


def test_cmp_tree_kv_requires_ancestors_and_prefix():
    kwargs = _base_kwargs()
    kwargs["cmp_tree_kv"] = torch.zeros(9, 8)
    kwargs["tree_ancestors"] = None
    with pytest.raises(ValueError, match="tree_ancestors"):
        build_tree_attn_kwargs(**kwargs)


def test_tree_attention_metadata_is_cached_per_plan():
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
    plan = build_node_plan(pre)
    kwargs = dict(
        query=torch.zeros(plan.n, 1, 8),
        metadata=pre.metadata,
        plan=plan,
        ratio=4,
        window=4,
        cmp_sparse_indices=None,
        compressed=None,
    )
    with mock.patch(
        "tree.attention.tree_backend", return_value="sgl_kernel_npu"
    ):
        first = build_tree_attn_metadata(**kwargs)
        second = build_tree_attn_metadata(**kwargs)
    assert second[0] is first[0]
    assert second[1] is first[1]
    assert second[2] is first[2]


def test_cmp_tree_kv_keeps_explicit_ancestors_and_prefix():
    kwargs = _base_kwargs()
    kwargs["cmp_tree_kv"] = torch.zeros(9, 8)
    out = build_tree_attn_kwargs(**kwargs)
    assert out["tree_ancestors"] is not None
    assert out["tree_prefix"] is not None
