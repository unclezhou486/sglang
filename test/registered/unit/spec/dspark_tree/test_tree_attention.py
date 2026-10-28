"""CPU tests for the tree attention argument contract."""

import pytest
import torch

from tree.attention import build_tree_attn_kwargs


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


def test_cmp_tree_kv_keeps_explicit_ancestors_and_prefix():
    kwargs = _base_kwargs()
    kwargs["cmp_tree_kv"] = torch.zeros(9, 8)
    out = build_tree_attn_kwargs(**kwargs)
    assert out["tree_ancestors"] is not None
    assert out["tree_prefix"] is not None
