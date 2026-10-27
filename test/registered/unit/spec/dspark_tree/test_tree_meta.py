"""CPU tests for DSV4 tree metadata, sparse indices, attention reference, commit."""

import os

import pytest
import torch

from tree.layout import TreeLayout, finalize_tree_layout_torch
from tree.metadata import build_chain_metadata, build_tree_metadata
from tree.sparse import build_tree_ori_sparse_indices
from tree.attention import (
    build_tree_attn_kwargs,
    tree_attention,
    tree_attention_reference,
    tree_attn_op_available,
)
from tree.commit import TreeCommitPlan, apply_tree_commit, plan_tree_commit
from tree.config import TreeConfig, resolve_tree_config, resolve_tree_for_server


def _buf(bs, budget):
    return TreeLayout(
        tokens=torch.full((bs, budget), -1, dtype=torch.long),
        depths=torch.zeros((bs, budget), dtype=torch.long),
        parents=torch.full((bs, budget), -1, dtype=torch.long),
        num_nodes=torch.zeros(bs, dtype=torch.int32),
        visibility=torch.zeros((bs, budget, budget), dtype=torch.bool),
        first_child=torch.full((bs, budget + 1), -1, dtype=torch.long),
        next_sibling=torch.full((bs, budget + 1), -1, dtype=torch.long),
    )


def _branch_layout():
    return finalize_tree_layout_torch(
        _buf(1, 4),
        tokens=torch.tensor([[5, 6, 7, 8]]),
        depths=torch.tensor([[1, 1, 2, 2]]),
        parent_ids=torch.tensor([[0, 0, 1, 1]]),
        num_nodes=4,
    )


class TestTreeMetadata:
    def test_positions_and_seqused(self):
        meta = build_tree_metadata(
            _branch_layout(),
            prefix_lens=torch.tensor([10]),
            anchor_tokens=torch.tensor([100]),
        )
        # root at 10; nodes 1/2 at 11; nodes 3/4 at 12
        assert meta.positions.tolist() == [10, 11, 11, 12, 12]
        assert meta.node_seqused_kv.tolist() == [11, 12, 12, 13, 13]
        assert meta.verify_ids.tolist() == [100, 5, 6, 7, 8]

    def test_ancestors_are_root_first(self):
        meta = build_tree_metadata(
            _branch_layout(),
            prefix_lens=torch.tensor([10]),
            anchor_tokens=torch.tensor([100]),
        )
        # rows: root, node1, node2, node3, node4
        assert meta.tree_ancestors.tolist() == [
            [0, -1, -1],
            [0, 1, -1],
            [0, 2, -1],
            [0, 1, 3],
            [0, 1, 4],
        ]

    def test_request_and_node_geometry(self):
        meta = build_tree_metadata(
            _branch_layout(),
            prefix_lens=torch.tensor([10]),
            anchor_tokens=torch.tensor([100]),
        )
        assert meta.node_cu_seqlens_q.tolist() == [0, 1, 2, 3, 4, 5]
        assert meta.request_cu_seqlens_q.tolist() == [0, 5]
        assert meta.request_seqused_kv.tolist() == [15]
        assert meta.tree_prefix.tolist() == [10, 10, 10, 10, 10]
        assert meta.node_req_ids.tolist() == [0, 0, 0, 0, 0]

    def test_chain_metadata(self):
        meta = build_chain_metadata(prefix_lens=torch.tensor([10]), depth=3, anchor_tokens=torch.tensor([100]))
        assert meta.positions.tolist() == [10, 11, 12, 13]
        assert meta.node_seqused_kv.tolist() == [11, 12, 13, 14]
        assert meta.request_cu_seqlens_q.tolist() == [0, 4]
        assert meta.request_seqused_kv.tolist() == [14]
        assert meta.verify_ids.tolist() == [100, -1, -1, -1]


class TestSparseIndices:
    def test_prefix_window_plus_ancestors_and_no_sibling(self):
        layout = _branch_layout()
        block_table = torch.tensor([[0, 1, 2, 3]])
        slots = build_tree_ori_sparse_indices(
            layout,
            prefix_lens=torch.tensor([3]),
            block_table=block_table,
            block_size=4,
            window=4,
            budget=4,
        )
        # width is 128-aligned
        assert slots.shape == (5, 1, 128)
        root = set(int(x) for x in slots[0, 0] if x >= 0)
        node1 = set(int(x) for x in slots[1, 0] if x >= 0)
        node2 = set(int(x) for x in slots[2, 0] if x >= 0)
        node3 = set(int(x) for x in slots[3, 0] if x >= 0)
        # prefix slots 0,1,2 + root slot 3 are visible to every node
        assert {0, 1, 2, 3}.issubset(root)
        # node1 sees itself (slot 4), not sibling node2 (slot 5)
        assert 4 in node1 and 5 not in node1
        # node2 sees itself (slot 5), not sibling node1 (slot 4) or node3
        assert 5 in node2 and 4 not in node2 and 6 not in node2
        # node3 sees ancestor node1 (slot 4) + itself (slot 6), not sibling node4
        assert 4 in node3 and 6 in node3 and 7 not in node3

    def test_chain_is_linear(self):
        from tree.layout import build_chain_layout

        layout = build_chain_layout(torch.tensor([[5, 6, 7]]))
        slots = build_tree_ori_sparse_indices(
            layout,
            prefix_lens=torch.tensor([2]),
            block_table=torch.tensor([[0, 1]]),
            block_size=4,
            window=4,
            budget=3,
        )
        node2 = set(int(x) for x in slots[2, 0] if x >= 0)
        # node2 is depth 2; it sees node1 and itself, not node3
        assert 3 in node2 and 4 in node2 and 5 not in node2


class TestAttentionReference:
    def _setup(self):
        layout = _branch_layout()
        prefix = 2
        # kv rows: 0,1 = prefix; 2 = root; 3 = node1; 4 = node2; 5 = node3; 6 = node4
        torch.manual_seed(0)
        kv = torch.randn(prefix + 5, 1, 4)
        q = torch.randn(5, 1, 4)
        return layout, q, kv, prefix

    def test_sibling_perturbation_does_not_change_output(self):
        layout, q, kv, prefix = self._setup()
        out = tree_attention_reference(
            q, kv, layout, prefix_lens=torch.tensor([prefix]), scale=0.5, window=8
        )
        kv2 = kv.clone()
        kv2[prefix + 2] += 100.0  # node2 (sibling of node1)
        out2 = tree_attention_reference(
            q, kv2, layout, prefix_lens=torch.tensor([prefix]), scale=0.5, window=8
        )
        # row 1 is node1; sibling must not affect it
        assert torch.allclose(out[1], out2[1], atol=1e-5)
        # row 2 is node2; changing itself does affect it
        assert not torch.allclose(out[2], out2[2], atol=1e-5)

    def test_ancestor_perturbation_changes_output(self):
        layout, q, kv, prefix = self._setup()
        out = tree_attention_reference(
            q, kv, layout, prefix_lens=torch.tensor([prefix]), scale=0.5, window=8
        )
        kv2 = kv.clone()
        kv2[prefix] += 100.0  # root is an ancestor of every node
        out2 = tree_attention_reference(
            q, kv2, layout, prefix_lens=torch.tensor([prefix]), scale=0.5, window=8
        )
        assert not torch.allclose(out[1], out2[1], atol=1e-5)

    def test_adapter_raises_without_op_and_without_env(self, monkeypatch):
        monkeypatch.delenv("SGLANG_DSPARK_TREE_ALLOW_CPU_REFERENCE", raising=False)
        assert tree_attn_op_available() is False
        layout, q, kv, prefix = self._setup()
        with pytest.raises(NotImplementedError, match="npu_sparse_attn_sharedkv_tree"):
            tree_attention(
                q=q,
                kv=kv,
                layout=layout,
                prefix_lens=torch.tensor([prefix]),
                scale=0.5,
                window=8,
            )

    def test_adapter_uses_reference_only_when_enabled(self, monkeypatch):
        layout, q, kv, prefix = self._setup()
        monkeypatch.setenv("SGLANG_DSPARK_TREE_ALLOW_CPU_REFERENCE", "1")
        out = tree_attention(
            q=q,
            kv=kv,
            layout=layout,
            prefix_lens=torch.tensor([prefix]),
            scale=0.5,
            window=8,
        )
        ref = tree_attention_reference(
            q, kv, layout, prefix_lens=torch.tensor([prefix]), scale=0.5, window=8
        )
        assert torch.allclose(out, ref)

    def test_kwargs_contract(self):
        kwargs = build_tree_attn_kwargs(
            q=torch.zeros(5, 1, 4),
            ori_kv=torch.zeros(8, 1, 4),
            ori_sparse_indices=torch.zeros(5, 1, 128, dtype=torch.int32),
            cu_seqlens_q=torch.arange(6, dtype=torch.int32),
            seqused_kv=torch.full((5,), 4, dtype=torch.int32),
            tree_ancestors=torch.zeros(5, 3, dtype=torch.int64),
            tree_prefix=torch.zeros(5, dtype=torch.int32),
            cmp_ratio=1,
            softmax_scale=0.5,
        )
        for key in (
            "q",
            "ori_kv",
            "ori_sparse_indices",
            "cu_seqlens_q",
            "seqused_kv",
            "tree_ancestors",
            "tree_prefix",
            "cmp_ratio",
            "softmax_scale",
        ):
            assert key in kwargs


class TestCommitPlan:
    def test_root_and_path_move_bonus_does_not(self):
        plan = plan_tree_commit(
            path_node_ids=torch.tensor([[1, 3, -1, -1]]),
            verify_cache_loc=torch.tensor([[100, 101, 102, 103, 104]]),
            prefix_lens=torch.tensor([10]),
            block_table=torch.tensor([[0, 1, 2, 3, 4]]),
            block_size=4,
        )
        assert plan.source_rows.tolist() == [[100, 101, 103]]
        assert plan.dest_positions.tolist() == [[10, 11, 12]]
        assert plan.dest_slots.tolist() == [[10, 11, 12]]
        assert plan.commit_lens.tolist() == [3]
        assert plan.new_seq_lens.tolist() == [13]

    def test_root_only(self):
        plan = plan_tree_commit(
            path_node_ids=torch.tensor([[-1, -1]]),
            verify_cache_loc=torch.tensor([[100, 101, 102]]),
            prefix_lens=torch.tensor([7]),
            block_table=torch.tensor([[0, 1, 2]]),
            block_size=4,
        )
        assert plan.source_rows.tolist() == [[100]]
        assert plan.dest_positions.tolist() == [[7]]
        assert plan.commit_lens.tolist() == [1]

    def test_apply_without_pool_or_hook_raises(self):
        plan = TreeCommitPlan(
            source_rows=torch.tensor([[1]]),
            dest_positions=torch.tensor([[0]]),
            dest_slots=torch.tensor([[0]]),
            commit_lens=torch.tensor([1]),
            new_seq_lens=torch.tensor([1]),
            state_source=None,
            state_dest=None,
        )
        with pytest.raises(NotImplementedError, match="move_kv_cache"):
            apply_tree_commit(plan)

    def test_device_hook_is_called(self):
        plan = plan_tree_commit(
            path_node_ids=torch.tensor([[1, -1]]),
            verify_cache_loc=torch.tensor([[100, 101, 102]]),
            prefix_lens=torch.tensor([5]),
            block_table=torch.tensor([[0, 1]]),
            block_size=4,
        )
        calls = []
        apply_tree_commit(plan, device_hook=lambda p: calls.append(p))
        assert len(calls) == 1 and calls[0] is plan


class TestTreeConfig:
    def test_disabled_returns_none(self):
        assert resolve_tree_config(enable=False, topk=2, budget=8, depth=7) is None

    def test_requires_budget(self):
        with pytest.raises(ValueError):
            resolve_tree_config(enable=True, topk=2, budget=None, depth=7)

    def test_budget_must_be_at_least_depth(self):
        with pytest.raises(ValueError):
            resolve_tree_config(enable=True, topk=2, budget=6, depth=7)

    def test_topk1_requires_chain_budget(self):
        with pytest.raises(ValueError):
            resolve_tree_config(enable=True, topk=1, budget=8, depth=7)

    def test_verify_width(self):
        cfg = resolve_tree_config(enable=True, topk=2, budget=8, depth=7)
        assert isinstance(cfg, TreeConfig)
        assert cfg.verify_width == 9

    def test_resolve_from_server_args(self):
        from types import SimpleNamespace

        off = SimpleNamespace(
            speculative_dspark_tree_enable=False,
            speculative_eagle_topk=2,
            speculative_dspark_tree_budget=5,
        )
        assert resolve_tree_for_server(off, gamma=5) is None

        on = SimpleNamespace(
            speculative_dspark_tree_enable=True,
            speculative_eagle_topk=2,
            speculative_dspark_tree_budget=5,
        )
        cfg = resolve_tree_for_server(on, gamma=5)
        assert cfg.verify_width == 6

        default_topk = SimpleNamespace(
            speculative_dspark_tree_enable=True,
            speculative_eagle_topk=None,
            speculative_dspark_tree_budget=5,
        )
        assert resolve_tree_for_server(default_topk, gamma=5).topk == 2

        with pytest.raises(ValueError, match="static"):
            resolve_tree_for_server(on, gamma=5, ragged_mode="compact")
