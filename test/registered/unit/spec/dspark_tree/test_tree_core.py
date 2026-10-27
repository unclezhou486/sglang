"""CPU tests for the DSpark tree core: shape, layout, scoring, builder, accept, output.

These modules must stay torch/dataclasses-only so they run without torch_npu.
"""

import pytest
import torch

from tree.shape import TreeShape
from tree.layout import TreeLayout, build_chain_layout, finalize_tree_layout_torch
from tree.scoring import CandidateLayer, DSparkMarkovScorer, IndependentScorer
from tree.builder import TreeBuilder
from tree.acceptance import greedy_tree_accept
from tree.output import build_tree_accept_outs, gather_accepted_rows


def _layout_buf(bs: int, budget: int) -> TreeLayout:
    return TreeLayout(
        tokens=torch.full((bs, budget), -1, dtype=torch.long),
        depths=torch.zeros((bs, budget), dtype=torch.long),
        parents=torch.full((bs, budget), -1, dtype=torch.long),
        num_nodes=torch.zeros(bs, dtype=torch.int32),
        visibility=torch.zeros((bs, budget, budget), dtype=torch.bool),
        first_child=torch.full((bs, budget + 1), -1, dtype=torch.long),
        next_sibling=torch.full((bs, budget + 1), -1, dtype=torch.long),
    )


class TestTreeShape:
    def test_topk1_width_is_depth_plus_one(self):
        assert TreeShape(depth=7, budget=7, topk=1).target_query_len == 8

    def test_topk_gt1_width_is_budget_plus_one(self):
        assert TreeShape(depth=7, budget=10, topk=2).target_query_len == 11

    def test_rejects_budget_below_depth(self):
        with pytest.raises(ValueError):
            TreeShape(depth=7, budget=6, topk=2)

    def test_rejects_zero_topk(self):
        with pytest.raises(ValueError):
            TreeShape(depth=7, budget=7, topk=0)

    def test_validate_selected_nodes(self):
        shape = TreeShape(depth=7, budget=10, topk=2)
        assert shape.validate_selected_nodes(10) == 10
        with pytest.raises(ValueError):
            shape.validate_selected_nodes(11)


class TestTreeLayout:
    def test_links_and_visibility(self):
        tokens = torch.tensor([[11, 12, 13, 14]])
        depths = torch.tensor([[1, 1, 2, 2]])
        parents = torch.tensor([[0, 0, 1, 1]])
        out = finalize_tree_layout_torch(_layout_buf(1, 6), tokens, depths, parents, 4)
        # Children are linked in reverse insertion order; traversal still visits
        # every sibling exactly once.
        assert out.first_child[0, 0].item() == 2
        assert out.next_sibling[0, 2].item() == 1
        assert out.first_child[0, 1].item() == 4
        assert out.next_sibling[0, 4].item() == 3
        vis = out.visibility[0]
        assert bool(vis[0, 0]) and not bool(vis[0, 1])
        assert bool(vis[1, 1]) and not bool(vis[1, 0])
        assert bool(vis[2, 0]) and bool(vis[2, 2]) and not bool(vis[2, 1])
        assert bool(vis[3, 0]) and bool(vis[3, 3]) and not bool(vis[3, 1])

    def test_padding_rows_are_ignored(self):
        tokens = torch.tensor([[11, 12]])
        depths = torch.tensor([[1, 2]])
        parents = torch.tensor([[0, 1]])
        out = finalize_tree_layout_torch(_layout_buf(1, 4), tokens, depths, parents, 2)
        assert out.tokens[0, 2:].tolist() == [-1, -1]
        assert out.depths[0, 2:].tolist() == [0, 0]
        assert out.parents[0, 2:].tolist() == [-1, -1]
        assert not bool(out.visibility[0, 2:].any())

    def test_chain_layout_is_linear(self):
        out = build_chain_layout(torch.tensor([[11, 12, 13]]))
        assert out.parents.tolist() == [[0, 1, 2]]
        assert out.depths.tolist() == [[1, 2, 3]]
        assert out.visibility[0].tolist() == [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]


class _BiasHead:
    """Stateless fake Markov head: token t biases vocabulary index t % vocab."""

    def __init__(self, vocab: int, *, stateful: bool = False, shard=None):
        self.vocab = vocab
        self.stateful = stateful
        self.vocab_shard = shard

    def step_bias(self, prev_tokens, *, hidden_states=None, state=None):
        bias = torch.zeros(*prev_tokens.shape, self.vocab)
        flat_tokens = prev_tokens.reshape(-1)
        flat_bias = bias.reshape(-1, self.vocab)
        for i, tok in enumerate(flat_tokens.tolist()):
            flat_bias[i, tok % self.vocab] = 10.0
        return bias, (torch.zeros(1) if self.stateful else None)


class TestMarkovScorer:
    def test_bias_is_applied_per_parent(self):
        scorer = DSparkMarkovScorer(_BiasHead(4), topk=2)
        base = torch.zeros(1, 4)
        frontier = torch.tensor([[0, 1]])
        layer = scorer.score(base_logits=base, frontier_tokens=frontier, width=2)
        assert isinstance(layer, CandidateLayer)
        assert layer.token_ids[0, 0, 0].item() == 0
        assert layer.token_ids[0, 1, 0].item() == 1

    def test_rejects_vocab_shard(self):
        with pytest.raises(NotImplementedError, match="TP-sharded"):
            DSparkMarkovScorer(_BiasHead(4, shard=object()), topk=2)

    def test_rejects_stateful_head(self):
        scorer = DSparkMarkovScorer(_BiasHead(4, stateful=True), topk=2)
        with pytest.raises(NotImplementedError, match="stateless"):
            scorer.score(torch.zeros(1, 4), torch.tensor([[0]]), width=2)

    def test_independent_scorer_selects_from_logits(self):
        scorer = IndependentScorer(topk=2)
        logits = torch.tensor([[0.0, 5.0, 1.0, 0.0]])
        layer = scorer.score(logits, torch.tensor([[0]]), width=2)
        assert layer.token_ids[0, 0, 0].item() == 1


class TestTreeBuilder:
    def test_topk1_is_a_chain(self):
        builder = TreeBuilder(budget=3, topk=1, depth=3)
        logits = torch.zeros(1, 3, 8)
        logits[:, 0, 4] = 10.0
        logits[:, 1, 5] = 10.0
        logits[:, 2, 6] = 10.0
        out = builder.build(logits, root_token_ids=torch.tensor([1]))
        assert out.parents[0, :3].tolist() == [0, 1, 2]
        assert out.depths[0, :3].tolist() == [1, 2, 3]
        assert out.num_nodes[0].item() == 3

    def test_budget_caps_nodes_and_is_ancestor_closed(self):
        builder = TreeBuilder(budget=4, topk=2, depth=3)
        logits = torch.zeros(1, 3, 8)
        out = builder.build(logits, root_token_ids=torch.tensor([1]))
        n = int(out.num_nodes[0])
        assert n <= 4
        parents = out.parents[0, :n].tolist()
        for slot, parent in enumerate(parents):
            # parent id is either root (0) or an earlier node id
            assert parent == 0 or 1 <= parent <= slot

    def test_deterministic_on_ties(self):
        builder = TreeBuilder(budget=4, topk=2, depth=3)
        logits = torch.zeros(2, 3, 8)
        a = builder.build(logits, root_token_ids=torch.tensor([1, 1]))
        b = builder.build(logits, root_token_ids=torch.tensor([1, 1]))
        assert torch.equal(a.tokens, b.tokens)
        assert torch.equal(a.parents, b.parents)


class TestGreedyAccept:
    def _tree(self):
        layout = _layout_buf(1, 4)
        return finalize_tree_layout_torch(
            layout,
            tokens=torch.tensor([[5, 6, 7, 8]]),
            depths=torch.tensor([[1, 1, 2, 2]]),
            parent_ids=torch.tensor([[0, 0, 1, 1]]),
            num_nodes=4,
        )

    def test_full_path_with_bonus(self):
        # root target 5 -> node1; node1 target 7 -> node3; node3 target 9 -> bonus.
        target = torch.tensor([[5, 7, 0, 9, 0]])
        result = greedy_tree_accept(self._tree(), target)
        assert result.path_node_ids[0, :2].tolist() == [1, 3]
        assert int(result.num_accepted[0]) == 2
        assert int(result.bonus[0]) == 9

    def test_root_only(self):
        target = torch.tensor([[99, 0, 0, 0, 0]])
        result = greedy_tree_accept(self._tree(), target)
        assert int(result.num_accepted[0]) == 0
        assert result.path_node_ids[0].tolist() == [-1, -1, -1, -1]
        assert int(result.bonus[0]) == 99

    def test_sibling_is_not_followed(self):
        # root target 6 -> node2 (sibling of node1). node2 has no child.
        target = torch.tensor([[6, 0, 0, 0, 0]])
        result = greedy_tree_accept(self._tree(), target)
        assert result.path_node_ids[0, 0].item() == 2
        assert int(result.num_accepted[0]) == 1
        assert int(result.bonus[0]) == 0


class TestOutput:
    def test_path_gather_and_accept_index(self):
        verify_ids = torch.tensor([[100, 5, 6, 7, 8]])
        path = torch.tensor([[1, 3, -1, -1]])
        out = build_tree_accept_outs(
            verify_ids=verify_ids,
            path_node_ids=path,
            bonus=torch.tensor([9]),
            prefix_lens=torch.tensor([20]),
        )
        assert out.out_tokens[0, :3].tolist() == [5, 7, 9]
        assert out.out_tokens[0, 3:].tolist() == [0, 0]
        assert out.accept_index[0].tolist() == [1, 3, -1, -1, -1]
        assert out.commit_lens[0].item() == 3
        assert out.new_seq_lens[0].item() == 23

    def test_root_only_output(self):
        verify_ids = torch.tensor([[100, 5, 6]])
        path = torch.tensor([[-1, -1]])
        out = build_tree_accept_outs(
            verify_ids=verify_ids,
            path_node_ids=path,
            bonus=torch.tensor([7]),
            prefix_lens=torch.tensor([10]),
        )
        assert out.out_tokens[0, 0].item() == 7
        assert out.out_tokens[0, 1:].tolist() == [0, 0]
        assert out.commit_lens[0].item() == 1
        assert out.new_seq_lens[0].item() == 11

    def test_gather_accepted_rows(self):
        tensor = torch.tensor([[[10.0], [11.0], [12.0]]])
        accept_index = torch.tensor([[1, 2, -1]])
        gathered = gather_accepted_rows(tensor, accept_index)
        assert gathered[0, :2, 0].tolist() == [11.0, 12.0]
        # padding clamps to column 0; the commit-length gate drops it
        assert gathered[0, 2, 0].item() == 10.0
