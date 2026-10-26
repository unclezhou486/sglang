"""End-to-end CPU pipeline: builder -> metadata -> attention ref -> accept -> output -> commit."""

import torch

from tree.builder import TreeBuilder
from tree.scoring import DSparkMarkovScorer, IndependentScorer
from tree.metadata import build_tree_metadata
from tree.acceptance import greedy_tree_accept
from tree.output import build_tree_accept_outs
from tree.commit import plan_tree_commit
from tree.attention import tree_attention_reference
from tree.sparse import build_tree_ori_sparse_indices


def test_chain_identity_end_to_end():
    builder = TreeBuilder(budget=3, topk=1, depth=3)
    logits = torch.zeros(1, 3, 8)
    logits[0, 0, 5] = 10.0
    logits[0, 1, 6] = 10.0
    logits[0, 2, 7] = 10.0
    layout = builder.build(logits, root_token_ids=torch.tensor([1]))
    assert layout.parents[0, :3].tolist() == [0, 1, 2]

    meta = build_tree_metadata(
        layout, prefix_lens=torch.tensor([4]), anchor_tokens=torch.tensor([1])
    )
    assert meta.verify_ids.tolist() == [1, 5, 6, 7]
    assert meta.positions.tolist() == [4, 5, 6, 7]
    assert meta.node_seqused_kv.tolist() == [5, 6, 7, 8]

    # root target 5 -> node1; node1 target 6 -> node2; node2 target 7 ->
    # node3; node3 target 99 -> bonus.
    target = torch.tensor([[5, 6, 7, 99]])
    acc = greedy_tree_accept(layout, target)
    assert acc.path_node_ids[0, :3].tolist() == [1, 2, 3]
    assert int(acc.bonus[0]) == 99

    outs = build_tree_accept_outs(
        verify_ids=meta.verify_ids.reshape(1, -1),
        path_node_ids=acc.path_node_ids,
        bonus=acc.bonus,
        prefix_lens=torch.tensor([4]),
    )
    assert outs.out_tokens[0, :4].tolist() == [5, 6, 7, 99]
    assert outs.commit_lens[0].item() == 4
    assert outs.new_seq_lens[0].item() == 8

    plan = plan_tree_commit(
        path_node_ids=acc.path_node_ids,
        verify_cache_loc=(torch.arange(4).unsqueeze(0) + 100),
        prefix_lens=torch.tensor([4]),
        block_table=torch.tensor([[0, 1]]),
        block_size=4,
    )
    assert plan.source_rows[0, :4].tolist() == [100, 101, 102, 103]
    assert plan.dest_positions[0, :4].tolist() == [4, 5, 6, 7]
    assert plan.commit_lens[0].item() == 4


class _BiasToSeven:
    vocab_shard = None

    def step_bias(self, prev_tokens, *, hidden_states=None, state=None):
        bias = torch.zeros(*prev_tokens.shape, 8)
        bias[..., 7] = 100.0
        return bias, None


def test_markov_scorer_changes_root_candidate():
    logits = torch.zeros(1, 3, 8)
    independent = TreeBuilder(budget=4, topk=2, depth=3, scorer=IndependentScorer(2))
    markov = TreeBuilder(
        budget=4, topk=2, depth=3, scorer=DSparkMarkovScorer(_BiasToSeven(), 2)
    )
    ind = independent.build(logits, root_token_ids=torch.tensor([1]))
    mar = markov.build(logits, root_token_ids=torch.tensor([1]))
    assert 7 not in ind.tokens[0].tolist()
    assert 7 in mar.tokens[0].tolist()


def test_tree_pipeline_shapes():
    builder = TreeBuilder(budget=4, topk=2, depth=3)
    logits = torch.zeros(1, 3, 8)
    layout = builder.build(logits, root_token_ids=torch.tensor([1]))
    budget = layout.tokens.shape[1]
    assert int(layout.num_nodes[0]) <= 4

    meta = build_tree_metadata(
        layout, prefix_lens=torch.tensor([3]), anchor_tokens=torch.tensor([1])
    )
    assert meta.query_width == budget + 1
    assert meta.tree_ancestors.shape == (budget + 1, int(layout.depths.max()) + 1)

    slots = build_tree_ori_sparse_indices(
        layout,
        prefix_lens=torch.tensor([3]),
        block_table=torch.tensor([[0, 1, 2, 3]]),
        block_size=4,
        window=4,
        budget=budget,
    )
    assert slots.shape == (budget + 1, 1, 128)

    torch.manual_seed(0)
    q = torch.randn(budget + 1, 1, 4)
    kv = torch.randn(3 + budget + 1, 1, 4)
    out = tree_attention_reference(
        q, kv, layout, prefix_lens=torch.tensor([3]), scale=0.5, window=4
    )
    assert out.shape == (budget + 1, 1, 4)

    target = torch.zeros(1, budget + 1, dtype=torch.long)
    acc = greedy_tree_accept(layout, target)
    outs = build_tree_accept_outs(
        verify_ids=meta.verify_ids.reshape(1, -1),
        path_node_ids=acc.path_node_ids,
        bonus=acc.bonus,
        prefix_lens=torch.tensor([3]),
    )
    assert outs.out_tokens.shape == (1, budget + 1)
    assert outs.accept_index.shape == (1, budget + 1)
    plan = plan_tree_commit(
        path_node_ids=acc.path_node_ids,
        verify_cache_loc=torch.arange(budget + 1).unsqueeze(0),
        prefix_lens=torch.tensor([3]),
        block_table=torch.tensor([[0, 1, 2, 3]]),
        block_size=4,
    )
    assert plan.commit_lens[0].item() == int(acc.num_accepted[0]) + 1
