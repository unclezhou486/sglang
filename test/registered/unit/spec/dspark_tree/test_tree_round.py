"""CPU tests for the one-call tree round orchestration."""

import torch

from tree.config import resolve_tree_config
from tree.round import plan_tree_post, plan_tree_pre, plan_tree_round


def test_pre_post_split_matches_round():
    cfg = resolve_tree_config(enable=True, topk=2, budget=4, depth=3)
    base = torch.zeros(1, 3, 8)
    kwargs = dict(
        base_logits=base,
        root_token_ids=torch.tensor([1]),
        prefix_lens=torch.tensor([3]),
        block_table=torch.tensor([[0, 1, 2, 3]]),
        block_size=4,
        window=4,
        config=cfg,
    )
    pre = plan_tree_pre(**kwargs)
    target = torch.zeros(1, 5, dtype=torch.long)
    vloc = torch.arange(5).unsqueeze(0)
    post = plan_tree_post(
        pre=pre,
        target_predict=target,
        verify_cache_loc=vloc,
        prefix_lens=torch.tensor([3]),
        block_table=torch.tensor([[0, 1, 2, 3]]),
        block_size=4,
    )
    combined = plan_tree_round(**kwargs, target_predict=target, verify_cache_loc=vloc)
    assert torch.equal(pre.layout.tokens, combined.layout.tokens)
    assert torch.equal(pre.layout.parents, combined.layout.parents)
    assert torch.equal(pre.ori_sparse_indices, combined.ori_sparse_indices)
    assert torch.equal(post.outputs.out_tokens, combined.outputs.out_tokens)
    assert torch.equal(post.commit.source_rows, combined.commit.source_rows)


def test_chain_round_matches_manual_pipeline():
    cfg = resolve_tree_config(enable=True, topk=1, budget=3, depth=3)
    base = torch.zeros(1, 3, 8)
    base[0, 0, 5] = 10.0
    base[0, 1, 6] = 10.0
    base[0, 2, 7] = 10.0
    plan = plan_tree_round(
        base_logits=base,
        root_token_ids=torch.tensor([1]),
        prefix_lens=torch.tensor([4]),
        target_predict=torch.tensor([[5, 6, 7, 99]]),
        block_table=torch.tensor([[0, 1]]),
        block_size=4,
        window=4,
        verify_cache_loc=torch.arange(4).unsqueeze(0) + 100,
        config=cfg,
    )
    assert plan.layout.parents[0, :3].tolist() == [0, 1, 2]
    assert plan.metadata.verify_ids.tolist() == [1, 5, 6, 7]
    assert plan.outputs.out_tokens[0, :4].tolist() == [5, 6, 7, 99]
    assert plan.commit.commit_lens[0].item() == 4


def test_tree_round_shapes_and_ancestor_only_slots():
    cfg = resolve_tree_config(enable=True, topk=2, budget=4, depth=3)
    base = torch.zeros(1, 3, 8)
    plan = plan_tree_round(
        base_logits=base,
        root_token_ids=torch.tensor([1]),
        prefix_lens=torch.tensor([3]),
        target_predict=torch.zeros(1, 5, dtype=torch.long),
        block_table=torch.tensor([[0, 1, 2, 3]]),
        block_size=4,
        window=4,
        verify_cache_loc=torch.arange(5).unsqueeze(0),
        config=cfg,
    )
    assert int(plan.layout.num_nodes[0]) <= 4
    assert plan.ori_sparse_indices.shape == (5, 1, 128)
    assert plan.outputs.out_tokens.shape == (1, 5)
    assert plan.outputs.accept_index.shape == (1, 5)
    assert plan.commit.commit_lens[0].item() == int(plan.acceptance.num_accepted[0]) + 1
    # root/self are visible; a sibling is not
    node1 = set(int(x) for x in plan.ori_sparse_indices[1, 0] if x >= 0)
    node2 = set(int(x) for x in plan.ori_sparse_indices[2, 0] if x >= 0)
    assert 4 in node1  # node1 self slot (prefix=3)
    assert 5 in node2  # node2 self slot
    assert 5 not in node1  # sibling excluded
