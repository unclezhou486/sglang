"""One-call tree round orchestration for the DSpark worker.

The worker supplies base logits, target argmax tokens, the verify window and
cache locations; this function returns every pure plan the device side needs.
No device op is called here.
"""

from dataclasses import dataclass

import torch

from tree.acceptance import TreeAcceptance, greedy_tree_accept
from tree.builder import TreeBuilder
from tree.commit import TreeCommitPlan, plan_tree_commit
from tree.config import TreeConfig
from tree.layout import TreeLayout
from tree.metadata import TreeMetadata, build_tree_metadata
from tree.output import TreeAcceptOuts, build_tree_accept_outs
from tree.scoring import IndependentScorer
from tree.sparse import build_tree_ori_sparse_indices


@dataclass
class TreeRoundPlan:
    layout: TreeLayout
    metadata: TreeMetadata
    ori_sparse_indices: torch.Tensor
    acceptance: TreeAcceptance
    outputs: TreeAcceptOuts
    commit: TreeCommitPlan


def plan_tree_round(
    *,
    base_logits: torch.Tensor,
    root_token_ids: torch.Tensor,
    prefix_lens: torch.Tensor,
    target_predict: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    window: int,
    verify_cache_loc: torch.Tensor,
    config: TreeConfig,
    scorer=None,
) -> TreeRoundPlan:
    """Build the tree, verify geometry, sparse slots, acceptance and commit plan.

    Shapes:
      base_logits       [bs, depth, vocab]
      root_token_ids    [bs]
      prefix_lens       [bs]
      target_predict    [bs, 1 + budget]  (argmax token per verify row)
      verify_cache_loc  [bs, 1 + budget]  (flat KV slots of the verify window)
    """
    if not config.enabled:
        raise ValueError("plan_tree_round requires an enabled TreeConfig")
    batch = base_logits.shape[0]
    scorer = scorer if scorer is not None else IndependentScorer(config.topk)
    builder = TreeBuilder(
        budget=config.budget, topk=config.topk, depth=config.depth, scorer=scorer
    )
    layout = builder.build(base_logits, root_token_ids)

    metadata = build_tree_metadata(
        layout, prefix_lens=prefix_lens, anchor_tokens=root_token_ids
    )
    if metadata.query_width != config.verify_width:
        raise ValueError(
            f"resolved verify width {config.verify_width} != "
            f"layout width {metadata.query_width}"
        )
    sparse = build_tree_ori_sparse_indices(
        layout,
        prefix_lens=prefix_lens,
        block_table=block_table,
        block_size=block_size,
        window=window,
        budget=config.budget,
    )
    acceptance = greedy_tree_accept(layout, target_predict)
    outputs = build_tree_accept_outs(
        verify_ids=metadata.verify_ids.reshape(batch, -1),
        path_node_ids=acceptance.path_node_ids,
        bonus=acceptance.bonus,
        prefix_lens=prefix_lens,
    )
    commit = plan_tree_commit(
        path_node_ids=acceptance.path_node_ids,
        verify_cache_loc=verify_cache_loc,
        prefix_lens=prefix_lens,
        block_table=block_table,
        block_size=block_size,
    )
    return TreeRoundPlan(
        layout=layout,
        metadata=metadata,
        ori_sparse_indices=sparse,
        acceptance=acceptance,
        outputs=outputs,
        commit=commit,
    )
