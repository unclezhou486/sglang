"""Tree round orchestration split into pre-verify and post-verify plans.

The worker builds the pre-plan before the target forward (topology, verify
geometry, sparse slots) and the post-plan after it (acceptance, output,
commit). No device op is called here.
"""

from dataclasses import dataclass, field

import torch

from .acceptance import TreeAcceptance, greedy_tree_accept
from .builder import TreeBuilder
from .commit import TreeCommitPlan, plan_tree_commit
from .config import TreeConfig
from .layout import TreeLayout
from .metadata import TreeMetadata, build_tree_metadata
from .output import TreeAcceptOuts, build_tree_accept_outs
from .scoring import IndependentScorer
from .sparse import build_tree_ori_sparse_indices


@dataclass
class TreePrePlan:
    layout: TreeLayout
    metadata: TreeMetadata
    ori_sparse_indices: torch.Tensor
    node_records: list = field(default_factory=list)


@dataclass
class TreePostPlan:
    acceptance: TreeAcceptance
    outputs: TreeAcceptOuts
    commit: TreeCommitPlan


@dataclass
class TreeRoundPlan:
    layout: TreeLayout
    metadata: TreeMetadata
    ori_sparse_indices: torch.Tensor
    acceptance: TreeAcceptance
    outputs: TreeAcceptOuts
    commit: TreeCommitPlan


def plan_tree_pre(
    *,
    base_logits: torch.Tensor,
    root_token_ids: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    window: int,
    config: TreeConfig,
    scorer=None,
    full_to_swa_mapping: torch.Tensor | None = None,
) -> TreePrePlan:
    """Build topology, per-node verify geometry and sparse ori slots.

    Shapes:
      base_logits     [bs, depth, vocab]
      root_token_ids  [bs]
      prefix_lens     [bs]
    """
    if not config.enabled:
        raise ValueError("plan_tree_pre requires an enabled TreeConfig")
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
        full_to_swa_mapping=full_to_swa_mapping,
    )
    return TreePrePlan(layout=layout, metadata=metadata, ori_sparse_indices=sparse)


def plan_tree_post(
    *,
    pre: TreePrePlan,
    target_predict: torch.Tensor,
    verify_cache_loc: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> TreePostPlan:
    """Walk the tree against target argmax tokens and build output/commit plans."""
    batch = pre.layout.tokens.shape[0]
    if target_predict.shape != (batch, pre.metadata.query_width):
        raise ValueError(
            f"target_predict must be [{batch}, {pre.metadata.query_width}], "
            f"got {tuple(target_predict.shape)}"
        )
    acceptance = greedy_tree_accept(pre.layout, target_predict)
    outputs = build_tree_accept_outs(
        verify_ids=pre.metadata.verify_ids.reshape(batch, -1),
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
    return TreePostPlan(acceptance=acceptance, outputs=outputs, commit=commit)


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
    """Compose pre and post plans (CPU convenience for tests)."""
    pre = plan_tree_pre(
        base_logits=base_logits,
        root_token_ids=root_token_ids,
        prefix_lens=prefix_lens,
        block_table=block_table,
        block_size=block_size,
        window=window,
        config=config,
        scorer=scorer,
    )
    post = plan_tree_post(
        pre=pre,
        target_predict=target_predict,
        verify_cache_loc=verify_cache_loc,
        prefix_lens=prefix_lens,
        block_table=block_table,
        block_size=block_size,
    )
    return TreeRoundPlan(
        layout=pre.layout,
        metadata=pre.metadata,
        ori_sparse_indices=pre.ori_sparse_indices,
        acceptance=post.acceptance,
        outputs=post.outputs,
        commit=post.commit,
    )
