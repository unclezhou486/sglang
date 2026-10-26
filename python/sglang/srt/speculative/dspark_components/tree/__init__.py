"""Pure-Python/torch core for DSpark tree speculative decoding.

This package intentionally imports only torch/dataclasses so it can be unit
tested on CPU without torch_npu, ACL, or the sglang runtime.
"""

from tree.shape import TreeShape
from tree.layout import TreeLayout, build_chain_layout, finalize_tree_layout_torch
from tree.scoring import CandidateLayer, DSparkMarkovScorer, IndependentScorer
from tree.builder import TreeBuilder
from tree.acceptance import TreeAcceptance, greedy_tree_accept
from tree.output import TreeAcceptOuts, build_tree_accept_outs
from tree.metadata import TreeMetadata, build_chain_metadata, build_tree_metadata
from tree.sparse import build_tree_ori_sparse_indices
from tree.attention import (
    tree_attention,
    tree_attention_reference,
    tree_attn_op_available,
)
from tree.commit import TreeCommitPlan, apply_tree_commit, plan_tree_commit
from tree.config import TreeConfig, resolve_tree_config

__all__ = [
    "TreeShape",
    "TreeLayout",
    "build_chain_layout",
    "finalize_tree_layout_torch",
    "CandidateLayer",
    "DSparkMarkovScorer",
    "IndependentScorer",
    "TreeBuilder",
    "TreeAcceptance",
    "greedy_tree_accept",
    "TreeAcceptOuts",
    "build_tree_accept_outs",
    "TreeMetadata",
    "build_tree_metadata",
    "build_chain_metadata",
    "build_tree_ori_sparse_indices",
    "tree_attention",
    "tree_attention_reference",
    "tree_attn_op_available",
    "TreeCommitPlan",
    "plan_tree_commit",
    "apply_tree_commit",
    "TreeConfig",
    "resolve_tree_config",
]
