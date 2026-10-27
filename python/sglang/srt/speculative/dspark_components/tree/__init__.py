"""Pure-Python/torch core for DSpark tree speculative decoding.

This package intentionally imports only torch/dataclasses so it can be unit
tested on CPU without torch_npu, ACL, or the sglang runtime.
"""

from .shape import TreeShape
from .layout import TreeLayout, build_chain_layout, finalize_tree_layout_torch
from .scoring import CandidateLayer, DSparkMarkovScorer, IndependentScorer
from .builder import TreeBuilder
from .acceptance import TreeAcceptance, greedy_tree_accept
from .output import TreeAcceptOuts, build_tree_accept_outs
from .metadata import TreeMetadata, build_chain_metadata, build_tree_metadata
from .sparse import build_tree_ori_sparse_indices
from .attention import (
    tree_attention,
    tree_attention_reference,
    tree_attn_op_available,
)
from .commit import TreeCommitPlan, apply_tree_commit, plan_tree_commit
from .config import TreeConfig, resolve_tree_config
from .round import (
    TreePostPlan,
    TreePrePlan,
    TreeRoundPlan,
    plan_tree_post,
    plan_tree_pre,
    plan_tree_round,
)
from .node_pool import (
    NodePlan,
    NodeRecord,
    attend_nodes,
    build_node_descriptors,
    build_node_plan,
    commit_nodes,
    compress_nodes,
    prepare_nodes,
    record_compressed_nodes,
)
from .indexer import (
    tree_indexer,
    tree_indexer_ops_available,
    tree_indexer_select_all,
)
from .torch_ops import ensure_tree_ops, load_tree_ops, tree_ops_loaded

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
    "TreeRoundPlan",
    "plan_tree_round",
    "TreePrePlan",
    "TreePostPlan",
    "plan_tree_pre",
    "plan_tree_post",
    "NodePlan",
    "NodeRecord",
    "build_node_plan",
    "build_node_descriptors",
    "record_compressed_nodes",
    "prepare_nodes",
    "commit_nodes",
    "compress_nodes",
    "attend_nodes",
    "ensure_tree_ops",
    "load_tree_ops",
    "tree_ops_loaded",
    "tree_indexer",
    "tree_indexer_select_all",
    "tree_indexer_ops_available",
]

