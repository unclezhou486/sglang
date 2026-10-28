"""Pure commit plan for tree acceptance; device application is a hook.

Only the root and accepted draft nodes are moved to contiguous history
positions. The bonus token is not moved: its KV is produced by the next step.
"""

from dataclasses import dataclass

import torch


@dataclass
class TreeCommitPlan:
    source_rows: torch.Tensor  # [r, max_rows] flat verify rows to move
    dest_positions: torch.Tensor  # [r, max_rows] logical history positions
    dest_slots: torch.Tensor  # [r, max_rows] physical cache slots
    commit_lens: torch.Tensor  # [r]
    new_seq_lens: torch.Tensor  # [r]
    state_source: torch.Tensor | None = None
    state_dest: torch.Tensor | None = None


def _slot_of(block_table: torch.Tensor, req: int, pos: int, block_size: int) -> int:
    page = int(block_table[req, pos // block_size])
    return page * block_size + (pos % block_size)


def plan_tree_commit(
    *,
    path_node_ids: torch.Tensor,
    verify_cache_loc: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> TreeCommitPlan:
    if path_node_ids.dim() != 2 or verify_cache_loc.dim() != 2:
        raise ValueError("path_node_ids and verify_cache_loc must be 2-D")
    batch = path_node_ids.shape[0]
    if verify_cache_loc.shape[0] != batch:
        raise ValueError("verify_cache_loc batch mismatch")
    if prefix_lens.shape[0] != batch or block_table.shape[0] != batch:
        raise ValueError("prefix_lens/block_table batch mismatch")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    device = path_node_ids.device
    valid = path_node_ids >= 0
    num_accepted = valid.sum(dim=1).to(torch.int64)
    max_rows = int(num_accepted.max().item()) + 1 if batch else 1

    source_rows = torch.full((batch, max_rows), -1, dtype=torch.long, device=device)
    dest_positions = torch.full((batch, max_rows), -1, dtype=torch.long, device=device)
    dest_slots = torch.full((batch, max_rows), -1, dtype=torch.long, device=device)
    for req in range(batch):
        prefix = int(prefix_lens[req])
        source_rows[req, 0] = verify_cache_loc[req, 0]
        dest_positions[req, 0] = prefix
        # The target verify already wrote the root/path rows into this
        # window.  Use those columns as destinations instead of re-reading
        # req_to_token, whose tree-depth columns may still be -1.
        dest_slots[req, 0] = verify_cache_loc[req, 0]
        for depth in range(int(num_accepted[req])):
            node = int(path_node_ids[req, depth])
            source_rows[req, depth + 1] = verify_cache_loc[req, node]
            dest_positions[req, depth + 1] = prefix + depth + 1
            dest_slots[req, depth + 1] = verify_cache_loc[req, depth + 1]

    commit_lens = num_accepted + 1
    new_seq_lens = prefix_lens.to(torch.int64) + commit_lens
    return TreeCommitPlan(
        source_rows=source_rows,
        dest_positions=dest_positions,
        dest_slots=dest_slots,
        commit_lens=commit_lens,
        new_seq_lens=new_seq_lens,
    )


def move_swa_kv(pool, plan: TreeCommitPlan) -> None:
    """Move accepted full/SWA KV rows to contiguous history slots.

    This covers the SWA/ori half for every layer (including compressed layers,
    whose recent window lives in the same SWA buffer). Compressed KV is
    committed separately by the node page kernels.
    """
    mask = plan.source_rows >= 0
    if not bool(mask.any()):
        return
    source = plan.source_rows[mask].to(torch.int64)
    dest = plan.dest_slots[mask].to(torch.int64)
    src_swa = pool.translate_loc_from_full_to_swa(source)
    dst_swa = pool.translate_loc_from_full_to_swa(dest)
    valid = (src_swa >= 0) & (dst_swa >= 0)
    if not bool(valid.any()):
        return
    src_swa = src_swa[valid]
    dst_swa = dst_swa[valid]
    for layer_id in range(int(pool.layer_num)):
        buffer = pool.get_swa_buffer(layer_id)
        flat = buffer.flatten(0, 1)
        flat[dst_swa] = flat[src_swa]


def apply_tree_commit(plan: TreeCommitPlan, *, pool=None, device_hook=None) -> None:
    """Legacy commit path: explicit device hook or pool-level move_kv_cache.

    The DSV4 worker now commits accepted nodes with ``move_swa_kv`` plus
    ``prepare_nodes`` / ``commit_nodes``; this function remains for pools that
    expose ``move_kv_cache`` and for explicit test hooks.
    """
    if device_hook is not None:
        device_hook(plan)
        return
    if pool is not None and hasattr(pool, "move_kv_cache"):
        mask = plan.source_rows >= 0
        pool.move_kv_cache(plan.dest_slots[mask], plan.source_rows[mask])
        return
    raise NotImplementedError(
        "DSV4 tree commit requires a pool-level move_kv_cache or an explicit "
        "device hook; this pool has neither. The DSV4 pools do not implement "
        "move_kv_cache yet."
    )
