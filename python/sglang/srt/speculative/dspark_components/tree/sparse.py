"""Per-query original-KV sparse slot construction for the tree attention op.

Row format: physical SWA slots of the committed prefix window, followed by the
visible ancestor draft slots (root and self included), then ``-1`` padding.
Siblings and descendants are never included.
"""

import torch


def _align_up(value: int, alignment: int = 128) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _ancestors(layout, req: int, node_id: int) -> list[int]:
    if node_id <= 0:
        return [0]
    row = []
    cur = node_id
    while cur > 0:
        row.append(int(cur))
        cur = int(layout.parents[req, cur - 1])
    row.append(0)
    row.reverse()
    return row


def build_tree_ori_sparse_indices(
    layout,
    *,
    prefix_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    window: int,
    budget: int | None = None,
    full_to_swa_mapping: torch.Tensor | None = None,
    draft_slots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [T, 1, W] int32 physical slots, W 128-aligned.

    ``block_table`` indexes the full KV pool. When ``full_to_swa_mapping`` is
    supplied, each full-pool slot is translated to its SWA slot; the attention
    op reads ``ori_kv`` from the SWA pool. ``draft_slots`` supplies this
    round's verify write slots for positions at/after the prefix; those rows
    are not visible in ``req_to_token`` yet.
    """
    batch, layout_budget = layout.tokens.shape
    budget = layout_budget if budget is None else int(budget)
    if block_size < 1 or window < 1:
        raise ValueError("block_size and window must be positive")
    if prefix_lens.shape[0] != batch:
        raise ValueError("prefix_lens batch mismatch")
    if block_table.shape[0] != batch:
        raise ValueError("block_table batch mismatch")
    query_width = budget + 1
    width = _align_up(window + budget, 128)
    device = prefix_lens.device
    slots = torch.full(
        (batch * query_width, 1, width), -1, dtype=torch.int32, device=device
    )

    for req in range(batch):
        prefix = int(prefix_lens[req])
        for node_id in range(query_width):
            depth = 0 if node_id == 0 else int(layout.depths[req, node_id - 1])
            positions: list[int] = []
            start = max(0, prefix + depth - window)
            positions.extend(range(start, prefix))
            positions.extend(prefix + ancestor for ancestor in _ancestors(layout, req, node_id))
            seen = set()
            slot_row = slots[req * query_width + node_id, 0]
            col = 0
            for pos in positions:
                if pos in seen:
                    continue
                seen.add(pos)
                if col >= width:
                    raise ValueError(
                        f"row sees more positions than slots ({width}) for req={req}, node={node_id}"
                    )
                if draft_slots is not None and pos >= prefix:
                    node = pos - prefix
                    if node >= draft_slots.shape[1]:
                        raise ValueError(
                            f"draft slot column {node} out of range {draft_slots.shape[1]}"
                        )
                    full_slot = int(draft_slots[req, node])
                else:
                    page = int(block_table[req, pos // block_size])
                    full_slot = page * block_size + (pos % block_size)
                if full_to_swa_mapping is not None:
                    full_slot = int(full_to_swa_mapping[full_slot])
                slot_row[col] = full_slot
                col += 1
    return slots
