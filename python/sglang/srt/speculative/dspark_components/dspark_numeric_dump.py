"""Per-rank numeric fingerprints for the DSpark draft forward.

Debug/reference path only. Enable with ``SGLANG_DSPARK_NUMERIC_DUMP=1`` and read
the ``DSPARK_DUMP`` log lines; they show the fingerprint of every rank in a
process group plus the cross-rank spread, so a failing ``attn_tp_size > 1`` run
can be diffed against a working ``attn_tp_size == 1`` run.

Every dump point must be reached by *all* ranks of the group (the group gather is
a collective). The helpers here swallow their own exceptions so debug code can
never break the forward.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# 6 floats per rank: mean, std, min, max, absmax, sum.
_FP_WIDTH = 6
_count = 0


def enabled() -> bool:
    return envs.SGLANG_DSPARK_NUMERIC_DUMP.get()


def _tags() -> Optional[frozenset]:
    raw = envs.SGLANG_DSPARK_NUMERIC_DUMP_TAGS.get()
    tags = frozenset(t for t in raw.replace(" ", "").split(",") if t)
    return tags or None


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _fingerprint(t: torch.Tensor) -> torch.Tensor:
    x = t.detach().float().reshape(-1)
    device = t.device
    if x.numel() == 0:
        return torch.zeros(_FP_WIDTH, dtype=torch.float64, device=device)
    std = x.std() if x.numel() > 1 else torch.zeros((), device=device)
    return torch.stack(
        [x.mean(), std, x.min(), x.max(), x.abs().max(), x.sum()]
    ).to(torch.float64)


def dump(tag: str, t: Optional[torch.Tensor], group=None) -> None:
    """Log a fingerprint of ``t``; gather every rank's when ``group`` is given."""
    global _count
    if t is None or not enabled():
        return
    allowed = _tags()
    if allowed is not None and tag not in allowed:
        return
    if _capturing():
        return
    limit = int(envs.SGLANG_DSPARK_NUMERIC_DUMP_MAX.get())
    if limit and _count >= limit:
        return
    _count += 1
    try:
        fp = _fingerprint(t)
        shape = tuple(t.shape)
        if group is not None and group.world_size > 1:
            gathered = group.all_gather(fp).reshape(group.world_size, _FP_WIDTH)
            means = gathered[:, 0]
            spread = float(means.max() - means.min())
            rows = " ".join(
                f"r{i}(mean={v[0]:.5e},std={v[1]:.3e},absmax={v[4]:.3e})"
                for i, v in enumerate(gathered)
            )
            if group.rank_in_group == 0:
                logger.warning(
                    "DSPARK_DUMP %s shape=%s dtype=%s mean_spread=%.3e | %s",
                    tag,
                    shape,
                    t.dtype,
                    spread,
                    rows,
                )
        else:
            logger.warning(
                "DSPARK_DUMP %s shape=%s dtype=%s mean=%.5e std=%.3e "
                "min=%.3e max=%.3e absmax=%.3e sum=%.5e",
                tag,
                shape,
                t.dtype,
                fp[0].item(),
                fp[1].item(),
                fp[2].item(),
                fp[3].item(),
                fp[4].item(),
                fp[5].item(),
            )
    except Exception as e:  # pragma: no cover - debug path must never raise
        logger.warning("DSPARK_DUMP %s failed: %s", tag, e)


def attn_tp_group():
    """The attention-TP group, or None when it is a single rank."""
    try:
        from sglang.srt.runtime_context import get_parallel

        group = get_parallel().attn_tp_group
        return group if group is not None and group.world_size > 1 else None
    except Exception:
        return None
