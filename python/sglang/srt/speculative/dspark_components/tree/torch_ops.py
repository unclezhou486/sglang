"""Load the vllm-ascend tree operator bindings into the sglang process.

The ACLNN kernels live in the CANN vendor, while the ``torch.ops._C_ascend``
registrations live in small C++ extensions built by vllm-ascend's
``tools/build_tree_*_binding.py``. sglang does not import vllm, so it must load
those ``.so`` files explicitly.
"""

import os
from pathlib import Path

import torch

_LOADED: set[str] = set()


def load_tree_ops(paths: str | None = None) -> tuple[str, ...]:
    """Load every existing ``.so`` from ``SGLANG_DSPARK_TREE_OPS`` (idempotent)."""
    raw_paths = (
        os.environ.get("SGLANG_DSPARK_TREE_OPS", "") if paths is None else paths
    )
    loaded: list[str] = []
    for raw in raw_paths.split(os.pathsep):
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_file():
            continue
        key = str(path.resolve())
        if key in _LOADED:
            continue
        try:
            torch.ops.load_library(key)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load DSpark tree op binding {key!r}: {exc}"
            ) from exc
        _LOADED.add(key)
        loaded.append(key)
    return tuple(loaded)


def tree_ops_loaded() -> bool:
    namespace = getattr(torch.ops, "_C_ascend", None)
    return namespace is not None and hasattr(
        namespace, "npu_sparse_attn_sharedkv_tree"
    )


def ensure_tree_ops() -> bool:
    """Load the bindings once; return whether the tree attention op exists."""
    if tree_ops_loaded():
        return True
    load_tree_ops()
    return tree_ops_loaded()
