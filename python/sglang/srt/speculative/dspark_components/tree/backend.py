"""Backend selection for the four DSpark tree operators.

``SGLANG_DSPARK_TREE_BACKEND`` is the single switch:

* ``vendor`` (default): all four operators use vllm-ascend ``_C_ascend``.
* ``sgl_kernel_npu``: all four operators use the Triton package.
* ``shadow``: indexer runs vendor + Triton and keeps vendor output; the other
  three operators stay on vendor because they are not yet performance-gated.
"""

import os

_BACKEND_ENV = "SGLANG_DSPARK_TREE_BACKEND"
_VALID = ("vendor", "sgl_kernel_npu", "shadow")
_ALIASES = {
    "native": "vendor",
    "vllm": "vendor",
    "triton": "sgl_kernel_npu",
    "sgl": "sgl_kernel_npu",
}


def tree_backend() -> str:
    raw = os.environ.get(_BACKEND_ENV, "vendor").strip().lower()
    backend = _ALIASES.get(raw, raw)
    if backend not in _VALID:
        raise ValueError(
            f"{_BACKEND_ENV} must be one of {_VALID}, got {raw!r}"
        )
    return backend
