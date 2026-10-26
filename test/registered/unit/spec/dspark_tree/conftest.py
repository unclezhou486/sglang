import pathlib
import sys

# Pure tree modules live under dspark_components/tree and must be importable
# without executing sglang/__init__.py (no torch_npu on CPU CI).
_ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_ROOT / "python" / "sglang" / "srt" / "speculative" / "dspark_components"))
