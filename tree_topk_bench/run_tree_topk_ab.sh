#!/usr/bin/env bash
# Run DSpark tree topk=1 / topk=2 with the same request workload.
#
# The script does NOT start the server. Start SGLang once per mode, then run:
#   ./run_tree_topk_ab.sh topk1
#   # restart server with topk=2 tree args
#   ./run_tree_topk_ab.sh topk2
#   ./run_tree_topk_ab.sh compare
#
# Server-side difference should be only:
#   topk1: --speculative-eagle-topk 1
#   topk2: --speculative-eagle-topk 2 \
#          --speculative-dspark-tree-enable \
#          --speculative-dspark-tree-budget 6 \
#          --disable-decode-cuda-graph
#
# Keep every other server arg identical.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"
MODE="${1:-}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL="${MODEL:-}"
DATASET="${DATASET:-semantic}"
OUT_DIR="${OUT_DIR:-$ROOT/perf/tree_topk}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
REPEATS="${REPEATS:-1}"
SEED="${SEED:-0}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"

case "$DATASET" in
  semantic) DATASET_PATH="$SCRIPT_DIR/datasets/tree_topk_semantic.jsonl" ;;
  math)     DATASET_PATH="$SCRIPT_DIR/datasets/tree_topk_math_control.jsonl" ;;
  *)        DATASET_PATH="$DATASET" ;;
esac

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "dataset not found: $DATASET_PATH" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"

run_one() {
  local label="$1"
  local i out
  if [[ -z "$MODEL" ]]; then
    echo "MODEL is required for topk1/topk2 runs" >&2
    exit 2
  fi
  for i in $(seq 1 "$REPEATS"); do
    out="$OUT_DIR/${label}_r${i}.jsonl"
    rm -f "$out"
    local args=(
      -m sglang.benchmark.serving
      --backend sglang
      --base-url "$BASE_URL"
      --model "$MODEL"
      --dataset-name custom
      --dataset-path "$DATASET_PATH"
      --sharegpt-output-len "$OUTPUT_LEN"
      --sharegpt-context-len "$CONTEXT_LEN"
      --num-prompts "$NUM_PROMPTS"
      --max-concurrency "$MAX_CONCURRENCY"
      --request-rate inf
      --warmup-requests "$WARMUP_REQUESTS"
      --seed "$SEED"
      --tag "$label"
      --output-file "$out"
      --output-details
      --flush-cache
    )
    if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
      args+=(--apply-chat-template)
    fi
    echo ">>> $label run $i/$REPEATS: bs<=$MAX_CONCURRENCY, prompts=$NUM_PROMPTS, output=$OUTPUT_LEN"
    "$PYTHON" "${args[@]}"
  done
}

case "$MODE" in
  topk1|topk2)
    run_one "$MODE"
    echo "result: $OUT_DIR/${MODE}_r*.jsonl"
    ;;
  compare)
    "$PYTHON" "$SCRIPT_DIR/compare_tree_topk.py" --out-dir "$OUT_DIR"
    ;;
  *)
    echo "usage: $0 {topk1|topk2|compare}" >&2
    echo "  topk1/topk2: run against an already-running server in that mode" >&2
    echo "  compare: summarize topk1_r*.jsonl and topk2_r*.jsonl" >&2
    exit 2
    ;;
esac
