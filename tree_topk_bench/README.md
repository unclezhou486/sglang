# DSpark Tree topk=1 vs topk=2 benchmark

This folder contains a small, reproducible serving benchmark for comparing the
DSpark linear chain (topk=1) with the DSpark node tree (topk=2).

It measures the same request workload against an already-running SGLang server.
The server must be restarted between the two modes.

## Files

```text
tree_topk_bench/
├── datasets/
│   ├── tree_topk_semantic.jsonl       # 16 high-entropy prompts
│   └── tree_topk_math_control.jsonl   # 8 deterministic math prompts
├── run_tree_topk_ab.sh                # topk1 / topk2 / compare wrapper
├── compare_tree_topk.py               # JSONL result comparison
└── README.md
```

The semantic set is the primary candidate: code completion, refactoring,
debugging, API design, SQL, shell, regex, testing, architecture, open-ended
explanations, creative writing, and business writing.

The math set is a negative control. A tree should not help a deterministic
single-path task; if it does appear to help there, check the measurement first.

## Server setup

Keep every server argument identical except the speculative arguments.

### topk=1

```bash
--speculative-algorithm DSPARK \
--speculative-eagle-topk 1
```

Do not enable `--speculative-dspark-tree-enable` for this baseline.

### topk=2

```bash
--speculative-algorithm DSPARK \
--speculative-eagle-topk 2 \
--speculative-dspark-tree-enable \
--speculative-dspark-tree-budget 6 \
--disable-decode-cuda-graph
```

For the first tree validation, keep decode graph disabled. The current tree
accept path is greedy-only, so benchmark with temperature 0.

## Run

`MODEL` is the tokenizer/model path used by the server. `BASE_URL` defaults to
`http://127.0.0.1:30000`.

### topk=1 server

```bash
cd <sglang-repo>

MODEL=/path/to/model \
NUM_PROMPTS=16 \
MAX_CONCURRENCY=1 \
REPEATS=3 \
bash tree_topk_bench/run_tree_topk_ab.sh topk1
```

### topk=2 server

Restart the server with the topk=2 arguments above, then:

```bash
MODEL=/path/to/model \
NUM_PROMPTS=16 \
MAX_CONCURRENCY=1 \
REPEATS=3 \
bash tree_topk_bench/run_tree_topk_ab.sh topk2
```

### Compare

```bash
bash tree_topk_bench/run_tree_topk_ab.sh compare
```

Results are written to:

```text
perf/tree_topk/topk1_r*.jsonl
perf/tree_topk/topk2_r*.jsonl
```

## Environment variables

```text
BASE_URL              server URL, default http://127.0.0.1:30000
MODEL                 tokenizer/model path; required for topk1/topk2
DATASET               semantic (default), math, or a custom JSONL path
OUT_DIR               output directory, default <repo>/perf/tree_topk
OUTPUT_LEN            fixed output tokens, default 256
NUM_PROMPTS           total requests, default 16
MAX_CONCURRENCY       client concurrency cap, default 1
WARMUP_REQUESTS       warmup count, default 2
REPEATS               repetitions per mode, default 1
SEED                  request seed, default 0
APPLY_CHAT_TEMPLATE   default 1; set 0 for raw completion prompts
CONTEXT_LEN           prompt+output filter, default 8192
```

For low-batch latency, use `MAX_CONCURRENCY=1` (or 2/4). For a throughput
comparison, increase `MAX_CONCURRENCY` and provide enough prompts; this folder
only has 16 semantic prompts, so copy/repeat the lines for a larger batch.

## Math control

```bash
DATASET=math NUM_PROMPTS=8 MAX_CONCURRENCY=1 REPEATS=3 \
  bash tree_topk_bench/run_tree_topk_ab.sh topk1

# restart server in topk=2 mode

DATASET=math NUM_PROMPTS=8 MAX_CONCURRENCY=1 REPEATS=3 \
  bash tree_topk_bench/run_tree_topk_ab.sh topk2

bash tree_topk_bench/run_tree_topk_ab.sh compare
```

## Reading the result

`compare_tree_topk.py` prints medians over repeated runs and the topk2/topk1
ratios:

```text
accept_length
output_throughput
median_tpot_ms
median_itl_ms
```

Decision rule:

```text
accept_length > 1 and output_throughput > 1
    tree is winning

accept_length > 1 but output_throughput <= 1
    branch coverage improved, but tree overhead is too high

accept_length <= 1
    topk branches are not covering a better path
```

`accept_length` comes from the server's speculative metrics. If it is `n/a`,
verify that the server exposes `avg_spec_accept_length` and that the benchmark
is hitting the native SGLang endpoint.

## Notes

- The script does not start or stop the server.
- topk=1 and topk=2 must use the same model, batch policy, input prompts,
  output length, and sampling settings.
- `MAX_CONCURRENCY` is only a cap. Confirm the actual `bs` from
  `dspark_info_record`/`num_running_reqs`; concurrency can be higher than the
  active decode batch.
- The semantic set is intentionally high-entropy; it is not a correctness
  benchmark. Validate greedy output equality separately before using these
  throughput numbers.
