# DSV4 + DSpark tree speculative decoding (topk > 1) on Ascend NPU — architecture

Status: **v5.2 implemented (CPU 66/66 passed; A3 gates pending). Branch `feat/npu-dsv4-dspark-tree-v5`, pushed to `unclezhou`.**

| item | decision |
|---|---|
| Architecture | **C** — backend-neutral torch tree core + NPU four-op adapters; commit uses the proven node-page path (post-accept re-compress deferred until A1) |
| Base | `dspark_topk/sglang` local clone, `main = 5d4a6f499e` (2026-10-30) |
| Reference (port source) | `dspark_topk/sglang-unclezhou/sglang`, `origin/feat/npu-dsv4-dspark-topk = b7c7fa0a` — **hardware-verified by the user at TP16 + DP16** (`[hw]`, user report; not re-verified here) |
| Cross-check (design evidence only) | `dspark_topk/vllm-ascend`, `tree261025-rebase`, commits `8947f1f9f`, `b7cd632f2`, `612b84ad6`, `d6048af3f`, `2fe76a173` — **not ported**, no vendor ops taken |
| Kernels | `dspark_topk/sgl-kernel-npu`, `feat/npu-dsv4-tree-triton = 905c136`, four ops in `python/sgl_kernel_npu/sgl_kernel_npu/tree/` |
| Target branch | `feat/npu-dsv4-dspark-tree-v5`, from local `main = 5d4a6f499e`, pushed to `unclezhou` |
| Scope | full production tree (arbitrary topk / depth / budget), not a fixed-config MVP |
| New files | exactly two: `dspark_tree.py`, `dspark_tree_ops.py` under existing `dspark_components/` |
| Not allowed | `tree/` folder, SGLang copy, second DSpark worker, vendor adapter, shadow mode, `TreeOpBackend`, `SGLANG_DSPARK_TREE_BACKEND`, kernel edits |

## 0.4 Implementation status (v5.1, static)

| item | value |
|---|---|
| Branch | `feat/npu-dsv4-dspark-tree-v5`, base `main = 5d4a6f499e` |
| Commits | `9abd9d74d0` implementation; `a8395c90b9` extended CPU tests; `d98cf6d2c5` harness fix; `6a7d2a2b4d` dead-code/duplicate trim |
| Remote | `unclezhou/feat/npu-dsv4-dspark-tree-v5` (pushed) |
| Diff | 17 files, **+3,832/−84** total; trim commit net −456 |
| New modules | `dspark_tree.py` 1,134 lines; `dspark_tree_ops.py` 835 lines |
| Tests | **66 CPU tests** (45 core + 20 op + 1 rope) vs 78 in the reference; the removed 12 cases covered reference-only helpers |
| Commit path | proven **node-page freeze/commit + SWA move**, not post-accept re-compress |
| Parity | all non-scaffolding reference tree names ported |
| Dropped | vendor `_C_ascend`, shadow, `SGLANG_DSPARK_TREE_BACKEND`, `tree/` package, dynamic `ForwardBatch` attrs |
| Improvements | node KV/scale carried on `tree_plan`; `sliding_window`→`window_size` fallback; grammar/watermark fail loud; one canonical commit plan |
| Actual size | production ≈2,790, tests ≈1,042 (still above the lean 1,950-2,150 estimate because the proven node-page commit and full integration are kept) |
| Verified | static + CPU: **66/66 tree CPU tests pass** (torch 2.9.1+cpu, Python 3.14 venv); `py_compile`/`compileall`; AST import/signature checks; forbidden-token grep |
| NOT verified | A3 eager/graph, TP16/DP16, accept-length ratio |

Why the size grew: to guarantee functional equivalence with the proven branch, the full tree core was ported rather than trimmed. The post-accept re-compress remains an optimization only after A1 (commit parity) is proven on hardware; the current commit path is the reference's proven node-page path.

Verification commands (run on a machine with torch / the A3 box):

```bash
python3 -m pytest -q \
  test/registered/unit/spec/test_dspark_tree_core.py \
  test/registered/unit/spec/test_dspark_tree_ops.py \
  test/registered/unit/spec/test_dspark_tree_rope.py
```

A3 gate order: (a) tree disabled/topk=1 baseline; (b) tree enabled, topk=2, budget>=depth, static ragged; (c) greedy short-context token equality; (d) accept-length vs verify-cost ratio; (e) repeat (a)-(c) at TP16/DP16; graph capture only if (d) is negative.

---

## 0.5 Decision (v5): architecture C

**C = backend-neutral torch tree core + four `sgl_kernel_npu` tree-op adapters + the reference's proven node-page commit (post-accept re-compress deferred until A1).**

Why not A (port the reference tree layer): it is ~3,619 production lines before tests, keeps a 19-file
`tree/` package, vendor/shadow/backend-switch scaffolding, and a bespoke node-page freeze/commit path — all
things the constraints forbid or the user rejected as too large.

Why not B (reuse NPU-native EAGLE `build_tree_kernel_efficient` + `verify_tree_greedy`): **falsified by
kernel semantics, not taste**:

- `build_tree_kernel.cpp:151,181` derives the parent as `selected_index // topk`; it is a **uniform
  topk-ary, parent-major** candidate space.
- `build_tree_kernel.cpp:163-165` silently skips a node whose parent was not selected: **no ancestor
  closure**; orphans consume budget but are unreachable.
- The reference DSpark builder is **ancestor-closed + global top-`width` per level + renumbered**
  (`tree/builder.py:47-49,99-129`). These are different trees.
- `verify_tree_greedy.py:51` compares `candidates[i]` with `target_predict[i-1]` and **never uses**
  `retrive_next_token`/`retrive_next_sibling`: the Triton accept kernel is chain-only. The native variant uses
  a different position-indexed convention.
- Reusing EAGLE therefore needs an adapter as large as the reference builder, with a different tree and a
  higher risk of silent orphan/accept errors.

Why not "skip current-block compressed KV during verify": **not safe**. The tree attention resolves
compressed slots `k >= prefix // ratio` through `tree_ancestors` → node pool
(`tree_ops_semantics.md`, `tree_reference.py:539-557`). When `prefix % ratio != 0`, the boundary partial block
must aggregate the pre-prefix token (paged state) with draft nodes (node pool); dropping it silently loses a
key. All four tree ops stay.

**Commit strategy (implemented):** the reference's proven path — accept → `move_swa_kv` for the accepted full/SWA rows → record compressed node rows during verify → `prepare_nodes`/`commit_nodes` (node-page freeze/commit). Post-accept chain re-compress is **not** used in this implementation; it remains a follow-up optimization only after A1 (commit parity) is proven on hardware.

**Two falsifications were planned as P1 gates but could not run on the analysis workstation** (no torch/NPU). The implementation therefore keeps the reference's proven commit path; A1/A2 must still be executed on hardware:

| # | assumption | falsifier |
|---|---|---|
| A1 | post-accept chain `forward_compress` reproduces the reference's node-page commit on the accepted path | two-step CPU/numeric harness: chain re-compress vs `node_commit.freeze/commit`, diff compressed KV pages and `state_cache` |
| A2 | the four `sgl_kernel_npu.tree` ops pass the independent `tree_reference.py` oracle within tolerance | run Repo C `test_tree_*` + oracle on A3 |

A2 must pass on A3 before hardware sign-off. A1 gates only the optional re-compress follow-up; the implemented path already uses the reference's proven node-page commit, so an A1 failure no longer changes the architecture.

Plan estimate **[static]**: production ~1,950-2,150; tests ~900-1,100 for the 16-invariant suite, or ~1,790 if all 78 reference CPU tests are ported. The actual implemented size is in §0.4 (production ≈2,790, tests ≈1,042). The remaining gap is the proven node-page commit and the mandatory backend integration.

---

## 0. Environment reality (must read before planning gates)

This workstation is analysis/edit only **[static]**:

- `python3` 3.14.8, `pip list` empty — no `torch`, no `torch_npu`, no `sgl_kernel_npu`.
- No `/dev/davinci*`.
- Remote `unclezhou` reachable; `feat/npu-dsv4-dspark-topk` is at `b7c7fa0a` (matches the handoff).

Consequence: nothing here can produce `[hw]` evidence, and even `[cpu]` unit tests need a CPU `torch` install.
Every claim in this document is `[static]` unless marked otherwise.

---

## 1. What already exists in `main` and is reused unchanged

**[static]** `main` contains the complete DSpark **chain (topk == 1)** path and no `tree/` directory.

Reused as-is, no edits:

| piece | evidence (main) |
|---|---|
| `SpeculativeAlgorithm.DSPARK` → `DSparkWorkerV2` | `python/sglang/srt/speculative/spec_info.py:335-340` |
| Worker orchestration `forward_batch_generation` / `_forward_decode` | `dspark_components/dspark_worker_v2.py:758, 972` |
| Draft proposal (`DraftBlockProposer`, markov block head, folded/non-folded) | `dspark_components/dspark_draft.py:448, 258-303, 487-524` |
| Target verify executor (eager + compact/static + graph epilogue) | `dspark_components/dspark_verify.py:302, 468, 179, 380` |
| Verify window allocation / budget / schedule planner | `dspark_components/dspark_planner.py:800, 376, 326` |
| `DFlashDraftInputV2` / `DFlashVerifyInput` / `DFlashVerifyInput.prepare_for_verify` | `dflash_info_v2.py:36-58`, `dflash_info.py` |
| Chain accept / output / commit | `dspark_verify.py:1103, 1296`, `dspark_kv_inject.py:142, 359`, `dspark_deferred_commit.py:134` |
| DSV4 attention backend, all chain paths | `hardware_backend/npu/attention/ascend_dsv4_backend.py` |
| Grammar tree (already tree-shaped, `from_device(next_token, next_sibling, ...)`) | `spec_utils.py:818-857` |
| Ragged verify modes, KV pool / cache loc, DP/TP plumbing | `ragged_verify.py`, memory pool, planner |

`dspark_planner.py` is **not modified at all**: `VerifyWindow` already carries `positions_2d [bs, W]`,
`verify_cache_loc [bs*W]`, `verify_cache_loc_2d [bs, W]` (`dspark_planner.py:59-66`).
A tree simply writes its own per-node positions into that window and consumes the same slots.
The only requirement is `verify_num_draft_tokens = 1 + budget` (window width), so the uniform allocation
stays valid. This is the main "reuse the framework" win.

`dspark_draft.py` is **not modified either**: the DSpark draft stays a chain block that produces
`base_logits [bs, depth, vocab]`; tree branching is decided afterwards by the markov-head scorer.
There is **no second draft forward** in tree mode (see §4).

---

## 2. Why a mask / sparse-index patch alone is not enough

**[static]** For DSV4 the decode state is hybrid: SWA (original KV) + compressed KV pools (ratio 4 / 128)
+ a lightning indexer. A tree verify step therefore needs, per node:

1. topology (`parent`, `ancestors`, `depth`) — neither exists in the chain path;
2. a per-node target hidden row so the node's KV can be committed to the draft cache
   (`TargetHiddenKvInjector`/node commit);
3. per-node absolute positions and `seqused_kv = prefix + depth + 1` (chain uses `prefix + arange`);
4. per-node compressed-KV projection (`compressor_tree_project` + `compressor_tree`), because
   compression boundaries fall on tree paths, not on a contiguous block;
5. per-node indexer top-k (`quant_lightning_indexer_tree`) over ancestor-aware history;
6. per-node attention mask over ancestors (`sparse_attn_sharedkv_tree`);
7. an accepted **node path** instead of a scalar `correct_len` prefix.

Plus removing three chain-only gates in `main`:
`speculative_hook.py:760-792` (forces `num_steps=1`, `speculative_eagle_topk=1`) and
`dspark_worker_v2.py:1496` (`assert speculative_eagle_topk in (None, 1)`).

---

## 3. Target per-step dataflow (tree vs chain)

Tree differs from chain in **what the target verify sees**, not in how many forwards run.

| step | chain (today) | tree (target design) |
|---|---|---|
| draft proposal | 1 draft forward over `bs*(gamma or gamma+1)` rows | same, unchanged |
| plan | none | `plan_tree_pre(base_logits, ...)` → topology + metadata + sparse slots + node plan |
| verify ids | `[anchor, d1..dγ]`, `bs*(gamma+1)` rows | `[root, n1..nN]`, `bs*(1+budget)` rows |
| target forward | 1 forward | 1 forward, same, wider row set |
| compressor | contiguous block boundaries | per-node project/compress (tree ops) |
| indexer | per-query chain top-k | per-node ancestor-aware top-k (tree op) |
| attention | causal within block | ancestor mask (tree op) |
| accept | scalar `correct_len` prefix | `greedy_tree_accept` → node path + bonus |
| commit | prefix `col < commit_lens` | accepted node rows → `dest_slots` / `dest_positions`, then post-accept compressed-KV/state re-compress |
| forwards/step | 2 | **2** |

Tree-specific calls in `_forward_decode` (from the reference implementation, to be re-derived against `main`):

| stage | reference anchor |
|---|---|
| tree config resolve (once, init) | `dspark_worker_v2.py:291` |
| verify window alloc (already in `main`) | `dspark_worker_v2.py:1068` |
| `plan_tree_pre` + scorer | `:1156-1183` |
| `run_non_compact(..., tree_plan=...)` | `:1239`; `dspark_verify.py:305-321` |
| target forward | `dspark_verify.py:368` |
| `plan_tree_post` (accept/output/commit plan) | `:1258` |
| `move_swa_kv` | `:1315` |
| `prepare_nodes` / `commit_nodes` | `:1318-1326` |
| `commit_hidden(accept_index, tree_positions, tree_commit)` | `:1411-1418`; `dspark_verify.py:408-465` |

---

## 4. Tree node KV and the "no extra draft forward" property

**[static]** In `main`, DSpark drafts from `base_logits [bs, gamma, vocab]` produced by
`compute_base_logits(fwd.raw_hidden)` (`dspark_draft.py:487-490`) and the markov head loop
(`:285-299`). The tree re-uses exactly these per-depth logits:

- candidate selection per depth uses `DSparkMarkovScorer` (`head.step_bias(frontier_tokens, ...)` + top-k);
- the builder keeps an ancestor-closed set, level by level;
- per-node **target** hidden comes from target verify (`CaptureHiddenMode.FULL`,
  `dspark_verify.py:319`), which is what the accept/commit path writes into the draft KV.

So tree mode needs **no tree draft attention and no second draft forward**. This is why the delta can stay
in the verify/commit path instead of restructuring the draft worker.

Node KV pipeline (three of the four kernels, direct calls, no adapter):

```
compressor_tree_project(x, wkv, wgate, nodes, ape, positions, cu)   # in-place, nodes preallocated
        -> nodes fp32 [T+1, 1, 2*c*d]  (row 0 reserved)
compressor_tree(state, nodes, norm, cos, sin, table, prefix,
                ancestors, depths, requests, ratio, rope_dim, epsilon)
        -> node_kv bf16 [T, d]      # zero row when (prefix+depth+1) % ratio != 0
quantize(node_kv) -> int8 [T,128] + fp16 scale    # framework-side cast, not an op
quant_lightning_indexer_tree(query, history, weights, query_scale, history_scale,
                             nodes_i8, node_scale, ancestors, prefixes, requests,
                             table, cu, lengths, residual, metadata, topk)
        -> cmp_sparse_indices int32 [T, 1, topk]
sparse_attn_sharedkv_tree(q, ori_kv=..., cmp_kv=..., cmp_tree_kv=node_kv,
                          tree_ancestors=..., tree_prefix=..., seqused_kv=...,
                          cmp_sparse_indices=..., ori_sparse_indices=..., ...)
        -> out, softmax_lse
```

Kernel contract facts that shaped the design **[static]** (audited from `sgl-kernel-npu`):

- `compressor_tree_project` is the only in-place op; `nodes` must be preallocated.
- All other three allocate their outputs; the caller preallocates only inputs/buffers.
- `metadata` (int32[1024]) is **validated then ignored** for both indexer and attention → pass one static
  zero buffer, no vendor metadata op, no `_C_ascend` `.so`.
- `cu` is validated to equal `arange(n+1)` in project/indexer → it only asserts the one-query-per-node rule.
- `seqused_kv = prefix + depth + 1` is required by attention; `depths` is only consumed by the compressor.
- `ori_sparse_indices` width must be a positive multiple of 128; `topk ∈ [1,2048]`; `ratio ∈ {4,128}`.
- **Kernel-side** capture blockers (inside `sgl_kernel_npu`, not ours to fix): indexer does
  `lengths.max().item()` and sizes scratch from it, and has a data-dependent "select-all" branch; project
  does two device-sync validations. Consequence: the captured **verify forward** stays a P6 gate and v1 runs
  eager. This does **not** excuse the framework side: plan/accept/commit must be written capture-shaped
  from P2 (§4.5), otherwise P6 becomes a rewrite.

---

## 4.5 Capture-shaped by construction (R1) — a P2 constraint, not a P6 task

The vllm-ascend branch (DSV4 `topk>1`) **requires** `cudagraph_mode == FULL`
(`worker/v2/spec_decode/tree/speculator.py:120-123`) and had to rewrite tree construction and node commit
before graph capture was viable (`612b84ad6`, `2fe76a173`). The lesson for us: on a low-latency workload the
eager target verify will likely eat the accept-length win, so the tree path must be graph-shaped by design.

Rules binding S2/S5/S6 and P1/P5 from P2 onward:

1. **Fixed capacity.** `1 + budget` rows per request are preallocated in `__init__`; nothing is sized from
   data. `budget`, `depth`, `topk` are constants for the life of the process.
2. **Static-k selection.** Use `torch.topk(k=constant)` — never `argsort` over a variable-length frontier, and
   never host-derived indices. Cost: one level at a time, `topk ≤ budget`.
3. **No host readback in the per-step path.** No `.item()`, `.tolist()`, `.cpu()`, `data_ptr`-based
   value branching on plan tensors. Host only supplies per-step prefixes through a pinned staging buffer
   (H2D), exactly like the vllm `PrefixStaging` pattern.
4. **No allocation inside the captured region.** Plan/index/visibility/link buffers are persistent; outputs
   are written with `copy_` / `index_copy_` / `scatter_` / `fill_` into preallocated tensors. `612b84ad6`
   failed exactly because of a `.contiguous().to(torch.long)` temporary at launch time.
5. **Uniform row count.** The tree always emits `1 + budget` rows; padded rows are explicitly invalidated
   (§8). No per-request truncation, no compact ragged layout.
6. **Mixed prefill+decode batches** cannot replay a fixed tree gear → tree verify must either run eager for
   that batch or be rejected (fail loudly). vllm does the former (`aclgraph_utils.py:245-255`).

What this buys: the plan/accept/commit layer is graph-capturable without new kernels; only the four
`sgl_kernel_npu` ops (indexer `.item()`/select-all, project validation syncs) remain a P6 blocker.

---

## 5. Module layout and exact delta

### 5.1 New file 1 — `dspark_components/dspark_tree.py` (pure host + generic torch)

No `torch_npu`, no `sgl_kernel_npu`, no `.so`. Sections:

| section | contents |
|---|---|
| S1 config | `TreeConfig`, `TreeShape`, width resolution (`gamma` = depth, `verify_num_draft_tokens` = 1+budget), validation |
| S2 scoring/builder | `DSparkMarkovScorer`, ancestor-closed level-wise `TreeBuilder`, capacity/padding rules |
| S3 metadata | `prefix`, `depth`, `ancestors` (root-first, `-1` padded), `seqused_kv`, per-node `positions`, `cu_seqlens`, `positions % ratio` residues |
| S4 sparse slots | SWA window slots + ancestor node slots + `full_to_swa` mapping |
| S5 accept/output | `greedy_tree_accept` (first_child/next_sibling walk) + **the four-value accept contract** (R4): `accept_index` (accepted node rows), `num_sampled` (**includes the bonus token**), `num_accepted` (**draft nodes only**), `endpoint_node` (0 for root-only). Plus `build_tree_accept_outs` → `out_tokens`, `commit_lens`, `new_seq_lens`; `path_node_ids`/`accept_index` suffixed with `-1` |
| S6 plan | `TreePrePlan`, `TreePostPlan`, `plan_tree_pre`, `plan_tree_post`, `plan_tree_commit` (dest slots/positions) + **SWA row-move plan**; compressed KV/state is written by the proven node-page freeze/commit; post-accept re-compress is not enabled |

**Commit strategy (implemented, the reference's node-page path):**

```
accept (root + accepted path)
  -> move_swa_kv moves the accepted full/SWA rows
  -> compressed node rows were recorded during verify
  -> prepare_nodes / commit_nodes (node-page freeze/commit)
  -> compact hidden/positions
```

Why the SWA move survives: tree nodes are written at verify-window columns; the next step reads positions
`prefix..prefix+commit_len-1`. Without the move those positions point at other nodes' columns.

Why node-page commit: the reference verified this path at TP16/DP16; the four tree ops are read-only with
respect to the persistent compressed KV/state, and the accepted rows are committed by `prepare_nodes`/
`commit_nodes`. Post-accept chain re-compress remains an optimization after A1.

Risks that must be tested (see §13): state-ring prefix, boundary residue (`prefix % ratio != 0`), c4 loc =
`full_loc // 4` only for boundary rows, c128 loc = contiguous prefix slice of the verify interval, order of
re-compress vs SWA move, hidden gather by `accept_index`, FP8 state, bonus-token state.

SWA row-move invariants (kept from the vllm KV-transport work, `b7cd632f2`):

1. **Plan once per (block-table group, capacity), reuse across layers.** Index-only; payload never reordered.
2. **Two phases: gather all sources into scratch, then scatter.** Never interleave over aliased rows.
3. **Overlapping sources: first-writer-wins.**
4. **Presence test is `path >= 0`, not `path != -1`.**
5. **Byte fidelity and no-touch:** destination `-1` never written; page padding never written; int8/FP8
   payload compares equal after the move.

### 5.2 New file 2 — `dspark_components/dspark_tree_ops.py` (node plan + four ops)

| section | contents |
|---|---|
| P1 node plan | `NodePlan`, `get_node_plan`, `prepare_nodes`, `commit_nodes`, `NodeRecord` descriptors; capacity buffers preallocated at init (fixed `1 + budget` rows) |
| P2 compressor | `compress_nodes` → `compressor_tree_project` + `compressor_tree` |
| P3 indexer | `quantize_indexer_nodes` + `tree_indexer` → `quant_lightning_indexer_tree` |
| P4 attention | `tree_attention` kwargs builder → `sparse_attn_sharedkv_tree` |
| P5 post-accept commit + SWA move | gather accepted hidden → existing chain compressor (synthetic prefill-like batch) writes compressed KV/state; SWA row move via `assign_cache_op`/`unidex_copy`/90-line plan; **no freeze/commit node pages** |

Imports are direct: `from sgl_kernel_npu.tree import (compressor_tree, compressor_tree_project,
quant_lightning_indexer_tree, sparse_attn_sharedkv_tree)`. No indirection, no env switch.

**Op import policy — no backend switch:**

| need | use only | never |
|---|---|---|
| SWA-only base attention | `torch.ops.npu.sparse_attn_sharedkv` + `metadata_host` | `_C_ascend.*` |
| node compressor | `sgl_kernel_npu.tree.compressor_tree_project` / `compressor_tree` | `_C_ascend.compressor_tree*` |
| node indexer | `sgl_kernel_npu.tree.quant_lightning_indexer_tree` | `_C_ascend.quant_lightning_indexer_tree` |
| node attention | `sgl_kernel_npu.tree.sparse_attn_sharedkv_tree` | `_C_ascend.sparse_attn_sharedkv_tree` |

- There is **no** `SGLANG_DSPARK_TREE_BACKEND`, no vendor/shadow/fallback, no runtime switch.
- `_C_ascend` appears in this document only as the reference's default / a rejected alternative; it is never imported.
- If `sgl_kernel_npu` or an op is missing, raise at import/first call. A missing kernel is a deployment error, not a reason to switch backends.
- `dspark_tree_ops.py` is the **single device boundary**. A GPU port later replaces this file with the same function signatures; `dspark_tree.py`, worker, and verify do not change. There is no `TreeOpBackend` protocol or registry.

### 5.3 Modified files (2 substantive new files + 4 substantive edits + 6 tiny)

| file | delta |
|---|---|
| `arg_groups/speculative_hook.py` | stop forcing `speculative_eagle_topk=1` **only** for DSV4 + NPU + tree enabled; new validation: `topk ≥ 2`, `num_draft_tokens ≥ depth + 1`, `depth ≥ 1`, budget = `num_draft_tokens - 1`, ragged mode forced `static`, tree ⇒ no compact/deferred/watermark/sampling-accept-graph. **Plus R5/R6 gates:** tree ⇒ adaptive spec / SPS / STS off, verify-width trimming off (full `1 + budget` every step), mixed prefill+decode batch runs eager or is rejected, and the per-position accept estimator is disabled or clamped to `budget`. **Plus DP/TP gates (§5.6):** tree ⇒ `attn_tp_size == 1` (or markov vocab shard provably absent), `--enable-dp-lm-head` when DP attention is on, and the tree verify width flows into `num_tokens_per_req` |
| `dspark_components/dspark_config.py` | split `gamma` (depth) from `verify_num_draft_tokens` (node width); keep the topk==1 relation `verify = gamma + 1` |
| `dspark_components/dspark_worker_v2.py` | `tree_plan` optional; `if tree_plan is not None:` around plan/accept/commit/grammar/mamba-step; replace the `assert topk in (None, 1)` with a tree branch that selects the accepted node row |
| `dspark_components/dspark_verify.py` | optional `tree_plan` argument, per-node positions/cache loc, `DFlashVerifyInput.tree_plan` carrier, tree accept/commit branch (gather accepted rows → `dest_slots`/`dest_positions`) |
| `hardware_backend/npu/attention/ascend_dsv4_backend.py` | **+400-450 — mandatory, largest surface.** `tree_plan is not None` branches: compressor (project+compress), indexer, SWA tree attention, compressed tree attention, per-node metadata. All existing chain code paths untouched |
| `dspark_components/dflash_info.py` | +2: `tree_plan: Optional[object] = None` |
| tiny (mechanical) | `spec_info.py` +8 (draft width), `dsv4_rope.py` +7 (clamp), `deepseek_v4_memory_pool.py` +8 (c4 ring), `arg_groups/fields/spec.py` +17, `arg_groups/field_order.py` +2, `dspark_draft.py` +5 (carry base_logits) |

Explicitly **unchanged**: `dspark_planner.py`, `dspark_draft.py`, `dspark_kv_inject.py`,
`dspark_deferred_commit.py`, `base_spec_worker.py`, `dflash_info_v2.py` (`DFlashVerifyInput` gains one
defaulting field only).

### 5.4 What is deliberately NOT ported from the reference implementation

Reference tree code is 19 files / 2,697 lines. Dropped entirely:

| dropped | why |
|---|---|
| `tree/backend.py` (`SGLANG_DSPARK_TREE_BACKEND`, vendor/sgl_kernel_npu/shadow switch, 30 lines) | single backend by decision |
| `tree/torch_ops.py` (vllm-ascend `.so` loader, 55 lines) | ops are Python+Triton in `sgl_kernel_npu` |
| indexer shadow/vendor half (`_shadow_outputs_match`, `_vendor_tree_indexer`, `_C_ascend` metadata op, ~150 lines) | no vendor path, no shadow |
| attention availability probe + dispatch (~60 lines) | one call site |
| node_pool availability probe + dual compressor branch (~40 lines) | one call site |
| `tree/shape.py` (41 lines) | unreferenced |
| duplicated `_empty_layout` in builder (layout already has it) | duplication |
| README backend-switch sections | not code |

Kept ≈ 1,950 of 2,697 lines (≈72%), restructured into the two files.

### 5.5 Cross-check against the vllm-ascend tree branch (design only, no code taken)

The vllm-ascend tree branch is a **different kernel stack** (`_C_ascend` vendor ops + ACLNN csrc), so nothing
is imported. Its value is that it shows where tree cost and correctness actually live.

Its four live Triton tree ops are `prune_layout`, `finalize_layout`, `freeze_node_pages`, `commit_node_pages`
— that is, **topology finalization and node-page commit**, not the compressor/indexer/attention ops we get
from `sgl_kernel_npu`. So on our side those two pieces are still framework work, and vllm's optimizations are
the reference for how to write them.

| vllm finding | what we do |
|---|---|
| budget/top-k finalization on host (`argsort`+scatter) blocked graph capture; solved with a device prune kernel (`612b84ad6`) | keep it framework-side but make it static-k `torch.topk` + preallocated index buffers (R1) — no new kernel |
| KV moves recomputed per layer and payload re-sorted; fixed by one index plan per group + gather-then-scatter (`b7cd632f2`) | adopt the plan/invariant design (R2), including `path >= 0` and first-writer-wins |
| node commit grid `(jobs, max_rows)` overflowed UB; fixed to one program per job (`2fe76a173`) | implemented via the ported node-page kernels (one program per job + internal row loop); this is the current commit path |
| scorer/policy was env-driven and model-coupled (`d6048af3f`) | adopt the injectable scorer + config policy; no env vars |
| DSV4 tree needs FULL graph and rejects PP/PCP/DCP>1, KV-transfer, dynamic depth, mixed-batch graphs | adopt as explicit startup gates (R5/R6) |
| their only in-repo test is a 21-case byte-exact KV-move test; the rest was validated externally | our test order puts byte-exact row moves and accept-plan accounting first |
| their `ancestors` padding uses a safe root address + `visible=false` | **do not copy** — the `sgl_kernel_npu` contract is `-1` padding with in-op skipping (§4) |

Not adopted: vendor `_C_ascend` ops and metadata operators, `npu_scatter_nd_update_` byte moves, global
`_LAYOUT`/`_PLAN` context hooks inside model code, vllm graph managers, Domino/Qwen GQA paths,
`candidate_size`, `expand_tree_layer_triton` (already orphaned upstream), and vllm's scheduler patches.

#### 5.5.1 Why EAGLE `build_tree` / `verify_tree_greedy` reuse is rejected (B falsified)

| evidence | consequence |
|---|---|
| `build_tree_kernel.cpp:151,181` derives parent as `selected_index // topk` | EAGLE is a **uniform topk-ary, parent-major** candidate space |
| `build_tree_kernel.cpp:163-165` skips a node whose parent was not selected | **no ancestor closure**; orphans consume budget but are unreachable |
| reference `tree/builder.py:47-49,99-129` | DSpark is **ancestor-closed + global top-`width` per level + renumbered** |
| `verify_tree_greedy.py:51` compares `candidates[i]` with `target_predict[i-1]` and never uses `retrive_next_token`/`retrive_next_sibling` | the Triton accept kernel is **chain-only**; wrong for any branch |
| `verify_tree_greedy_native` | position-indexed convention, different from the reference's node-id + `first_child` walk |

Therefore B would need an adapter as large as the reference builder, produce a different tree, and carry a
higher silent-error risk. **Rejected.**

#### 5.5.2 Why "skip current-block compressed KV during verify" is rejected

The tree attention resolves compressed slots `k >= prefix // ratio` through `tree_ancestors` → node pool
(`tree_ops_semantics.md`, `tree_reference.py:539-557`). When `prefix % ratio != 0`, the boundary partial block
must aggregate the pre-prefix token (paged state) with draft nodes (node pool). Dropping the current-block
node KV silently loses a key and changes attention. `window=128 >= verify block` only bounds the SWA window;
it does not make the compressed node KV redundant. **All four tree ops stay.**

### 5.6 Deployment topologies: DP16 / TP16 (verified by the user)

The reference branch runs on hardware at **TP16 + DP16**. This is a much stronger constraint than
"works on one card", because DP attention changes the verify tier and the collective structure.

#### DP attention (DP16)

- The verified shape is TP16 + DP16 (`--tp 16 --dp 16 --enable-dp-attention`). `--dp-size` is the
  deprecated spelling of `attn_dp_size` (`fields/parallel.py:203-205`), so this resolves to
  `attn_dp_size = 16`, `dp_size = 1`, and **`attn_tp_size = 16 // 16 // 1 = 1`**
  (`runtime_context.derive_attn_tp_size:73-76`).
- The tree is built **per DP rank on the local batch** (`bs = len(batch.seq_lens)`), not on a global batch.
  There is no cross-DP broadcast of the tree plan; each rank's tree differs because its requests differ.
  That is correct because attention is DP-local.
- DSpark requires `--enable-dp-lm-head`. With `attn_tp_size == 1` the LM head is unsharded per rank, so the
  logits are full-vocab and local; `_gather_attn_tp_logits` is a no-op.
- DP MoE collectives are **token-keyed** and driven by `batch.global_num_tokens`. The verify width must reach
  `num_tokens_per_req` (`DFlashVerifyInput` derives it from `draft_token_num`), so per-rank counts become
  `local_bs * (1 + budget)`. Hardcoding `gamma + 1` would under-count and deadlock ranks with different
  local batch sizes.
- `DPSpecPrefillCoordinationPlan` is built with `draft_width = query_token_num` and
  `verify_width = verify_num_draft_tokens`; the tree width must flow through that too.
- Idle DP ranks still enter the draft and target forwards via `run_idle_participation` to join the token-keyed
  collectives. The tree plan is skipped for idle ranks.
- Tree forces ragged mode `static`, which also disables the planner's dynamic budget broadcast and the DP
  tier all-gather. That is a simplification, not a loss.
- Consequence: under DP, tree requires `attn_tp_size == 1`. The verified DP16 + TP16 shape satisfies this.

#### Tensor parallel (TP16)

- The branch has no `get_attention_tp_size`; use `get_parallel().attn_tp_size` / `.tp_size`.
- Tree needs **full-vocab base logits and a full-vocab markov bias**. `DSparkMarkovScorer` rejects
  `head.vocab_shard != None`, stateful heads, and a bias width that differs from the logits width.
- `SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD` defaults to True and sets `vocab_shard` at TP>1, which the tree
  scorer rejects. With `attn_tp_size == 1` the head's per-partition width equals the full vocab, so
  `configure_tp_shard` leaves `vocab_shard = None` and tree proceeds. Otherwise the shard flag must be off
  and the port must guarantee the gathered full-vocab logits.
- Base-logit gather runs **once per draft proposal** (`gather_and_crop_vocab` or the multimem gatherer),
  producing `[bs*depth, vocab]` replicated on every rank.
- Per draft step there is a `[bs]` broadcast of `prev_tokens` (site `DSPARK_DRAFT`), `depth` times per
  proposal. Tree relies on it so every rank builds the same root anchor.
- Tree accept bypasses `build_accept_outs`, so the `[bs]` accept broadcasts do not run. Tree acceptance is
  deterministic target-argmax and therefore already identical across ranks.
- Tree planning, scorer, node pool, and post-accept re-compress add **no new** collectives.

#### Replicated vs sharded tree tensors

| tensor | across TP ranks |
|---|---|
| `base_logits` (after gather) | replicated, full vocab |
| tree layout / metadata / verify ids / ancestors / sparse slots / node plan | replicated (deterministic from replicated inputs) |
| target hidden rows fed to the node compressor | replicated (post attention/MoE all-reduce) |
| node KV (compressor output) | replicated (replicated compressor weights, single latent head) |
| compressed KV / indexer output | replicated (indexer weights replicated, `n_local_heads == n_heads`) |
| per-head Q and attention head outputs | **sharded** (`n_local_heads = n_heads // attn_tp`) |
| transient LM-head local logits | **sharded** by vocab, gathered before the tree sees them |

#### Unsupported combinations (must be startup gates)

- tree + sampling accept, folded proposal, or a captured decode graph (tree v1 is greedy target-argmax only);
- tree + any ragged mode other than `static`;
- tree + `attn_tp_size > 1` **and** a sharded markov head (the scorer rejects it);
- tree + DP attention without `--enable-dp-lm-head`;
- tree + non-INT8 indexer (ratio 4), or node KV dtypes outside BF16/FP16;
- tree + request-level KV transfer, PP>1, CP>1.

No checked-in test covers TP>1 + topk>1; the only in-repo tree test is the 21-case KV-transport test, and
the rest was validated externally. Our P6 must therefore add its own TP16/DP16 gates.

---

## 6. Line budget (measured on the reference, then reduced)

Reference measurements **[static]** (git blob current at `b7c7fa0a`):

| reference item | lines |
|---|---|
| `tree/` total (19 files) | 2,697 |
| of which scaffolding dropped | ~750 |
| `node_pool.py` | 476 (keep ~300) |
| `indexer.py` | 289 (keep ~130) |
| `attention.py` | 252 (keep ~140) |
| `node_commit.py` | 96 (dropped in C; ported only if A1 fails) |
| metadata + sparse + commit + accept + output + builder + layout + scoring + config + round | ~1,100 (keep all) |
| worker tree hunks | ~130-186 |
| verify tree hunks | ~68-90 |
| config width-split | ~30-35 |
| planner tree code | 0 |
| ascend backend tree branches | ~150-250 (mixed into a 755-line diff) |

Target for this work:

| item | estimate |
|---|---|
| `dspark_tree.py` (backend-neutral core) | 700-850 |
| `dspark_tree_ops.py` (node plan + 4 ops + re-compress + SWA move) | 500-600 |
| `ascend_dsv4_backend.py` (mandatory) | 400-450 |
| worker / verify / config / hook + 6 tiny files | 330-390 |
| **production total** | **1,950-2,150** |
| CPU tests (16-invariant suite + e2e) | 900-1,100 |
| optional: port all 78 reference CPU tests | ~1,790 |

Two independent reviews converged on this range (plan agent ~1,980; ultrabrain reviewer ~1,900-2,100). The
reference is ~3,619 production lines; C is ~45% smaller because it drops vendor/shadow/backend-switch,
node-page freeze/commit, and the full EAGLE adapter path. The `ascend_dsv4_backend.py +400-450` is the
mandatory entry point for tree attention inside the DSV4 layer stack and cannot be hidden under "~5 modified
files". The reference's `metadata.py:57-58`, `commit.py:48`, `sparse.py:62-94`, `builder.py:109` use
`.item()`/`.tolist()`/`argsort`; making them capture-shaped is **additive ~150 lines**, already included.

---

## 7. topk == 1 identical — enforceable invariants

The claim to prove (gate a): **output and accept length bit-identical**, with the tree flag off.

Design rules that make it mechanical:

1. `speculative_eagle_topk` keeps default `1`; the hook only relaxes its override for
   `device=npu` + `DSpark` + draft is DeepSeek-V4 + explicit tree flag.
2. `verify_num_draft_tokens == gamma + 1` whenever `topk == 1` (unchanged relation).
3. Every new entry point takes `tree_plan: TreePlan | None = None`; every new branch is
   `if tree_plan is not None:`. `None` means the exact `main` code path.
4. No new environment variables; no change to `dspark_planner.py` / `dspark_draft.py`.
5. `DFlashVerifyInput.tree_plan` defaults to `None`; backend reads it via `getattr(..., None)`.
6. Tree mode is rejected at startup if `SGLANG_DSPARK_DEFER_CONTEXT_KV`, compact ragged mode, watermark,
   or the sampling-accept graph is enabled — those are prefix/stride based and would silently truncate.
7. **R5**: tree mode is rejected if adaptive spec / SPS / STS is enabled, or if the planner would trim the
   verify width below `1 + budget`. The per-position accept estimator is disabled, or its position index is
   clamped to `budget` (vllm hit an `IndexError` here because upstream assumes exactly `num_spec_tokens`
   positions).
8. **R6**: a mixed prefill+decode batch never enters the tree verify gear; it runs eager or is rejected.
9. **R1**: every tree-path tensor is preallocated and static-shaped; no `.item()`/`.tolist()`/`.cpu()` in the
   per-step path; no allocation inside the captured region.
10. Regression proof: static diff audit (only guarded hunks), CPU unit test that runs the chain path with
    `tree_plan=None` through plan/metadata/accept, then gate (a) on A3 against the unmodified build.

Caveats found in the reference implementation that we will **not** repeat:

- the reference `_tree_config is not None` sliding-window check fires even for `topk=1`;
- the reference hook publishes `speculative_dspark_block_size` where `main` left it `None`;
- the reference `resolve_runtime_config` changes the gamma fallback order (checkpoint gamma before
  `verify-1`), which can turn a silent mismatch into a hard error.

---

## 8. Fail-loud rules (no silent truncation)

- node row with `ancestors == -1` or `>= n` reaching attention/commit → raise, do not "attend nothing";
- `#nodes > capacity`, `depth > gamma`, or `budget < depth` → raise;
- `topk < 2` while the tree path is requested, or `topk` outside `[1,2048]` → raise (1 means chain, not tree);
- `ori_sparse_indices` width not a multiple of 128, `ratio` outside `{4,128}` → raise;
- padded/unused node rows carry the full invalidation tuple — `token=-1`, `depth=0`, `parent=-1`,
  `first_child=-1`, `next_sibling=-1`, `visibility=false`, `position=-1`, `cache_loc=-1` — and can never be
  accepted; a partially invalidated row is a bug;
- a destination claimed by two moves outside the first-writer-wins rule → raise, never silently drop;
- tree enabled with an incompatible feature (deferred commit / compact / watermark / accept graph / adaptive
  spec / SPS / STS / KV transfer / PP>1 / CP>1 / DP-attention) → raise at startup;
- mixed prefill+decode batch inside a tree gear → eager or raise;
- any tensor created inside a captured forward → forbidden; buffers preallocated in `__init__`.

---

## 9. Phases and gates

| phase | work | evidence class |
|---|---|---|
| P0 | **Falsification (planned; not runnable on the analysis workstation, no torch/NPU)**: A1 re-compress vs node-page commit; A2 four-op oracle; A3/A4/A5 code assertions. Must run on hardware | [hw] |
| P1 | branch from `main`, push to `unclezhou`; venv; config/hook gates + validation only (tree unreachable) | [static] |
| P2 | `dspark_tree.py` core (config/builder/metadata/sparse/accept/output/plan), **capture-shaped per §4.5** + the 16-invariant CPU suite | [cpu] |
| P3 | `dspark_tree_ops.py` (node plan, 4 op adapters, post-accept re-compress, SWA move) + oracle contract tests | [cpu] |
| P4 | `ascend_dsv4_backend.py` routing (+ `dsv4_rope`, `memory_pool`) | [static] |
| P5 | worker/verify/config integration + commit path; **topk==1 diff audit** | [cpu] + [static] |
| P6 | A3 gates: (a) topk=1 identical (DP1 **and** TP16/DP16) → (b) topk=2 eager DP1 → (b2) topk=2 eager TP16/DP16 → (c) greedy short-context equality → (d) accept_len vs verify cost → (e) graph capture only if (d) is negative | [hw] |

Notes:

- **P0 is a hard gate.** A1 fail ⇒ port `node_commit.py` (+~350 lines) before P1. A2 fail ⇒ stop; kernel work
  in `sgl-kernel-npu` is out of this repository's scope.
- (b2) is not optional: the reference is only known-good at TP16/DP16; a tree that works at DP1 but not DP16
  is not shippable. It also exercises the `num_tokens_per_req` / coordination plumbing from §5.6.
- (d) is the decision point: vllm's DSV4 tree only claims FULL-graph operation, and their measured win includes
  the graph. If eager (d) fails on verify cost, the answer is (e), not kernel tuning.
- (e) has two known kernel-side blockers in `sgl_kernel_npu` (indexer `.item()`/select-all, project validation
  syncs). Those are upstream kernel issues; the framework side must already be capture-shaped.
- No kernel/tiling tuning before (a)-(d) pass.

---

## 10. Decisions (resolved) and remaining gate

Resolved with the user:

1. **CPU torch**: create a local venv under `dspark_topk/` and install CPU `torch`, so P2/P3 unit tests run
   on this workstation (`[cpu]`). NPU stays `[hw]` on the A3 box.
2. **v1 feature gating**: tree mode **fails loudly at startup** when any of `SGLANG_DSPARK_DEFER_CONTEXT_KV`,
   compact ragged mode, watermark, or the sampling-accept graph is enabled. No silent prefix-semantics use.
3. **Padding policy**: every request always emits exactly `1 + budget` candidate rows; unused rows are
   explicitly invalidated (`position=-1`, `ancestors=-1`, `cache_loc=-1`) and can never be accepted. Stride
   is therefore constant, which keeps the existing uniform verify window and keeps a future graph path open.
4. **Branch**: `feat/npu-dsv4-dspark-tree`, from local `main = 5d4a6f499e`, pushed to remote `unclezhou`.
5. **Hardware baseline**: the reference branch is verified at TP16 + DP16, so the port must reproduce that
   topology (see §5.6), not just DP1. This replaces the earlier, incorrect "DP unvalidated" assumption.

Remaining gate: explicit user go-ahead before P1 creates the branch / venv and touches no sglang code.

---

## 11. Traceability

| this design | reference implementation |
|---|---|
| S1 config | `tree/config.py`; `speculative_hook.py:774-793` |
| S2 scoring/builder | `tree/scoring.py`, `tree/builder.py` |
| S3 metadata | `tree/metadata.py` |
| S4 sparse | `tree/sparse.py` |
| S5 accept/output | `tree/acceptance.py`, `tree/output.py` |
| S6 plan/commit plan | `tree/round.py`, `tree/commit.py` |
| P1 node plan | `tree/node_pool.py:26-96, 301-476` |
| P2 compressor | `tree/node_pool.py:118-205` |
| P3 indexer | `tree/indexer.py:69-229` (minus vendor/shadow) |
| P4 attention | `tree/attention.py:36-231` (minus dispatch) |
| P5 post-accept commit | chain `forward_compress` + `dspark_kv_inject` + `tree/commit.py:78-101` (SWA move); node-page freeze/commit deliberately **not** ported |
| backend branches | `ascend_dsv4_backend.py:528-681, 1041-1128, 2452-2659, 2144-2271` |
| R2 row-move plan | vllm-ascend `tree/transport.py`, `tree/kv_project.py`, `tree/kv_binding.py` (design only) |
| R3 commit order | v5: gather hidden → chain re-compress → SWA move; v3 freeze/commit order kept only as fallback |
| R4 accept contract | vllm-ascend `tree/acceptance.py`, `tree/rejection_sampler.py` |
| R5/R6 gates | vllm-ascend `tree/speculator.py` (FULL, PP/PCP/DCP, KV transfer, fixed depth), `aclgraph_utils.py` (mixed batch) |

---

## 12. Revision log

| rev | change | trigger |
|---|---|---|
| v1 | initial design: 2 new files, 5 modified files, topk==1 inert via `tree_plan=None`, graph deferred to P6 | `require.txt` + sglang reference audit |
| v2 (R1) | plan/accept/commit must be capture-shaped from P2 (fixed capacity, static-k topk, no host readback, no in-capture allocation, uniform rows) | vllm DSV4 tree requires FULL graph; `612b84ad6` shows host finalization as the blocker |
| v2 (R2) | KV row-move plan invariants: plan-per-group reuse, index-only sort, two-phase gather/scatter, first-writer-wins, `path >= 0`, byte fidelity | `b7cd632f2` + its 21-case NPU test |
| v2 (R3) | explicit commit ordering freeze → relocate → commit → compact hidden | `2fe76a173` / `full_graph.py` |
| v2 (R4) | four-value accept contract (`accept_index`, `num_sampled` incl. bonus, `num_accepted` drafts only, `endpoint_node`) | vllm acceptance semantics |
| v2 (R5) | tree ⇒ adaptive spec/SPS/STS off, no verify-width trimming, accept estimator disabled or clamped | sglang has budget-trimming machinery vllm does not; vllm hit the per-position stats `IndexError` |
| v2 (R6) | mixed prefill+decode batch → eager or rejected | vllm `aclgraph_utils.py` mixed-batch guard |
| v2 (budget) | production 1,950-2,550; with tests 2,400-3,200 | R2/R4/R5/R6 additions |
| v3 (DP16) | tree is built per DP rank on the local batch; verify width must flow into `num_tokens_per_req` and the prefill coordination plan; idle ranks must still join token-keyed collectives; require `--enable-dp-lm-head` | user report: reference verified at TP16/DP16 |
| v3 (TP16) | tree requires full-vocab base logits and an unsharded markov bias; `attn_tp_size == 1` makes `vocab_shard` absent, otherwise the shard flag must be off; per-proposal full-vocab gather + per-step `[bs]` draft broadcast; tree accept has no broadcast | reference branch TP handling (`DSparkMarkovScorer` shard rejection, `gather_and_crop_vocab`, `SpecTpSync`) |
| v3 (gates) | startup gates for tree + non-static ragged / sampling accept / folded proposal / captured decode / `attn_tp>1` with sharded markov / DP without dp-lm-head / non-INT8 indexer / KV transfer / PP>1 / CP>1 | §5.6 |
| v3 (budget) | production 2,030-2,610; with tests 2,480-3,260 | DP/TP plumbing |
| v4 (EAGLE falsified) | B (reuse `build_tree_kernel_efficient` + `verify_tree_greedy`) rejected: uniform topk-ary + no ancestor closure; Triton accept is chain-only; produces a different tree | `build_tree_kernel.cpp:151,181,163-165`; `verify_tree_greedy.py:51` |
| v4 (compressed KV) | "skip current-block compressed KV" rejected: boundary partial block (`prefix % ratio != 0`) needs node KV | `tree_ops_semantics.md`; `tree_reference.py:539-557` |
| v5 (architecture) | **C**: lean torch tree core + four ops + implemented node-page commit; post-accept re-compress deferred; no EAGLE reuse | plan agent + ultrabrain reviewer, both ranked C first |
| v5 (commit) | commit = accept → gather accepted hidden → chain `forward_compress` (prefill-like, ≤ depth+1 rows) → SWA row move; node-page freeze/commit dropped (~300 lines) | chain compressor is first writer; tree ops read-only |
| v5 (budget) | production 1,950-2,150; tests 900-1,100 (16 invariants) or ~1,790 (all 78); backend +400-450 mandatory | two independent estimates |
| v5.2 (trim) | removed reference-only helpers (`build_chain_*`, `apply_tree_commit`, `plan_tree_round`) and the duplicate ops commit plan; production ≈2,790, tests 66 | CPU call-graph audit; 66/66 tests pass |
| v5 (P0) | P0 is a hard falsification gate: A1 re-compress parity, A2 four-op oracle; A1 fail ⇒ port `node_commit.py` (+~350) | reviewer |

---

## 13. Risk register (v5)

| # | risk | severity | mitigation |
|---|---|---|---|
| 1 | `verify_num_draft_tokens` is simultaneously depth, window width, KV stride, commit length, and mamba step budget; tree splits it into `depth` and `1 + budget` | high | single point of truth in `dspark_config.py`; startup asserts; topk==1 relation untouched |
| 2 | DP token-keyed collectives deadlock if the verify width is not reflected in `num_tokens_per_req` / coordination | high | (b2) TP16/DP16 gate before any perf work; explicit width plumbing |
| 3 | Tree silently diverges across TP ranks if base logits are sharded or the draft broadcast is disabled | high | scorer rejects `vocab_shard`; assert full-vocab width; keep `DSPARK_DRAFT` broadcast |
| 4 | Kernel-side graph blockers (indexer `.item()`/select-all, project validation syncs) | medium | framework side capture-shaped; kernel fix is a separate upstream change |
| 5 | Per-position accept statistics assume exactly `depth` positions; `budget > depth` breaks them | medium | disable or clamp the estimator in tree mode |
| 6 | Mamba/KDA step mapping (`commit_lens - 1`) is chain-only | medium | select the last accepted node row explicitly; assert |
| 7 | Padding convention mismatch: `sgl_kernel_npu` uses `-1`, vllm uses safe-root + invisible | medium | follow the kernel contract; fail loudly on partially invalidated rows |
| 8 | Mixed prefill+decode batch cannot replay a fixed tree gear | low/medium | eager or reject per batch |
| 9 | No in-repo test covers TP>1 + topk>1 | medium | P6 (b2) is a mandatory hardware gate; add CPU tests for the width plumbing |
| 10 | Line-count creep beyond the v5 estimate | low | 1,950-2,150 production; backend +400-450 is mandatory; report net diff, separate ported vs new logic |
| 11 | **A1** (follow-up only): post-accept chain re-compress diverges from the proven node-page commit | low (current path unaffected) | keep the node-page commit; run A1 only before enabling the optional re-compress |
| 12 | **A2**: `sgl_kernel_npu.tree` ops diverge from the `tree_reference.py` oracle | **critical** | P0: run Repo C `test_tree_*` + oracle on A3; fail ⇒ kernel work, stop |
| 13 | Boundary partial block (`prefix % ratio != 0`) drops or mis-aggregates a compressed key | high | test `prefix=9, ratio=4`; inspect `(k+1)*ratio-1-prefix` step and paged+node aggregation |
| 14 | State-ring corruption / double-write / bonus-token state | high | two-step loop, compare `state_cache` to pure-chain baseline; assert bonus counted once |
| 15 | `verify_tree_greedy_native` flat-row → node-id mapping off by one, corrupting `dest_slots` | high | golden test native vs `greedy_tree_accept` on a branched tree; assert root at col 0 |
| 16 | DP16 token-keyed collective under-count / hang with unequal local batch | high | static assert `num_tokens_per_req == 1+budget`; CPU mock with mismatched local bs; P6 (b2) |

---

## 14. Review record (subagent rounds)

| round | reviewer | scope | verdict |
|---|---|---|---|
| R1 | `explore` ×4 | EAGLE reuse surface; post-accept re-compress feasibility; repo conventions / GPU boundary; NPU-native EAGLE op ABI | found the EAGLE builder/accept mismatches and the NPU-native op set |
| R1 | `Oracle` | A/B/C adversarial review | **unavailable** (`Cannot connect to API`, twice) |
| R1 | `Artistry` | three non-conventional smaller designs | proposed the SWA-covers-verify-block shortcut; later rejected as unproven |
| R2 | `plan` | decision-complete architecture + plan | **C**; B rejected; production ~1,980; backend +400 mandatory |
| R2 | `ultrabrain` (Oracle substitute) | attempt to falsify C/B | **C > A > B**; B falsified by kernel semantics; skip-compressed-KV rejected; 6 invariant tests + 5 failure modes |

Reference evidence base: 47 commits (33 behavior fixes), 45 files, +5,024/-94; `tree/` 2,697 lines;
CPU tests 1,337 lines / 78 tests.

Evidence anchors for the two decisive falsifications:

- B: `build_tree_kernel.cpp:151,181` (`parent = selected_index // topk`), `:163-165` (orphan `continue`),
  `verify_tree_greedy.py:51` (chain-only accept), reference `tree/builder.py:47-49,99-129` (ancestor-closed).
- Skip compressed KV: `tree_ops_semantics.md` (tree mapping), `tree_reference.py:539-557`, boundary
  `prefix % ratio != 0` partial block.

No sglang code has been modified. P0 is the first executable step and only after explicit approval.
