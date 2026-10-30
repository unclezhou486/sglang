# DSV4 + DSpark tree speculative decoding (topk > 1) on Ascend NPU

> **Status: v5.2 implemented, CPU 66/66 passing, A3 hardware gates pending.**
> Branch: `feat/npu-dsv4-dspark-tree-v5`, baseline `main = 5d4a6f499e`, pushed to `unclezhou`.
> Scale: 17 files, `+3,832 / −84`; production about 2,790 lines, tests about 1,042 lines (66 cases).
> CPU verification: `python3 -m pytest -q test/registered/unit/spec/test_dspark_tree_core.py test/registered/unit/spec/test_dspark_tree_ops.py test/registered/unit/spec/test_dspark_tree_rope.py`

---

## 1. Background and goals

### 1.1 Problem

DSpark in `main` implements only chain speculative decoding (`topk == 1`): each request verifies one candidate chain per step, so the accepted length is limited by single-chain hits. Widening the candidates into a tree (`topk > 1`) can raise the accepted length, but the DSV4 decode state is hybrid: SWA original KV, compressed KV pools (ratio 4/128), lightning indexer. Every node on the tree needs its own topology, absolute positions, compressed KV, indexer top-k, and ancestor attention mask. A mask or sparse-index patch alone is not enough; these per-node inputs must all be supplied.

### 1.2 Goals

Deliver a production-grade tree that supports arbitrary `topk` / `depth` / `budget`, not a fixed-config MVP.

1. With the tree switch off, output and accept length are bit-identical to `main`.
2. With the tree on, real requests run end to end, with no device assert and no host index error.
3. Under greedy (temperature 0) short context, `topk=2` and `topk=1` produce identical tokens.
4. On low-latency workloads (small batch, 1k/4k/8k input, 256 output), the accept-length gain exceeds the verify cost.
5. Every conclusion is labeled hardware-verified / CPU-only / static-only.

### 1.3 Terminology

| term | meaning |
|---|---|
| `topk` | candidate width kept per level (branching factor) |
| `depth` | tree depth, equal to the draft horizon `gamma` |
| `budget` | number of non-root nodes kept per request; verify width is `1 + budget` |
| ancestor closure | when a node is kept, its entire ancestor chain must be kept |
| node-page commit | write the compressed KV/state of accepted nodes to a staging page first, then freeze/commit into the persistent pool |
| SWA | sliding window attention, the original KV of the recent window |
| indexer | the DSV4 lightning indexer, top-k over compressed history |
| V1 / V2 | two falsification experiments: V1 is the equivalence of post-accept re-compress and node-page commit, V2 is the agreement of the four operators with the CPU oracle |
| A3 | Ascend A3 machine, where the hardware gates run |

---

## 2. Architecture conclusions

### 2.1 Scheme C

**A backend-neutral pure torch tree core + four `sgl_kernel_npu` tree-operator adapters + node-page commit verified by the reference implementation.**

- `dspark_tree.py`: topology, ancestor closure, metadata, sparse slots, accept/output, commit plan. Depends only on torch and the standard library, testable on CPU, reusable on GPU.
- `dspark_tree_ops.py`: the single device boundary. NodePlan, the four `sgl_kernel_npu.tree` operators, node-page commit, SWA row move. No vendor fallback, no shadow, no environment-variable switch.
- `ascend_dsv4_backend.py`: routing only. When `tree_plan is not None`, it dispatches compressor, indexer, SWA, and compressed attention to `dspark_tree_ops`.

topk=1 never goes through the tree: `tree_plan` defaults to `None`, all tree branches short-circuit, and the original `main` chain path runs.

### 2.2 Per-step dataflow

| stage | chain (topk=1) | tree (topk>1) |
|---|---|---|
| draft proposal | 1 draft forward, `bs*(gamma or gamma+1)` rows | same, unchanged |
| planning | none | `plan_tree_pre`: topology, metadata, sparse slots, NodePlan |
| target verify | `[anchor, d1..dγ]` | `[root, n1..nN]`, `bs*(1+budget)` rows |
| compression / indexer / attention | contiguous blocks | per node, the four tree operators |
| accept | scalar `correct_len` | node path + bonus + commit plan |
| commit | prefix gate `col < commit_lens` | SWA row move + node-page commit |
| forwards per step | 2 | 2 |

Tree mode has no second draft forward. Draft still produces only `base_logits [bs, depth, vocab]`, and the markov scorer expands the tree level by level over these logits; a node's target hidden comes from target verify, which is also where the node KV comes from.

### 2.3 Four-operator pipeline

```text
compressor_tree_project(x, wkv, wgate, nodes, ape, positions, cu)
    -> nodes fp32 [T+1, 1, 2*c*d] (row 0 reserved)

compressor_tree(state, nodes, norm, cos, sin, table, prefix,
                ancestors, depths, requests, ratio, rope_dim, epsilon)
    -> node_kv bf16 [T, d] (zero rows for non-compressed boundary nodes)

quantize(node_kv) -> int8 [T,128] + fp16 scale

quant_lightning_indexer_tree(query, history, weights, query_scale, history_scale,
                             nodes_i8, node_scale, ancestors, prefixes, requests,
                             table, cu, lengths, residual, metadata, topk)
    -> cmp_sparse_indices int32 [T, 1, topk]

sparse_attn_sharedkv_tree(q, ori_kv, cmp_kv, cmp_tree_kv=node_kv,
                          tree_ancestors, tree_prefix, seqused_kv,
                          ori_sparse_indices, cmp_sparse_indices, ...)
    -> out, softmax_lse
```

Contract points:

- `compressor_tree_project` is the only in-place operator; `nodes` must be preallocated.
- `metadata` is int32[1024], validated then ignored, so passing a static zero buffer is enough.
- `cu` is used only to assert "one query, one node".
- `seqused_kv = prefix + depth + 1`.
- The `ori_sparse_indices` width must be a multiple of 128.
- SWA-only layers do not use the tree attention operator; they use the base `torch.ops.npu.sparse_attn_sharedkv` with `metadata_host`.

### 2.4 Commit path

```text
verify phase:
  compressor_tree_project + compressor_tree
      -> compute temporary compressed node KV for all tree nodes
      -> record a NodeRecord for each node

after accept:
  move_swa_kv
      -> move the accepted full/SWA rows to contiguous history positions
  prepare_nodes + commit_nodes
      -> node-page freeze/commit, writing compressed KV/state into the persistent pool
  commit_hidden
      -> commit hidden in accept_index and target-depth order
```

This is the path the reference implementation already verified on TP16/DP16.

Post-accept chain re-compress (V1) is not enabled: it would re-compress the accepted path after accept with the existing chain compressor, which could remove node-page commit, but it must first be proven bit-identical to node-page commit. V1 can only be done on A3, so the verified path is kept for now.

---

## 3. Why not other schemes

### 3.1 Scheme A not adopted: port the reference tree layer wholesale

The reference implementation's 19-file `tree/` package has about 3,619 production lines before tests, and it carries vendor `_C_ascend`, shadow, backend switch, and node-page commit scaffolding. This is exactly what the constraints forbid and what the user rejected as "too large".

### 3.2 Scheme B not adopted: reuse the EAGLE tree operators

EAGLE's `build_tree_kernel_efficient` and `verify_tree_greedy` cannot be used directly for DSpark:

- `build_tree_kernel.cpp:151,181` derives the parent with `selected_index // topk`, a uniform topk-ary, parent-major candidate space.
- `build_tree_kernel.cpp:163-165` skips a node outright when its parent was not selected, so ancestor closure is not guaranteed.
- The reference DSpark builder is an ancestor-closed, per-level global top-width, renumbered tree; the two are not the same tree.
- `verify_tree_greedy.py:51` compares `candidates[i]` with `target_predict[i-1]` and never uses `retrive_next_token` and `retrive_next_sibling`; this Triton accept kernel supports chains only.

Reusing it would need an adapter as large as the reference builder, and it would produce a different tree, amplifying silent orphan and accept errors.

### 3.3 Not skipping the current-block compressed KV during verify

Tree attention resolves compressed slots `k >= prefix // ratio` to the node pool selected by `tree_ancestors`. When `prefix % ratio != 0`, the boundary partial block must aggregate the tokens before prefix (paged state) with the draft nodes (node pool). Skipping the current-block compressed KV silently loses a key. All four tree operators therefore stay.

---

## 4. Constraints and invariants

### 4.1 topk=1 fully identical

1. `speculative_eagle_topk` defaults to 1; it is relaxed only when NPU, DSpark, DSV4, and an explicit tree switch all hold.
2. With `topk=1`, `verify_num_draft_tokens == gamma + 1`.
3. Every tree entry point accepts `tree_plan=None`, and every tree branch is `if tree_plan is not None`.
4. No new environment variables; `dspark_planner.py` and `dspark_draft.py` are unchanged.
5. `DFlashVerifyInput.tree_plan` defaults to `None`.
6. Tree mode is mutually exclusive with deferred commit, compact ragged, watermark, and the sampling accept graph; it errors at startup.
7. Tree mode turns off adaptive spec, SPS, and STS, and does not trim the verify width.
8. A mixed prefill+decode batch does not enter the tree gear.
9. Tree-path tensors are preallocated with fixed shapes; the per-step path has no `.item()`, `.tolist()`, or `.cpu()`.
10. Regression method: static diff audit plus CPU tests, then token-by-token alignment against the unmodified build on A3.

### 4.2 Fail-loud

- A node row with `ancestors == -1` or out of bounds that reaches attention/commit raises.
- `#nodes > capacity`, `depth > gamma`, `budget < depth` raise.
- Requesting the tree path with `topk < 2` raises.
- An `ori_sparse_indices` width that is not a multiple of 128, or a `ratio` outside `{4,128}`, raises.
- Unused nodes must be fully invalidated: `token=-1`, `depth=0`, `parent=-1`, `first_child=-1`, `next_sibling=-1`, `visibility=false`, `position=-1`, `cache_loc=-1`.
- Enabling tree mode together with an incompatible feature errors at startup.
- A mixed prefill+decode batch inside the tree gear runs eager or errors.
- Creating tensors inside the capture region is not allowed; buffers are preallocated in `__init__`.

### 4.3 DP16 / TP16

The reference implementation was verified on TP16 + DP16, a topology that corresponds to `attn_tp_size = tp_size // attn_dp_size = 1`.

- The tree is built on each DP rank's local batch, not the global batch.
- DP requires `--enable-dp-lm-head`.
- The verify width must flow into `num_tokens_per_req`; the per-rank count is `local_bs * (1 + budget)`.
- Idle DP ranks still enter draft and target forward, joining the token-keyed collective.
- The tree requires `attn_tp_size == 1`; otherwise `DSparkMarkovScorer` rejects vocab-sharded markov.
- Tree mode forces static ragged and turns off the planner's dynamic budget broadcast and DP tier all-gather.

### 4.4 Graph capture shape

The tree path was written against graph capture constraints from P2 onward:

- Fixed capacity, `1 + budget` rows per request, preallocated.
- Static-k `torch.topk`, not variable-length `argsort`.
- No host readback in the per-step path; no `.item()`, `.tolist()`, `.cpu()`.
- No allocation inside the capture region; outputs go into preallocated buffers.
- Constant row count, with unused rows explicitly invalidated.
- Mixed batches run eager or are rejected.

Current known kernel-side graph blockers: the indexer's `lengths.max().item()` and select-all branch, and the validation sync in `compressor_tree_project`. These live in `sgl-kernel-npu`, not in this repository.

---

## 5. Implementation inventory

### 5.1 New files

| file | lines | responsibility |
|---|---:|---|
| `dspark_components/dspark_tree.py` | 1,134 | device-independent tree core |
| `dspark_components/dspark_tree_ops.py` | 835 | NodePlan, four tree operators, node-page commit, SWA move |

`dspark_tree.py` is split into eight sections: config, scoring/builder, layout, metadata, sparse, accept/output, commit plan, round. `dspark_tree_ops.py` is split into five sections: node plan, compressor, indexer, attention, node commit.

### 5.2 Modified files

| file | change |
|---|---|
| `ascend_dsv4_backend.py` | +489: tree branches for compressor, indexer, SWA, and compressed attention |
| `dspark_worker_v2.py` | +134: tree config, plan pre/post, accept/commit |
| `dspark_verify.py` | +78: `tree_plan` argument, tree commit branch |
| `dspark_draft.py` | +5: `DraftProposal.base_logits` |
| `dspark_config.py` | +61: decouple gamma from verify width |
| `speculative_hook.py` | +94: tree arguments, width resolution, gating |
| `dflash_info.py` | +2: `DFlashVerifyInput.tree_plan` |
| `dsv4_rope.py` | +7: negative position clamp |
| `deepseek_v4_memory_pool.py` | +8: c4 ring covering the tree width |
| `fields/spec.py`, `field_order.py`, `spec_info.py` | +27: tree arguments and draft width |

### 5.3 Explicitly not doing

- No new `tree/` folder.
- No copying SGLang.
- No second DSpark worker, algorithm enum, `SpecInputType`, or `ForwardBatch` field.
- No vendor `_C_ascend`, shadow, or `SGLANG_DSPARK_TREE_BACKEND`.
- No rewriting the four operators at the framework layer.
- No tiling or kernel tuning before (a) to (d) pass on A3.

---

## 6. Verification status

### 6.1 Verified

- CPU: 66/66 passing. 45 core, 20 op contract, 1 RoPE clamp. Environment is torch 2.9.1+cpu, Python 3.14 venv.
- `py_compile` and `compileall` pass on every changed file.
- Production code contains no `_C_ascend`, `SGLANG_DSPARK_TREE_BACKEND`, or `tree/` package reference.
- AST check: no missing cross-module imports or call signatures.
- Tests cover tree topology, ancestor closure, metadata, sparse slots, accept/output, commit plan, NodePlan, operator call contracts, RoPE clamp, and width configuration.

### 6.2 Not verified

- The real numerics of the four `sgl_kernel_npu.tree` operators (V2).
- Real runtime integration of backend and worker.
- A3 eager, greedy consistency, accept length versus verify cost.
- TP16/DP16.
- Graph capture.

### 6.3 A3 gates

Run in order:

1. (a) Tree off, topk=1, record output and accept length as the baseline.
2. (b) Tree on, `topk=2`, `budget >= depth`, `SGLANG_RAGGED_VERIFY_MODE=static`, eager.
3. (c) Greedy short context: topk=2 tokens match topk=1.
4. (d) Same workload: `accept_len(topk=2)/accept_len(topk=1) > verify_cost(topk=2)/verify_cost(topk=1)`.
5. (e) Repeat (a) to (c) on TP16/DP16.
6. (f) Do graph capture only if (d) is negative.

V2 (the four operators against the `tree_reference.py` oracle) must pass on A3 before sign-off.

---

## 7. Risks

| # | risk | severity | mitigation |
|---|---|---|---|
| 1 | The four operators disagree with the CPU oracle (V2) | high | Run `test_tree_*` and the oracle on A3; stop on failure |
| 2 | Backend/worker runtime integration has shape/dtype/order problems | high | Verify A3 (a) to (d) step by step, fix on failure |
| 3 | DP token count mismatch hangs the collective | high | Assert `num_tokens_per_req == 1+budget`; TP16/DP16 gate |
| 4 | Markov head sharding or `attn_tp > 1` | high | `DSparkMarkovScorer` rejects vocab_shard; the tree requires attn_tp=1 |
| 5 | mamba + tree | medium | `_commit_target_mamba_states_after_verify` keeps the topk assert; DSV4 is not mambaish |
| 6 | grammar / watermark + tree | medium | Fail loudly for now; adapt later if needed |
| 7 | Graph capture blocked by kernel-side host sync | medium | The framework side is already written for graphs; the kernel fix belongs to `sgl-kernel-npu` |
| 8 | V1 re-compress is not equivalent to node-page commit | low (current path unaffected) | Only an optional optimization after V1 passes |
| 9 | Size above the lean estimate | low | Current production about 2,790; going smaller must wait for V1 to replace the commit path |

---

## Appendix A: Traceability

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
| P3 indexer | `tree/indexer.py:69-229` |
| P4 attention | `tree/attention.py:36-231` |
| P5 node commit | `tree/node_commit.py`, `tree/commit.py:78-101` |
| backend branches | `ascend_dsv4_backend.py:528-681, 1041-1128, 2452-2659, 2144-2271` |

## Appendix B: Evidence anchors

- Scheme B falsified: `build_tree_kernel.cpp:151,181` (parent = selected_index // topk), `:163-165` (orphans hit a bare continue), `verify_tree_greedy.py:51` (chain-only accept), reference `tree/builder.py:47-49,99-129` (ancestor-closed).
- Skipping compressed KV falsified: `tree_ops_semantics.md`, `tree_reference.py:539-557`, and the boundary partial block at `prefix % ratio != 0`.
- DP/TP: `runtime_context.derive_attn_tp_size:73-76`, `fields/parallel.py:203-205`.
