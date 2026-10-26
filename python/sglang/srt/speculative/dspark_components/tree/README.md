# DSpark tree core (CPU-testable)

Pure-Python/torch implementation of the DSV4 + DSpark tree topk>1 building
blocks. This package imports only `torch` and `dataclasses`; it never imports
`torch_npu`, ACL, or the sglang runtime, so every module is CPU-testable.

## Modules

| Module | Responsibility |
|---|---|
| `shape.py` | `TreeShape`: depth (gamma), non-root node budget, topk, verify width. |
| `layout.py` | `TreeLayout` + torch `finalize_tree_layout_torch` + `build_chain_layout`. |
| `scoring.py` | `select_candidates`, `IndependentScorer`, `DSparkMarkovScorer`. |
| `builder.py` | Level-wise tree builder with ancestor-closed pruning. |
| `acceptance.py` | Greedy tree walk over `first_child` / `next_sibling`. |
| `output.py` | Accepted-path gather + `accept_index` + commit lengths. |
| `metadata.py` | Per-node DSV4 verify geometry (positions, seqused, ancestors). |
| `sparse.py` | Per-query original-KV slots (prefix window + ancestors, `-1` pad). |
| `attention.py` | Native op adapter + CPU reference (test-only). |
| `commit.py` | Root/accepted-node commit plan + device hook boundary. |
| `config.py` | Pure `resolve_tree_config` validation. |
| `round.py` | One-call `plan_tree_round`: builder → metadata → sparse → accept → output → commit. |

## Contracts

- Node 0 is the accepted root. Non-root nodes are 1..N; `budget` counts them.
- `topk=1` is the chain: `budget == depth`, verify width `1 + depth`.
- `topk>1`: verify width `1 + budget`.
- `tree_ancestors` is root-first, includes self, `-1` padded.
- `positions`: root at `prefix`, node depth `d` at `prefix + d`.
- `node_seqused_kv = prefix + depth + 1`.
- Sparse ori slots: prefix window + visible ancestors, siblings/descendants
  excluded, `-1` padding, width 128-aligned.
- Greedy accept: follow the first child matching the parent's target argmax;
  the bonus is the target argmax at the endpoint.

## Attribution

`shape.py`, `layout.py`, `acceptance.py` and the builder/scorer structure are
adapted from vllm-ascend's `tree/` package (Apache-2.0). The vllm Python tree
layer is not vendored wholesale because it is coupled to vllm's runner, pool,
and `MarkovScorer` model API; sglang's Markov head exposes
`step_bias(prev_tokens, hidden_states, state)` instead.

## Device-readiness (important)

- `builder.py` and `acceptance.py` are tensor-vectorized and suitable for device
  execution.
- `metadata.py`, `sparse.py`, `output.py` and `commit.py` currently use
  host-planning loops over small tensors. They are CPU-correct, but they must
  not run inside a captured forward: before graph capture they need
  tensorized/device-side variants (see vllm-ascend's
  `native_attention._original_slots` and `triton/tree/node_pool.py`).
- `plan_tree_round` is a host planner; call it per round outside the captured
  region.

## Device-only boundaries (never faked)

- Native tree attention: `torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree`.
- DSV4 pool relocation: the pools have no `move_kv_cache`; `apply_tree_commit`
  raises unless a device hook is supplied.
- TP-sharded Markov `w2`: rejected (`SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD`).
- Stateful Markov heads, sampling accept, graph capture: rejected in v1.
- The CPU attention reference is reachable only when
  `SGLANG_DSPARK_TREE_ALLOW_CPU_REFERENCE=1`; it is test-only.

## Tests

```bash
PYTHONPATH=python python -m pytest test/registered/unit/spec/dspark_tree -q
```
