# DSV4 + DSpark 树形投机解码：后续开发路线

> **基线**：`feat/npu-dsv4-dspark-tree-v5 @ 6a7d2a2b4d`  
> **当前状态**：CPU 66/66 通过；A3/V2 未跑；树模式 eager；mixed batch 未支持；非 greedy 未显式拒绝  
> **直接参考**：sglang 参考分支 `feat/npu-dsv4-dspark-topk @ b7c7fa0a`  
> **优化参考**：本地 vllm-ascend `tree261025-rebase @ 2fe76a173`  
> **目标**：从 correctness-first 的 eager 树，推进到可生产、可回滚、可测性能的树路径  
> **非目标**：重写 PCTree 算法；改变 topk=1 链式路径；重新引入 vendor `_C_ascend` / shadow / backend switch

---

## 0. TL;DR

建议按下面的顺序推进，不要跳步：

```text
P0 正确性闸门
  ├─ 非 greedy fail-loud
  ├─ compact 路径显式拒绝/接入
  ├─ V2：四算子对 CPU oracle
  ├─ A3 (a)-(e)：topk=1、树 eager、greedy 一致性、accept length、TP16/DP16
  └─ DP token 计数 + 测试覆盖补齐
        │
        ▼
P1 capture-safe 重构
  ├─ TreeBuilder 预分配
  ├─ metadata / sparse / accept / commit 设备化
  └─ ops 静态 buffer + node commit 设备化
        │
        ├──────────────┐
        ▼              ▼
P2 入图            P3 mixed batch
  ├─ target verify 图（MVP）   ├─ two-pass（先做）
  ├─ tree build 图             ├─ single-forward phase view
  └─ accept/commit 图          └─ mixed + graph（最后）
        │
        ▼
P4 性能与可观测性
  ├─ accept length / 每轮延迟 / tokens/s / p50/p99
  ├─ budget / topk / depth 调参
  └─ chain/tree 混合策略
```

**最小可生产版本**：

```text
P0 全部通过
+ A3 (d) 判定
+ 若 (d) 为负：target verify 图
+ tree 默认关 / 失败回 topk=1
+ A3 证据
```

**完整生产版本**：

```text
P0 + P1 + P2 全量 + P3 + P4
```

**最重要的两条纪律**：

1. 不要直接把 `disable_decode_cuda_graph` 删掉。当前树路径有 host sync 和动态分配，直接开图会在 capture 阶段失败。
2. 不要同时做 graph 和 mixed batch。先分别跑通，再做 mixed + graph。

---

## 1. 当前基线

### 1.1 已经有的

| 能力 | 状态 |
|---|---|
| `dspark_tree.py`：PCTree 建树、metadata、sparse、accept、commit plan | 已实现，CPU 可测 |
| `dspark_tree_ops.py`：NodePlan、四个树算子、节点页提交、SWA 搬移 | 已实现，直连 `sgl_kernel_npu` |
| topk=1 链式路径 | 保留，`tree_plan=None` 全短路 |
| 四算子调用契约 | 20 个 CPU 测试 |
| RoPE 负位置 clamp | 1 个 CPU 测试 |
| 节点页提交 | 已接入 `move_swa_kv` + `prepare_nodes/commit_nodes` |

### 1.2 还不能算生产完成的

| 缺口 | 现状 | 影响 |
|---|---|---|
| A3 硬件验证 | 未跑 | 不能证明真实 NPU 行为 |
| V2 四算子对 oracle | 未跑 | 不能证明算子数值正确 |
| 非 greedy 树路径 | 未显式拒绝 | 采样请求会静默走 argmax，输出错误 |
| compact 路径 | 树按 non-compact 设计 | `run_compact` 没有接 `tree_plan`，需要显式拒绝或接入 |
| 图捕获 | 显式关闭 decode graph | 性能可能明显落后于链式图模式 |
| mixed batch | `is_extend_in_batch` 被路由到 prefill | 树不会在 mixed batch 里执行 |
| DP token 计数 | 未在 A3 验证 | collective 计数不一致可能挂死 |
| 测试覆盖 | 66 个，低于参考 78 个 | attention/indexer/node_pool 专项覆盖不足 |

### 1.3 当前树路径的主要 host sync

`dspark_tree.py` 里有 18 处 `# HOST-SYNC`，关键集中在：

- `build_tree_metadata`：`num_nodes.max().item()`、`depths.max().item()`、逐请求/逐节点祖先循环；
- `build_tree_ori_sparse_indices`：`int(prefix_lens[req])`、`int(depths[...])`、block table / draft slot 读回；
- `build_tree_accept_outs`：`bool(mask.any())`、按 depth 的 Python 循环；
- `plan_tree_commit`：`num_accepted.max().item()`、逐请求提交计划。

`dspark_tree_ops.py` 里还有：

- `prepare_nodes` / `commit_nodes` / `move_swa_kv` 的 host 描述符和循环；
- indexer / attention metadata 的每次分配；
- `compress_nodes` 的每步 `torch.zeros`。

这些是 P1 的主要工作，也是 P2 入图的硬前提。

---

## 2. 目标与非目标

### 2.1 目标

1. **正确性可证明**：V2 + A3 + fail-loud，不靠“参考分支能跑”推断。
2. **可回滚**：树默认关，失败回 topk=1；图失败回 eager；mixed 失败回 two-pass/禁用。
3. **可测性能**：accept length、每轮延迟、tokens/s、p50/p99 都有数据。
4. **可维护**：算法仍留在 `dspark_tree.py`，设备边界仍留在 `dspark_tree_ops.py`。

### 2.2 非目标

- 不重写 PCTree；builder/scoring/accept 语义保持不变；
- 不改变 topk=1 的链式输出；
- 不恢复 vendor `_C_ascend`、shadow、backend switch；
- 不为了开图而牺牲正确性；
- 不在没有 A3 数据的情况下承诺性能收益。

---

## 3. P0：正确性闸门（最高优先级）

P0 不通过，后面的图、mixed batch 都没有意义。

### 3.1 非 greedy fail-loud

**问题**：树 accept 是 `greedy_tree_accept` + `target_predict = argmax`，但 worker 的树分支没有检查 `sampling_info.is_all_greedy`。非 greedy 请求会静默按 argmax 接受，输出错误。

**修改点**：

- `dspark_worker_v2.py` 的树分支，在 `plan_tree_pre` 之后加：

```python
if (
    (sampling_info is not None and not sampling_info.is_all_greedy)
    or simulate_bonus_sampling_info is not None
):
    raise RuntimeError(
        "DSpark tree supports greedy decoding only; "
        "disable sampling or use topk=1."
    )
```

**验收**：

- greedy 请求：树正常跑；
- 非 greedy / simulated bonus sampling 请求：明确报错，不产生错误 token；
- topk=1 + 非 greedy：仍然正常走链式采样。

**后续可选**：实现 tree rejection sampling（参考本地 vllm 的 `rejection_sampler.py` / `greedy_reject.py`）。第一阶段先 fail-loud。

### 3.2 compact 路径显式拒绝或接入

**问题**：`run_compact` 分支没有传 `tree_plan`；树目前按 non-compact 设计。

**修改点**：

- 在 `dspark_worker_v2.py` 入口加：

```python
if tree_pre is not None and run_compact:
    raise RuntimeError("DSpark tree requires non-compact verify")
```

或者后续把 `tree_plan` 接入 `run_compact`。P0 先拒绝，避免静默错路径。

注意：`resolve_tree_config` 已经拒绝非 `static` ragged，所以 compact 在当前配置下通常不可达；这里的显式断言是 defense-in-depth，防止未来接入 compact 或绕过配置。

### 3.3 V2：四算子对 CPU oracle

**目标**：四个 `sgl_kernel_npu.tree` 算子与 `tree_reference.py`（或等价 CPU oracle）一致。

覆盖：

1. `compressor_tree_project`
2. `compressor_tree`
3. `quant_lightning_indexer_tree`
4. `sparse_attn_sharedkv_tree`

验收：

- 小 shape、边界 shape、多请求、ratio 4/128；
- 数值误差在约定 tolerance 内；
- 失败用例必须定位到具体算子，不允许“整体跑通”代替。

### 3.4 A3 关卡

| 关卡 | 内容 | 通过标准 |
|---|---|---|
| (a) | 关树，topk=1 | 输出/accept length 与 main 无回归 |
| (b) | 开树，topk=2，budget≥depth，static ragged，eager | 端到端无 assert/index error |
| (c) | greedy 短上下文，树 vs 链 | token 序列一致 |
| (d) | 同 workload 比较 accept length / 每轮延迟 | 有数据，能判断是否划算 |
| (e) | TP16/DP16 重复 (a)-(c) | 部署拓扑正确 |

### 3.5 DP token 计数

- 断言 `num_tokens_per_req == local_bs * (1 + budget)`；
- 覆盖不同 local batch 的 DP rank；
- 确认 MoE collective 不挂死；
- 确认 idle rank 仍参与必要 collective。

### 3.6 测试覆盖补齐

当前 66 个测试低于参考 78 个。建议迁移/补齐：

- attention metadata / tree attention 契约；
- indexer select-all / 短历史；
- node pool / node commit；
- worker 非 greedy gate；
- compact 拒绝；
- topk=1 回归。

P0 退出标准：

```text
非 greedy fail-loud 生效
compact 明确拒绝/接入
V2 四算子通过
A3 (a)-(e) 通过
DP token 计数通过
覆盖参考分支的关键专项（attention/indexer/node pool/topk=1），不要求文件结构相同
```

---

## 4. P1：capture-safe 重构

目标：eager 行为不变，但所有树规划函数不再依赖 host readback 和动态分配。

### 4.1 `TreeBuilder` 预分配

- 新增 `prepare_for_capture(num_reqs, vocab, depth, device)`；
- 预分配 `tokens/depths/parents/path_scores/frontier/node_arange/selection_buf`；
- `build()` 只写 buffer，不再 `torch.full/zeros/arange`；
- 提供 `workspace_tensors()` 给 graph 记账；
- 参考本地 vllm 的 `tree/builder.py`。

### 4.2 metadata / sparse / accept / commit 设备化

| 函数 | 去什么 | 怎么去 |
|---|---|---|
| `build_tree_metadata` | `.max().item()`、逐节点祖先循环 | 固定 `query_width=1+budget`；ancestors 用 gather/scatter |
| `build_tree_ori_sparse_indices` | `int(prefix_lens[req])`、block table host 读 | 设备 gather；host 侧只做 pinned H2D |
| `greedy_tree_accept` | 数据相关 host 分支 | 保持静态 budget 循环，内部全 tensor |
| `build_tree_accept_outs` | `bool(mask.any())`、depth Python 循环 | 固定列向量化，`scatter_` 写 bonus |
| `plan_tree_commit` | `num_accepted.max().item()`、逐请求循环 | `max_rows=1+budget` 静态；gather/scatter |

### 4.3 `dspark_tree_ops.py` 静态化

- `compress_nodes` 的 `nodes` 预分配；
- indexer/attention 的 1024 metadata 预分配；
- `prepare_nodes/commit_nodes` 的 descriptor 设备化，或至少提前到 capture 外；
- `move_swa_kv` 的 host 循环设备化；
- port 本地 vllm 的 `node_commit.py` / `freeze_node_pages` 优化。

### 4.4 backend 静态化

- `ascend_dsv4_backend.py` 的 `tree_node_kv[layer_id]` 等 dict 在 init 按层预建；
- `_op_supports_kwarg` 的 schema 反射只在 init 做；
- 确认 `_forward_tree_indexer/_swa/_attention` 内没有动态 shape / host 分支。

P1 退出标准：

```text
eager 输出与重构前一致
capture 区内无 .item()/.tolist()/.cpu()/动态分配
profiler 看不到树路径 host sync
```

---

## 5. P2：入图

### 5.1 边界选择

按依赖拆，不要一上来 capture 整个 decode step：

```text
draft graph（已有链式能力）
  → target verify graph（先做 MVP）
  → tree build graph（P1 后）
  → accept/commit graph（最后）
```

### 5.2 最小可行：只开 target verify 图

> 决策门：先看 A3 (d)。如果 eager 树的 accept length 增益已经覆盖 verify 成本，图捕获是可选优化，不阻塞 P3；如果 (d) 为负，才把 target verify 图作为性能兜底。不要为了开图而开图。

- `plan_tree_pre` 仍 eager；
- 把 `positions / seqused / ori_sparse_indices / cmp_sparse_indices / node_kv` 拷进静态 graph 输入；
- capture 目标模型的 tree verify forward；
- `plan_tree_post` 仍 eager；
- 失败回 eager。

这是收益最大、风险最小的一步。

### 5.3 tree build 图

P1 完成后：

- 输入 `base_logits`、root ids、prefix/block table 静态 buffer；
- 输出 `TreePrePlan` 的设备 tensor；
- 按 `num_reqs` 做 gear；
- 参考本地 vllm `full_graph.py` / `prepare_for_capture`。

### 5.4 accept/commit 图

- 设备侧 `greedy_tree_accept`；
- `build_tree_accept_outs`；
- `move_swa_kv` + `prepare_nodes/commit_nodes` + `commit_hidden`；
- 参考 `full_graph.commit`、`graph_commit.py`、`node_commit.py`。

### 5.5 graph 管理

- 新增 `dspark_tree_graph.py`；
- `capture(num_reqs)` / `run()` / fallback；
- 复用 sglang `NPUGraphRunner` / `DecodeCudaGraphRunner`；
- `speculative_hook.py` 增加 `--speculative-dspark-tree-graph`；
- 默认可以先 off，A3 验证后再 on。

P2 退出标准：

```text
eager vs graph：token / accept length / commit_lens 一致
覆盖不同 batch size gear
capture 区无 host sync
topk=1 不受影响
```

---

## 6. P3：mixed batch

### 6.1 当前行为

`dspark_worker_v2._forward`：

```python
if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
    return self._forward_prefill(...)
```

所以 `is_extend_in_batch` 的 mixed batch 不会进 `_forward_decode`，树不会执行。

### 6.2 two-pass（先做）

```text
mixed batch
  ├─ prefill rows -> _forward_prefill
  └─ decode rows  -> _forward_decode + tree
合并两个 GenerationBatchResult
```

优点：改动小，正确性容易验证；缺点：两次 forward，少了 overlap。

### 6.3 single-forward（目标形态）

参考本地 vllm `mixed_batch.py`：

1. 构造 phase view：
   - decode request indices / token indices；
   - prefill request indices / token indices；
   - query_start_loc；
   - omitted_tokens（prefill tail / placeholder draft）。
2. 只对 decode rows 做 tree：
   - `base_logits`、root ids、prefix、block table、draft slots 按 decode indices 切片；
   - tree batch = decode bs，不是全局 bs。
3. 一次 target forward：
   - decode rows 用 tree metadata；
   - prefill rows 用普通 metadata；
   - 合并 attention metadata / positions / slot mapping。
4. 只提交 decode rows：
   - `plan_tree_post` 只处理 decode 子 batch；
   - prefill 输出单独合并。
5. DP/TP：
   - `num_tokens_per_req = prefill_tokens + decode_bs * (1+budget)`；
   - collective 看完整 mixed token 数；
   - idle rank 仍参与。

### 6.4 mixed + graph（最后）

- mixed 需要独立 gear；
- 不要把纯 decode graph 直接复用到 mixed；
- 先 two-pass eager，再 single-forward eager，最后 mixed + graph。

P3 退出标准：

```text
mixed 输出 == 先 prefill 再 decode 的串行输出
prefill-only / decode-only / mixed 三种 case
DP16/TP16 token 计数一致
```

---

## 7. P4：性能与可观测性

### 7.1 必测指标

| 指标 | 说明 |
|---|---|
| accept length | 每轮接受 token 数（含 bonus） |
| 每轮延迟 | draft / tree pre / target / accept / commit 分段 |
| tokens/s | 端到端吞吐 |
| p50 / p99 | 尾延迟 |
| verify rows | `1+budget` 的实际成本 |
| host sync 数 | 每步树路径的同步点 |
| graph hit rate | 图 replay 覆盖率 |

### 7.2 调参

- budget：不是越大越好，verify 成本会上升；
- topk：影响 frontier 宽度和 Markov batch；
- depth：决定最大接受长度；
- 可选：`depth_penalty`、`keep_origin_chain`（当前未实现，参考 vllm 分支）；
- 混合策略：高置信度走链，低置信度走树。

### 7.3 验收

- 同 workload 下 tree vs topk=1 的 tokens/s、p50/p99；
- 只有 accept length 增益覆盖额外成本，才默认开树；
- 否则保持树默认关，作为可选功能。

---

## 8. 文件改动矩阵

| 文件 | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| `dspark_tree.py` | 非 greedy/compact 辅助 | 预分配、设备化 metadata/sparse/accept/commit | tree build graph 输入输出 | mixed 子 batch 支持 |
| `dspark_tree_ops.py` | V2 对接 | 静态 buffer、node commit、SWA move | accept/commit graph | mixed node plan/commit |
| `dspark_worker_v2.py` | fail-loud gate | graph replay 分支 | target/tree/commit graph 接线 | mixed phase 路由与合并 |
| `dspark_verify.py` | compact/tree 校验 | 静态 tree commit IO | verify graph 输出对接 | 只提交 decode 行 |
| `ascend_dsv4_backend.py` | 算子能力检查 | capture-safe 路由 | graph metadata 静态化 | mixed metadata 视图 |
| `speculative_hook.py` | 门禁 | graph flag | graph 开关/回退 | mixed 开关/门禁 |
| **新增** `dspark_tree_graph.py` | — | — | graph capture/replay | mixed gear |
| **新增** `dspark_tree_mixed.py` | — | — | — | phase view |
| 测试 | 非 greedy/compact/V2 | host-sync 检查 | graph parity | mixed parity |

---

## 9. 验证矩阵

| 场景 | 证据 |
|---|---|
| topk=1 无回归 | A3 (a) 输出/accept length 对比 |
| 树 eager 能跑 | A3 (b) 端到端日志 |
| greedy 树/链一致 | A3 (c) token 序列对比 |
| 树是否划算 | A3 (d) accept length + 每轮延迟 |
| TP16/DP16 | A3 (e) |
| 非 greedy 拒绝 | 请求非 greedy，进程报错且不输出错误 token |
| compact 拒绝 | compact 配置下树模式明确报错 |
| V2 | 四算子 vs oracle 数值报告 |
| graph parity | eager vs graph token/accept/commit 对比 |
| mixed parity | mixed vs 串行 prefill+decode 对比 |
| 性能 | tokens/s、p50/p99、分段延迟 |

---

## 10. 风险与回滚

| 风险 | 回滚/缓解 |
|---|---|
| 非 greedy 输出错误 | P0 先 fail-loud；后续实现 tree sampling |
| 四算子数值不一致 | V2 不通过不开树 |
| graph capture 失败 | 自动回 eager；flag 默认 off |
| mixed 计数错 | 回 two-pass；再不行禁用 mixed |
| node commit 错 | 回 topk=1；保留节点页开关 |
| 性能不划算 | 树默认关；按 workload 开 |
| DP collective 挂死 | 断言 token 计数；A3 DP16 必跑 |

**总回滚开关**：

```text
树关闭 -> 完全走 topk=1 链式路径
```

这是 v5 现成的安全网，必须保留。

---

## 11. 里程碑

| 里程碑 | 内容 | 退出标准 |
|---|---|---|
| M0 | P0 正确性 | 非 greedy/compact/V2/A3/DP/测试全过 |
| M1 | target verify 图（A3 (d) 为负时优先） | graph parity + 延迟收益 |
| M2 | 完整 tree graph | tree build + accept/commit 图 |
| M3 | mixed two-pass | mixed == 串行 |
| M4 | mixed single-forward | 单次 mixed forward |
| M5 | mixed + graph | 独立 gear，parity |
| M6 | 性能签收 | accept length / tokens/s / p99 达标 |

---

## 12. 参考实现映射

| 本路线 | 本地 vllm-ascend 参考 |
|---|---|
| capture-safe builder | `tree/builder.py` 的 `prepare_for_capture` / `_ensure_scratch` |
| target/tree/commit graph | `tree/full_graph.py`、`ops/triton/spec_decode/tree/graph_commit.py` |
| mixed batch phase view | `tree/mixed_batch.py` |
| 设备 greedy accept | `tree/rejection_sampler.py`、`greedy_reject.py` |
| node commit | `tree/node_pool.py`、`node_commit.py` |
| KV transport | `tree/chain_pack.py`、`kv_binding.py`、`kv_layout.py`、`kv_project.py` |
| 指标 | `tree/metrics.py` |

注意：本地 vllm 的 runtime、C++ 算子、KV 传输与 sglang 不同，**只移植语义，不复制代码**。

---

## 13. 待确认问题

1. 非 greedy：P0 只 fail-loud，还是本期就实现 tree sampling？
2. `keep_origin_chain` / `depth_penalty` 是否要纳入 v5？（当前没有）
3. A3 的 accept length 收益是否足以覆盖 verify 成本？
4. c4 ring 扩宽的显存代价是否可接受？
5. mixed batch 是否要求 single-forward，还是 two-pass 即可？
6. 图捕获优先 target verify 还是等完整 tree graph？
7. SWA history window 的 `start = prefix + depth - window` 是否应为 `prefix + depth - window + 1`？文档 §6.3 的 `[9]` 与当前代码一致，但需按 DSV4 SWA 约定再核对一次，避免把 off-by-one 带进图/混合 batch。

---

## 14. 结论

当前 v5 的正确性基础已经有了，但生产化还差三块：

1. **证明**：V2 + A3 + fail-loud；
2. **入图**：先 target verify，再 tree build，最后 accept/commit；
3. **mixed batch**：先 two-pass，再 single-forward。

推荐顺序是：

```text
P0 正确性
  → P1 capture-safe
  → P2 target verify 图
  → P2 完整 tree 图
  → P3 two-pass mixed
  → P3 single-forward mixed
  → P4 性能签收
```

在 P0/P1/P2 完成之前，不要把“树比 topk=1 快”当作既定事实；在 A3 (d) 跑出 accept length 和每轮延迟之前，也不要把树设为默认路径。
