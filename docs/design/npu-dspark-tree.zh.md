# DSV4 + DSpark 树形投机解码（topk > 1）Ascend NPU 架构

状态：**v5.2 已实现（CPU 66/66 通过；A3 关卡待执行）。分支 `feat/npu-dsv4-dspark-tree-v5`，已推送到 `unclezhou`。**

| 项 | 决定 |
|---|---|
| 架构 | **C**：后端中立的 torch 树核心 + NPU 四算子适配器；提交使用已验证的节点页路径（accept 后链式复算推迟到 A1 通过后） |
| 基线 | `dspark_topk/sglang` 本地克隆，`main = 5d4a6f499e`（2026-10-30） |
| 参考（移植来源） | `dspark_topk/sglang-unclezhou/sglang`，`origin/feat/npu-dsv4-dspark-topk = b7c7fa0a`，**用户已在 TP16 + DP16 上机验证**（`[hw]`，用户报告，本文未复验） |
| 交叉验证（仅设计证据） | `dspark_topk/vllm-ascend`，`tree261025-rebase`，commit `8947f1f9f`、`b7cd632f2`、`612b84ad6`、`d6048af3f`、`2fe76a173`，**不移植**，不取 vendor 算子 |
| 算子 | `dspark_topk/sgl-kernel-npu`，`feat/npu-dsv4-tree-triton = 905c136`，四个算子在 `python/sgl_kernel_npu/sgl_kernel_npu/tree/` |
| 目标分支 | `feat/npu-dsv4-dspark-tree-v5`，从本地 `main = 5d4a6f499e`，已推送到 `unclezhou` |
| 范围 | 完整生产级树（任意 topk / depth / budget），不是固定配置 MVP |
| 新增文件 | 恰好两个：`dspark_tree.py`、`dspark_tree_ops.py`，放在已有 `dspark_components/` 下 |
| 禁止项 | `tree/` 文件夹、SGlang 副本、第二个 DSpark worker、vendor 适配器、shadow 模式、`TreeOpBackend`、`SGLANG_DSPARK_TREE_BACKEND`、改 kernel |

## 0.4 实现状态（v5.1，静态）

| 项 | 值 |
|---|---|
| 分支 | `feat/npu-dsv4-dspark-tree-v5`，基线 `main = 5d4a6f499e` |
| Commit | `9abd9d74d0` 实现；`a8395c90b9` 扩展 CPU 测试；`d98cf6d2c5` harness 修复；`6a7d2a2b4d` 死代码/重复定义裁剪 |
| 远端 | `unclezhou/feat/npu-dsv4-dspark-tree-v5`（已推送） |
| Diff | 17 文件，**+3,832/−84**；裁剪 commit 净 −456 |
| 新模块 | `dspark_tree.py` 1,134 行；`dspark_tree_ops.py` 835 行 |
| 测试 | **66 个 CPU 测试**（45 core + 20 op + 1 rope），参考为 78；删掉的 12 个只覆盖参考专用 helper |
| 提交路径 | 已验证的 **node-page freeze/commit + SWA move**，不是 accept 后复算 |
| 对齐 | 参考实现所有非脚手架 tree 名称均已移植 |
| 丢弃 | vendor `_C_ascend`、shadow、`SGLANG_DSPARK_TREE_BACKEND`、`tree/` 包、动态 `ForwardBatch` 属性 |
| 改进 | node KV/scale 挂在 `tree_plan` 上；`sliding_window`→`window_size` 回退；grammar/watermark 直接报错；commit plan 只保留一份 |
| 实际规模 | 生产约 2,790，测试约 1,042（仍高于精简估算 1,950-2,150，因为保留已验证的节点页提交和完整集成） |
| 已验证 | 静态 + CPU：**66/66 tree CPU 测试通过**（torch 2.9.1+cpu，Python 3.14 venv）；`py_compile`/`compileall`；AST import/签名检查；禁用 token grep |
| 未验证 | A3 eager/graph、TP16/DP16、accept-length 比值 |

为什么规模变大：为保证与已验证分支的功能等价，完整移植了 tree 核心而不是裁剪。accept 后链式复算仍只是 A1（提交等价）在硬件上通过后的优化；当前提交路径是参考实现已验证的节点页路径。

验证命令（需在有 torch 的机器或 A3 上执行）：

```bash
python3 -m pytest -q \
  test/registered/unit/spec/test_dspark_tree_core.py \
  test/registered/unit/spec/test_dspark_tree_ops.py \
  test/registered/unit/spec/test_dspark_tree_rope.py
```

A3 关卡顺序：(a) 关树/topk=1 基线；(b) 开树、topk=2、budget>=depth、static ragged；(c) greedy 短上下文 token 一致；(d) accept length vs verify 成本；(e) TP16/DP16 重复 (a)-(c)；图捕获仅在 (d) 为负时再做。

---

## 0.5 决策（v5）：架构 C

**C = 后端中立的 torch 树核心 + 四个 `sgl_kernel_npu` 树算子适配器 + 参考实现已验证的节点页提交（accept 后复算推迟到 A1）。**

为什么不是 A（移植参考实现的树层）：它在测试之前就有约 3,619 行生产代码，保留一个 19 文件的
`tree/` 包、vendor/shadow/backend-switch 脚手架，以及一条定制的节点页 freeze/commit 路径。这些都是
约束禁止、或被用户以「太大」为由否决的东西。

为什么不是 B（复用 NPU 原生 EAGLE 的 `build_tree_kernel_efficient` + `verify_tree_greedy`）：**被
kernel 语义证伪，不是口味问题**：

- `build_tree_kernel.cpp:151,181` 用 `selected_index // topk` 推导 parent；它是**均匀
  topk 叉、parent-major** 的候选空间。
- `build_tree_kernel.cpp:163-165` 会静默跳过 parent 未被选中的节点：**没有 ancestor closure**；
  孤儿节点消耗 budget 但不可达。
- 参考 DSpark builder 是 **ancestor-closed + 每层全局 top-`width` + 重新编号**
  （`tree/builder.py:47-49,99-129`）。这些是不同的树。
- `verify_tree_greedy.py:51` 把 `candidates[i]` 与 `target_predict[i-1]` 比较，**从不使用**
  `retrive_next_token`/`retrive_next_sibling`：Triton accept kernel 只支持链式。native 变体使用
  另一套按位置索引的约定。
- 因此复用 EAGLE 需要一个和参考 builder 一样大的适配器，还会产出一棵不同的树，并带来更高的
  静默 orphan/accept 错误风险。

为什么不是「verify 时跳过当前块的压缩 KV」：**不安全**。树注意力通过 `tree_ancestors` 到 node pool
来解析压缩槽 `k >= prefix // ratio`（`tree_ops_semantics.md`，`tree_reference.py:539-557`）。当
`prefix % ratio != 0` 时，边界残块必须把 prefix 之前的 token（paged state）与 draft 节点（node pool）
聚合起来；丢掉它会静默丢失一个 key。四个树算子全部保留。

**提交策略（已实现）：** 参考实现已验证的节点页路径 —— accept → `move_swa_kv` 搬移被接受的 full/SWA 行 → verify 期间记录压缩节点行 → `prepare_nodes`/`commit_nodes`（节点页 freeze/commit）。本实现**没有**启用 accept 后链式复算；它只在 A1（提交等价）于硬件上通过后作为优化保留。

**两个证伪实验原计划作为 P1 关卡，但本机无法运行**（无 torch/NPU）。因此实现保留参考实现已验证的提交路径；A1/A2 仍必须在硬件上执行：

| # | 假设 | 证伪方法 |
|---|---|---|
| A1 | accept 后的链式 `forward_compress` 能在已接受路径上复现参考实现的节点页 commit | 两步 CPU/数值 harness：链式再压缩 vs `node_commit.freeze/commit`，对比压缩 KV 页与 `state_cache` |
| A2 | 四个 `sgl_kernel_npu.tree` 算子在容差内通过独立的 `tree_reference.py` oracle | 在 A3 上运行 Repo C 的 `test_tree_*` + oracle |

A2 必须在 A3 上通过后才能签收。A1 只决定可选的 accept 后复算，不再改变当前架构（当前已使用参考的节点页提交）。

计划估算 **`[static estimate]`**：生产约 1,950-2,150；测试约 900-1,100（16 条不变量套件），若移植全部 78 个参考 CPU 测试则约 1,790。实际实现规模见 §0.4（生产约 2,790，测试约 1,042）；剩余差距来自已验证的节点页提交和强制的 backend 集成。

---

## 0. 环境现实（规划 Gate 前必读）

本工作站只用于分析和编辑 `[static]`：

- `python3` 3.14.8，`pip list` 为空，没有 `torch`，没有 `torch_npu`，没有 `sgl_kernel_npu`。
- 没有 `/dev/davinci*`。
- 远端 `unclezhou` 可达；`feat/npu-dsv4-dspark-topk` 在 `b7c7fa0a`（与交接一致）。

结论：这里产不出任何 `[hw]` 证据，连 `[cpu]` 单测都要先装 CPU 版 `torch`。本文所有结论默认
`[static]`，除非另有标注。

---

## 1. `main` 中已存在、且原样复用的部分

`[static]` `main` 包含完整的 DSpark **链式（topk == 1）**路径，且没有 `tree/` 目录。

原样复用，不改：

| 组件 | 证据（main） |
|---|---|
| `SpeculativeAlgorithm.DSPARK` → `DSparkWorkerV2` | `python/sglang/srt/speculative/spec_info.py:335-340` |
| Worker 编排 `forward_batch_generation` / `_forward_decode` | `dspark_components/dspark_worker_v2.py:758, 972` |
| Draft 提议（`DraftBlockProposer`、markov block head、folded/non-folded） | `dspark_components/dspark_draft.py:448, 258-303, 487-524` |
| Target verify 执行器（eager + compact/static + graph epilogue） | `dspark_components/dspark_verify.py:302, 468, 179, 380` |
| Verify 窗口分配 / 预算 / 调度 planner | `dspark_components/dspark_planner.py:800, 376, 326` |
| `DFlashDraftInputV2` / `DFlashVerifyInput` / `DFlashVerifyInput.prepare_for_verify` | `dflash_info_v2.py:36-58`，`dflash_info.py` |
| 链式 accept / output / commit | `dspark_verify.py:1103, 1296`，`dspark_kv_inject.py:142, 359`，`dspark_deferred_commit.py:134` |
| DSV4 attention backend，全部链式路径 | `hardware_backend/npu/attention/ascend_dsv4_backend.py` |
| Grammar 树（本身已是树形，`from_device(next_token, next_sibling, ...)`） | `spec_utils.py:818-857` |
| ragged verify 模式、KV pool / cache loc、DP/TP 管道 | `ragged_verify.py`、memory pool、planner |

`dspark_planner.py` **完全不动**：`VerifyWindow` 已经携带 `positions_2d [bs, W]`、
`verify_cache_loc [bs*W]`、`verify_cache_loc_2d [bs, W]`（`dspark_planner.py:59-66`）。
树只是把自己的 per-node position 写进这个窗口，并消费同一批 slot。
唯一要求是 `verify_num_draft_tokens = 1 + budget`（窗口宽度），这样 uniform 分配继续成立。这是
「复用框架」的最大收益。

`dspark_draft.py` **也不动**：DSpark draft 仍然是链式块，产出 `base_logits [bs, depth, vocab]`；
树分支随后由 markov-head scorer 决定。树模式里**没有第二次 draft forward**（见 §4）。

---

## 2. 为什么只补 mask / sparse-index 补丁不够

`[static]` DSV4 的 decode 状态是混合的：SWA（原始 KV）+ 压缩 KV 池（ratio 4 / 128）+ lightning
indexer。因此一次树 verify 需要每个节点具备：

1. 拓扑（`parent`、`ancestors`、`depth`）：链式路径里都不存在；
2. 每个节点的 target hidden 行，这样该节点的 KV 才能提交进 draft cache
   （`TargetHiddenKvInjector`/node commit）；
3. 每节点绝对位置和 `seqused_kv = prefix + depth + 1`（链式用 `prefix + arange`）；
4. 每节点压缩 KV 投影（`compressor_tree_project` + `compressor_tree`），因为压缩边界落在树路径上，
   而不是一个连续块；
5. 每节点 indexer top-k（`quant_lightning_indexer_tree`），历史是 ancestor-aware 的；
6. 每节点 ancestor 注意力 mask（`sparse_attn_sharedkv_tree`）；
7. 一条被接受的**节点路径**，而不是标量 `correct_len` 前缀。

另外还要移除 `main` 里三处链式专用门禁：
`speculative_hook.py:760-792`（强制 `num_steps=1`、`speculative_eagle_topk=1`）和
`dspark_worker_v2.py:1496`（`assert speculative_eagle_topk in (None, 1)`）。

---

## 3. 目标每步数据流（树 vs 链）

树的差别在**target verify 看到什么**，不在 forward 跑几次。

| 步骤 | 链（今天） | 树（目标设计） |
|---|---|---|
| draft 提议 | 1 次 draft forward，覆盖 `bs*(gamma 或 gamma+1)` 行 | 相同，不改 |
| 规划 | 无 | `plan_tree_pre(base_logits, ...)` → 拓扑 + 元数据 + 稀疏槽 + 节点计划 |
| verify ids | `[anchor, d1..dγ]`，`bs*(gamma+1)` 行 | `[root, n1..nN]`，`bs*(1+budget)` 行 |
| target forward | 1 次 | 1 次，相同，行集更宽 |
| compressor | 连续块边界 | 逐节点 project/compress（树算子） |
| indexer | 每 query 链式 top-k | 每节点 ancestor-aware top-k（树算子） |
| attention | 块内 causal | ancestor mask（树算子） |
| accept | 标量 `correct_len` 前缀 | `greedy_tree_accept` → 节点路径 + bonus |
| commit | 前缀 `col < commit_lens` | 被接受节点行 → `dest_slots` / `dest_positions`，然后做 accept 后的压缩 KV/state 再压缩 |
| 每步 forward 次数 | 2 | **2** |

`_forward_decode` 中的树专用调用（来自参考实现，需在 `main` 上重新推导）：

| 阶段 | 参考锚点 |
|---|---|
| 树配置解析（一次，init） | `dspark_worker_v2.py:291` |
| verify 窗口分配（`main` 已有） | `dspark_worker_v2.py:1068` |
| `plan_tree_pre` + scorer | `:1156-1183` |
| `run_non_compact(..., tree_plan=...)` | `:1239`；`dspark_verify.py:305-321` |
| target forward | `dspark_verify.py:368` |
| `plan_tree_post`（accept/output/commit 计划） | `:1258` |
| `move_swa_kv` | `:1315` |
| `prepare_nodes` / `commit_nodes` | `:1318-1326` |
| `commit_hidden(accept_index, tree_positions, tree_commit)` | `:1411-1418`；`dspark_verify.py:408-465` |

---

## 4. 树节点 KV 与「没有额外 draft forward」的性质

`[static]` 在 `main` 中，DSpark 从 `compute_base_logits(fwd.raw_hidden)`（`dspark_draft.py:487-490`）
和 markov head 循环（`:285-299`）产出的 `base_logits [bs, gamma, vocab]` 起草。树精确复用这些
per-depth logits：

- 每层候选选择用 `DSparkMarkovScorer`（`head.step_bias(frontier_tokens, ...)` + top-k）；
- builder 逐层维护 ancestor-closed 集合；
- 每个节点的 **target** hidden 来自 target verify（`CaptureHiddenMode.FULL`，
  `dspark_verify.py:319`），这正是 accept/commit 路径写进 draft KV 的内容。

所以树模式**不需要树版 draft attention，也不需要第二次 draft forward**。这就是为什么 delta 能留在
verify/commit 路径里，而不必重构 draft worker。

节点 KV 管线（四个算子中的三个，直接调用，无适配器）：

```
compressor_tree_project(x, wkv, wgate, nodes, ape, positions, cu)   # 原地操作，nodes 预分配
        -> nodes fp32 [T+1, 1, 2*c*d]  （第 0 行保留）
compressor_tree(state, nodes, norm, cos, sin, table, prefix,
                ancestors, depths, requests, ratio, rope_dim, epsilon)
        -> node_kv bf16 [T, d]      # (prefix+depth+1) % ratio != 0 时为零行
quantize(node_kv) -> int8 [T,128] + fp16 scale    # 框架侧 cast，不是算子
quant_lightning_indexer_tree(query, history, weights, query_scale, history_scale,
                             nodes_i8, node_scale, ancestors, prefixes, requests,
                             table, cu, lengths, residual, metadata, topk)
        -> cmp_sparse_indices int32 [T, 1, topk]
sparse_attn_sharedkv_tree(q, ori_kv=..., cmp_kv=..., cmp_tree_kv=node_kv,
                          tree_ancestors=..., tree_prefix=..., seqused_kv=...,
                          cmp_sparse_indices=..., ori_sparse_indices=..., ...)
        -> out, softmax_lse
```

塑造了设计的算子契约事实 `[static]`（审计自 `sgl-kernel-npu`）：

- `compressor_tree_project` 是唯一原地算子；`nodes` 必须预分配。
- 另外三个自己分配输出；调用方只预分配输入/buffer。
- `metadata`（int32[1024]）在 indexer 和 attention 中都是**先校验后忽略** → 传一块静态零 buffer
  即可，不需要 vendor metadata 算子，也不需要 `_C_ascend` 的 `.so`。
- `cu` 在 project/indexer 中被校验必须等于 `arange(n+1)` → 它只是断言「一 query 一节点」规则。
- attention 要求 `seqused_kv = prefix + depth + 1`；`depths` 只有 compressor 使用。
- `ori_sparse_indices` 宽度必须是 128 的正整数倍；`topk ∈ [1,2048]`；`ratio ∈ {4,128}`。
- **kernel 侧**的图捕获阻塞点（在 `sgl_kernel_npu` 内，不是我们能修的）：indexer 会做
  `lengths.max().item()` 并据此确定 scratch 大小，还有数据相关的「全选」分支；project 有两处
  device-sync 校验。结论：被捕获的 **verify forward** 仍是 P6 关卡，v1 跑 eager。但这**不能**成为
  框架侧的借口：plan/accept/commit 必须从 P2 起就按可捕获的形状写（§4.5），否则 P6 会变成重写。

---

## 4.5 从构造上即可被图捕获（R1）：P2 的约束，不是 P6 的任务

vllm-ascend 分支（DSV4 `topk>1`）**强制要求** `cudagraph_mode == FULL`
（`worker/v2/spec_decode/tree/speculator.py:120-123`），并且在图捕获可行之前不得不重写树构建和节点
提交（`612b84ad6`、`2fe76a173`）。给我们的教训是：在低延迟负载上，eager 的 target verify 很可能
吃掉 accept-length 的收益，所以树路径必须从设计上就是 graph-shaped。

从 P2 起约束 S2/S5/S6 与 P1/P5 的规则：

1. **固定容量。** 每请求 `1 + budget` 行在 `__init__` 预分配；没有任何东西按数据定尺寸。
   `budget`、`depth`、`topk` 在进程生命周期内是常量。
2. **静态 k 选择。** 用 `torch.topk(k=常量)`，绝不对变长前沿做 `argsort`，也绝不用 host 派生索引。
   代价是逐层做，`topk ≤ budget`。
3. **每步路径不做 host 读回。** 不出现 `.item()`、`.tolist()`、`.cpu()`，也不基于 plan 张量做
   `data_ptr` 值分支。host 只通过 pinned staging buffer 提供每步 prefix（H2D），与 vllm 的
   `PrefixStaging` 模式完全一致。
4. **捕获区内不分配。** plan/index/visibility/link buffer 全部持久；输出用 `copy_` / `index_copy_` /
   `scatter_` / `fill_` 写进预分配张量。`612b84ad6` 出问题正是因为 launch 时有一个
   `.contiguous().to(torch.long)` 临时量。
5. **行数恒定。** 树永远输出 `1 + budget` 行；填充行显式失效（§8）。不做 per-request 截断，
   不用 compact ragged 布局。
6. **混合 prefill+decode batch** 无法重放固定的树 gear → 树 verify 对该 batch 要么 eager，要么拒绝
   （fail loudly）。vllm 采用前者（`aclgraph_utils.py:245-255`）。

收益：plan/accept/commit 层不需要新 kernel 就能被图捕获；只剩四个 `sgl_kernel_npu` 算子
（indexer 的 `.item()`/全选、project 的校验同步）仍是 P6 阻塞点。

---

## 5. 模块布局与确切 delta

### 5.1 新文件 1：`dspark_components/dspark_tree.py`（纯 host + 通用 torch）

不 import `torch_npu`，不 import `sgl_kernel_npu`，不加载 `.so`。分节：

| 节 | 内容 |
|---|---|
| S1 config | `TreeConfig`、`TreeShape`、宽度解析（`gamma` = depth，`verify_num_draft_tokens` = 1+budget）与校验 |
| S2 scoring/builder | `DSparkMarkovScorer`、逐层 ancestor-closed `TreeBuilder`、容量/填充规则 |
| S3 metadata | `prefix`、`depth`、`ancestors`（root-first、`-1` 填充）、`seqused_kv`、per-node `positions`、`cu_seqlens`、`positions % ratio` 余数 |
| S4 sparse slots | SWA 窗口槽 + ancestor 节点槽 + `full_to_swa` 映射 |
| S5 accept/output | `greedy_tree_accept`（first_child/next_sibling 走路径）+ **四值 accept 契约**（R4）：`accept_index`（被接受的节点行）、`num_sampled`（**包含 bonus token**）、`num_accepted`（**只算 draft 节点**）、`endpoint_node`（root-only 为 0）。加上 `build_tree_accept_outs` → `out_tokens`、`commit_lens`、`new_seq_lens`；`path_node_ids`/`accept_index` 未用后缀为 `-1` |
| S6 plan | `TreePrePlan`、`TreePostPlan`、`plan_tree_pre`、`plan_tree_post`、`plan_tree_commit`（目标 slot/position）+ **SWA 行搬移计划**；压缩 KV/state 由已验证的节点页 freeze/commit 写入；accept 后复算未启用 |

**提交策略（已实现，参考实现的节点页路径）：**

```
accept（root + 被接受路径）
  -> move_swa_kv 搬移被接受的 full/SWA 行
  -> verify 期间 record 压缩节点行
  -> prepare_nodes / commit_nodes（节点页 freeze/commit）
  -> compact hidden/positions
```

为什么保留 SWA 搬移：树节点写在 verify 窗口的列上；下一步读取的位置是
`prefix..prefix+commit_len-1`。不做搬移，这些位置会指向其他节点的列。

为什么使用节点页提交：参考实现已在 TP16/DP16 上验证该路径；四个树算子对持久化压缩 KV/state 是只读的，accept 后由 `prepare_nodes`/`commit_nodes` 写入。accept 后链式复算作为 A1 通过后的优化保留。

必须测试的风险（见 §13）：state-ring 前缀、边界残块（`prefix % ratio != 0`）、c4 loc 仅在边界行等于
`full_loc // 4`、c128 loc 为 verify 区间的连续前缀切片、再压缩与 SWA 搬移的顺序、按 `accept_index`
gather hidden、FP8 state、bonus-token state。

SWA 行搬移不变量（保留自 vllm KV-transport 工作，`b7cd632f2`）：

1. **按 (block-table group, capacity) 只规划一次，跨层复用。** 只排索引；payload 永不重排。
2. **两阶段：先把所有源 gather 到 scratch，再 scatter。** 绝不对 alias 行边读边写。
3. **重叠源：first-writer-wins。**
4. **存在性判断用 `path >= 0`，不是 `path != -1`。**
5. **字节保真、不触碰：** 目标 `-1` 永不写；page padding 永不写；int8/FP8 payload 搬移后逐字节相等。

### 5.2 新文件 2：`dspark_components/dspark_tree_ops.py`（节点计划 + 四个算子）

| 节 | 内容 |
|---|---|
| P1 node plan | `NodePlan`、`get_node_plan`、`prepare_nodes`、`commit_nodes`、`NodeRecord` 描述符；容量 buffer 在 init 预分配（固定 `1 + budget` 行） |
| P2 compressor | `compress_nodes` → `compressor_tree_project` + `compressor_tree` |
| P3 indexer | `quantize_indexer_nodes` + `tree_indexer` → `quant_lightning_indexer_tree` |
| P4 attention | `tree_attention` kwargs 构造器 → `sparse_attn_sharedkv_tree` |
| P5 accept 后提交 + SWA 搬移 | gather 被接受的 hidden → 现有链式 compressor（类似 prefill 的合成 batch）写入压缩 KV/state；SWA 行搬移通过 `assign_cache_op`/`unidex_copy`/90 行计划；**没有 freeze/commit 节点页** |

import 是直连的：`from sgl_kernel_npu.tree import (compressor_tree, compressor_tree_project,
quant_lightning_indexer_tree, sparse_attn_sharedkv_tree)`。没有中间层，没有 env 开关。

**算子 import 策略：没有 backend 切换**

| 用途 | 只用 | 绝不用 |
|---|---|---|
| SWA-only 的 base 注意力 | `torch.ops.npu.sparse_attn_sharedkv` + `metadata_host` | `_C_ascend.*` |
| 节点压缩 | `sgl_kernel_npu.tree.compressor_tree_project` / `compressor_tree` | `_C_ascend.compressor_tree*` |
| 节点 indexer | `sgl_kernel_npu.tree.quant_lightning_indexer_tree` | `_C_ascend.quant_lightning_indexer_tree` |
| 节点注意力 | `sgl_kernel_npu.tree.sparse_attn_sharedkv_tree` | `_C_ascend.sparse_attn_sharedkv_tree` |

- **没有** `SGLANG_DSPARK_TREE_BACKEND`，没有 vendor/shadow/fallback，没有运行时切换。
- `_C_ascend` 在本文档里只作为“参考实现的默认 / 被拒绝的替代方案”出现，代码里从不 import。
- `sgl_kernel_npu` 或某个算子缺失时，在 import / 首次调用直接 raise；缺 kernel 是部署错误，不是切回 vendor 的理由。
- `dspark_tree_ops.py` 是**唯一的设备边界**。后续 GPU 适配只替换这个文件（保持同样的函数签名），`dspark_tree.py`、worker、verify 都不动。没有 `TreeOpBackend` 协议或注册表。

### 5.3 修改的文件（2 个实质新文件 + 4 个实质修改 + 6 个小改）

| 文件 | delta |
|---|---|
| `arg_groups/speculative_hook.py` | **仅**对 DSV4 + NPU + 开树时不再强制 `speculative_eagle_topk=1`；新增校验：`topk ≥ 2`、`num_draft_tokens ≥ depth + 1`、`depth ≥ 1`、budget = `num_draft_tokens - 1`、ragged 模式强制 `static`、树 ⇒ 禁 compact/deferred/watermark/sampling-accept-graph。**再加 R5/R6 门禁：** 树 ⇒ adaptive spec / SPS / STS 关闭、verify 宽度不裁剪（每步完整 `1 + budget`）、混合 prefill+decode batch 跑 eager 或被拒绝、per-position accept 估计器禁用或 clamp 到 `budget`。**再加 DP/TP 门禁（§5.6）：** 树 ⇒ `attn_tp_size == 1`（或可证明 markov vocab shard 不存在）、DP attention 打开时必须 `--enable-dp-lm-head`、树 verify 宽度流入 `num_tokens_per_req` |
| `dspark_components/dspark_config.py` | 把 `gamma`（深度）与 `verify_num_draft_tokens`（节点宽度）拆开；保留 topk==1 的 `verify = gamma + 1` 关系 |
| `dspark_components/dspark_worker_v2.py` | `tree_plan` 可选；`if tree_plan is not None:` 包住 plan/accept/commit/grammar/mamba-step；把 `assert topk in (None, 1)` 换成一个选择被接受节点行的树分支 |
| `dspark_components/dspark_verify.py` | 可选 `tree_plan` 参数、per-node positions/cache loc、`DFlashVerifyInput.tree_plan` 载体、树 accept/commit 分支（gather 被接受行 → `dest_slots`/`dest_positions`） |
| `hardware_backend/npu/attention/ascend_dsv4_backend.py` | **+400-450，强制，最大集成面。** `tree_plan is not None` 分支：compressor（project+compress）、indexer、SWA 树注意力、压缩树注意力、per-node metadata。所有现有链式代码路径保持原样 |
| `dspark_components/dflash_info.py` | +2：`tree_plan: Optional[object] = None` |
| 小改（机械性） | `spec_info.py` +8（draft 宽度）、`dsv4_rope.py` +7（clamp）、`deepseek_v4_memory_pool.py` +8（c4 ring）、`arg_groups/fields/spec.py` +17、`arg_groups/field_order.py` +2、`dspark_draft.py` +5（携带 base_logits） |

明确**不改**：`dspark_planner.py`、`dspark_draft.py`、`dspark_kv_inject.py`、
`dspark_deferred_commit.py`、`base_spec_worker.py`、`dflash_info_v2.py`（`DFlashVerifyInput` 只增加
一个带默认值的字段）。

### 5.4 明确不从参考实现移植的部分

参考树代码共 19 个文件 / 2,697 行。整块丢弃：

| 丢弃项 | 原因 |
|---|---|
| `tree/backend.py`（`SGLANG_DSPARK_TREE_BACKEND`，vendor/sgl_kernel_npu/shadow 切换，30 行） | 决策是单 backend |
| `tree/torch_ops.py`（vllm-ascend `.so` 加载器，55 行） | 算子是 `sgl_kernel_npu` 里的 Python+Triton |
| indexer 的 shadow/vendor 半边（`_shadow_outputs_match`、`_vendor_tree_indexer`、`_C_ascend` metadata 算子，约 150 行） | 无 vendor 路径，无 shadow |
| attention availability 探测 + dispatch（约 60 行） | 只有一个调用点 |
| node_pool availability 探测 + 双 compressor 分支（约 40 行） | 只有一个调用点 |
| `tree/shape.py`（41 行） | 未被引用 |
| builder 里重复的 `_empty_layout`（layout 已有） | 重复 |
| README 的 backend 切换章节 | 不是代码 |

保留 2,697 行中的约 1,950 行（约 72%），重组进两个文件。

### 5.5 与 vllm-ascend 树分支的交叉验证（只用于设计，不取代码）

vllm-ascend 树分支是**另一套 kernel 栈**（`_C_ascend` vendor 算子 + ACLNN csrc），所以什么都不
import。它的价值在于展示树的成本和正确性实际落在哪里。

它实际在用的四个 Triton 树算子是 `prune_layout`、`finalize_layout`、`freeze_node_pages`、
`commit_node_pages`，也就是**拓扑收敛和节点页提交**，而不是我们从 `sgl_kernel_npu` 拿到的
compressor/indexer/attention 算子。所以这两块在我们这边仍然是框架侧工作，vllm 的优化是该怎么写的
参考。

| vllm 发现 | 我们的做法 |
|---|---|
| host 侧 budget/top-k 收敛（`argsort`+scatter）卡住图捕获；用 device prune kernel 解决（`612b84ad6`） | 仍留在框架侧，但改成静态 k 的 `torch.topk` + 预分配 index buffer（R1），不新增 kernel |
| KV 搬移每层重算、payload 被重排；改成每组一份 index plan + 先 gather 后 scatter（`b7cd632f2`） | 采纳该计划/不变量设计（R2），包括 `path >= 0` 和 first-writer-wins |
| 节点提交 grid `(jobs, max_rows)` 溢出 UB；改成 one program per job（`2fe76a173`） | C 里不移植（节点页 commit 已丢）；只有当 A1 迫使走回退时才保留 one-program-per-job 的教训 |
| scorer/policy 由 env 驱动、与模型耦合（`d6048af3f`） | 采纳可注入 scorer + 配置化 policy；不用 env 变量 |
| DSV4 树需要 FULL 图，且拒绝 PP/PCP/DCP>1、KV-transfer、动态 depth、混合 batch 图 | 作为显式启动 Gate 采纳（R5/R6） |
| 仓库内唯一测试是 21 例字节级 KV 搬移测试；其余在外部验证 | 我们的测试顺序把字节级行搬移和 accept 计划记账放最前 |
| 其 `ancestors` 填充用安全 root 地址 + `visible=false` | **不照抄**：`sgl_kernel_npu` 契约是 `-1` 填充，算子内部跳过（§4） |

不采纳：vendor `_C_ascend` 算子和 metadata 算子、`npu_scatter_nd_update_` 字节搬移、model 代码里的
全局 `_LAYOUT`/`_PLAN` 上下文钩子、vllm 图管理器、Domino/Qwen GQA 路径、`candidate_size`、
`expand_tree_layer_triton`（上游已是孤儿代码），以及 vllm 的 scheduler patch。

#### 5.5.1 为什么拒绝复用 EAGLE 的 `build_tree` / `verify_tree_greedy`（B 被证伪）

| 证据 | 后果 |
|---|---|
| `build_tree_kernel.cpp:151,181` 用 `selected_index // topk` 推导 parent | EAGLE 是**均匀 topk 叉、parent-major** 的候选空间 |
| `build_tree_kernel.cpp:163-165` 跳过 parent 未被选中的节点 | **没有 ancestor closure**；孤儿节点消耗 budget 但不可达 |
| 参考 `tree/builder.py:47-49,99-129` | DSpark 是 **ancestor-closed + 每层全局 top-`width` + 重新编号** |
| `verify_tree_greedy.py:51` 把 `candidates[i]` 与 `target_predict[i-1]` 比较，从不使用 `retrive_next_token`/`retrive_next_sibling` | Triton accept kernel 是**链式专用**；对任何分支都错 |
| `verify_tree_greedy_native` | 按位置索引的约定，与参考实现的 node-id + `first_child` 走路径不同 |

因此 B 需要一个和参考 builder 一样大的适配器，产出一棵不同的树，并承担更高的静默错误风险。
**拒绝。**

#### 5.5.2 为什么拒绝「verify 时跳过当前块压缩 KV」

树注意力通过 `tree_ancestors` 到 node pool 来解析压缩槽 `k >= prefix // ratio`
（`tree_ops_semantics.md`，`tree_reference.py:539-557`）。当 `prefix % ratio != 0` 时，边界残块
必须把 prefix 之前的 token（paged state）与 draft 节点（node pool）聚合起来。丢掉当前块的节点 KV
会静默丢失一个 key 并改变注意力。`window=128 >= verify block` 只约束 SWA 窗口；它并不能让压缩
节点 KV 变得冗余。**四个树算子全部保留。**

### 5.6 部署拓扑：DP16 / TP16（用户已验证）

参考分支在硬件上以 **TP16 + DP16** 运行。这比「单卡能跑」强得多的约束，因为 DP attention 会改变
verify 分层与 collective 结构。

#### DP attention（DP16）

- 已验证的形状是 TP16 + DP16（`--tp 16 --dp 16 --enable-dp-attention`）。`--dp-size` 是
  `attn_dp_size` 的旧写法（`fields/parallel.py:203-205`），所以解析为 `attn_dp_size = 16`、
  `dp_size = 1`，并且 **`attn_tp_size = 16 // 16 // 1 = 1`**
  （`runtime_context.derive_attn_tp_size:73-76`）。
- 树是**每个 DP rank 按本地 batch 构建**（`bs = len(batch.seq_lens)`），不是全局 batch。不存在
  树计划的跨 DP 广播；每个 rank 的树不同，因为它的请求不同。这是正确的，因为 attention 是 DP 局部的。
- DSpark 要求 `--enable-dp-lm-head`。当 `attn_tp_size == 1` 时，LM head 在每个 rank 上不分片，
  logits 是本地 full-vocab；`_gather_attn_tp_logits` 是 no-op。
- DP MoE collective 是**按 token 计数**的，由 `batch.global_num_tokens` 驱动。verify 宽度必须
  到达 `num_tokens_per_req`（`DFlashVerifyInput` 从 `draft_token_num` 推导），所以每 rank 的计数
  变成 `local_bs * (1 + budget)`。硬编码 `gamma + 1` 会少计，并让本地 batch 大小不同的 rank 死锁。
- `DPSpecPrefillCoordinationPlan` 用 `draft_width = query_token_num`、
  `verify_width = verify_num_draft_tokens` 构建；树宽度也必须流经这里。
- 空闲 DP rank 仍会通过 `run_idle_participation` 进入 draft 和 target forward，以加入按 token
  计数的 collective。树计划对空闲 rank 跳过。
- 树强制 ragged 模式 `static`，这同时关闭了 planner 的动态预算广播和 DP tier all-gather。这是
  简化，不是损失。
- 结论：DP 下树要求 `attn_tp_size == 1`。已验证的 DP16 + TP16 形状满足这一点。

#### Tensor parallel（TP16）

- 该分支没有 `get_attention_tp_size`；用 `get_parallel().attn_tp_size` / `.tp_size`。
- 树需要 **full-vocab 的 base logits 和 full-vocab 的 markov bias**。`DSparkMarkovScorer` 会拒绝
  `head.vocab_shard != None`、有状态的 head，以及宽度与 logits 宽度不同的 bias。
- `SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD` 默认 True，并在 TP>1 时设置 `vocab_shard`，树 scorer
  会拒绝它。当 `attn_tp_size == 1` 时，head 的每分区宽度等于完整 vocab，所以
  `configure_tp_shard` 让 `vocab_shard = None`，树可以继续。否则必须关掉 shard 开关，且移植必须
  保证 gather 后的 full-vocab logits。
- base-logit gather **每次 draft 提议运行一次**（`gather_and_crop_vocab` 或 multimem gatherer），
  产出 `[bs*depth, vocab]`，在每个 rank 上复制。
- 每个 draft step 有一次 `[bs]` 的 `prev_tokens` 广播（站点 `DSPARK_DRAFT`），每次提议 `depth`
  次。树依赖它，让每个 rank 构建相同的 root anchor。
- 树 accept 绕过 `build_accept_outs`，所以 `[bs]` 的 accept 广播不会运行。树接受是确定性的
  target-argmax，因此本来就在各 rank 间一致。
- 树规划、scorer、node pool、accept 后再压缩**不增加任何** collective。

#### 树张量在 rank 间是复制还是分片

| 张量 | 在 TP rank 间 |
|---|---|
| `base_logits`（gather 后） | 复制，full vocab |
| 树 layout / metadata / verify ids / ancestors / 稀疏槽 / 节点计划 | 复制（由复制的输入确定性推导） |
| 喂给节点 compressor 的 target hidden 行 | 复制（attention/MoE all-reduce 之后） |
| 节点 KV（compressor 输出） | 复制（compressor 权重复制，单一 latent head） |
| 压缩 KV / indexer 输出 | 复制（indexer 权重复制，`n_local_heads == n_heads`） |
| per-head Q 与 attention head 输出 | **分片**（`n_local_heads = n_heads // attn_tp`） |
| 临时 LM-head local logits | **按 vocab 分片**，在树看到之前已 gather |

#### 不支持的组合（必须是启动 Gate）

- 树 + sampling accept、folded 提议，或被捕获的 decode 图（树 v1 只做 greedy target-argmax）；
- 树 + `static` 之外的任何 ragged 模式；
- 树 + `attn_tp_size > 1` **且** markov head 分片（scorer 会拒绝）；
- 树 + DP attention 但没有 `--enable-dp-lm-head`；
- 树 + 非 INT8 indexer（ratio 4），或节点 KV dtype 不在 BF16/FP16 内；
- 树 + request 级 KV transfer、PP>1、CP>1。

没有任何 checked-in 测试覆盖 TP>1 + topk>1；仓库内唯一的树测试是 21 例 KV-transport 测试，其余
在外部验证。所以我们的 P6 必须自己加 TP16/DP16 Gate。

---

## 6. 行数预算（先按参考实测，再缩减）

参考实测 `[static]`（`b7c7fa0a` 上的 git blob 当前值）：

| 参考项 | 行数 |
|---|---|
| `tree/` 总计（19 文件） | 2,697 |
| 其中丢弃的脚手架 | 约 750 |
| `node_pool.py` | 476（保留约 300） |
| `indexer.py` | 289（保留约 130） |
| `attention.py` | 252（保留约 140） |
| `node_commit.py` | 96（C 中丢弃；只有 A1 失败才移植） |
| metadata + sparse + commit + accept + output + builder + layout + scoring + config + round | 约 1,100（全保留） |
| worker 树相关 hunk | 约 130-186 |
| verify 树相关 hunk | 约 68-90 |
| config 宽度拆分 | 约 30-35 |
| planner 树代码 | 0 |
| ascend backend 树分支 | 约 150-250（混在 755 行 diff 中） |

本工作的目标：

| 项 | 估算 |
|---|---|
| `dspark_tree.py`（后端中立核心） | 700-850 |
| `dspark_tree_ops.py`（节点计划 + 4 算子 + 再压缩 + SWA 搬移） | 500-600 |
| `ascend_dsv4_backend.py`（强制） | 400-450 |
| worker / verify / config / hook + 6 个小文件 | 330-390 |
| **生产合计** | **1,950-2,150** |
| CPU 测试（16 条不变量套件 + e2e） | 900-1,100 |
| 可选：移植全部 78 个参考 CPU 测试 | 约 1,790 |

两个独立评审收敛到这个区间（plan agent 约 1,980；ultrabrain reviewer 约 1,900-2,100）。参考实现约
3,619 行生产代码；C 小约 45%，因为它丢掉了 vendor/shadow/backend-switch、节点页 freeze/commit，
以及完整的 EAGLE 适配路径。`ascend_dsv4_backend.py +400-450` 是 DSV4 层栈内树注意力的强制入口，
无法藏在「约 5 个修改文件」下面。参考实现的 `metadata.py:57-58`、`commit.py:48`、`sparse.py:62-94`、
`builder.py:109` 使用 `.item()`/`.tolist()`/`argsort`；把它们改成可捕获形状是**约 150 行的增量**，
已包含在内。

---

## 7. topk == 1 完全一致：可强制执行的不变量

要证明的命题（Gate a）：**树 flag 关闭时，输出与 accept length 逐位一致。**

让这件事机械化的设计规则：

1. `speculative_eagle_topk` 保持默认 `1`；hook 只在
   `device=npu` + `DSpark` + draft 是 DeepSeek-V4 + 显式树 flag 时才放开覆盖。
2. 只要 `topk == 1`，就有 `verify_num_draft_tokens == gamma + 1`（关系不变）。
3. 每个新入口都接受 `tree_plan: TreePlan | None = None`；每个新分支都是
   `if tree_plan is not None:`。`None` 表示完全走 `main` 路径。
4. 不新增环境变量；不改 `dspark_planner.py` / `dspark_draft.py`。
5. `DFlashVerifyInput.tree_plan` 默认 `None`；backend 用 `getattr(..., None)` 读取。
6. 树模式在启动时被拒绝，如果 `SGLANG_DSPARK_DEFER_CONTEXT_KV`、compact ragged 模式、watermark
   或 sampling-accept graph 被启用。这些都是基于前缀/stride 的，会静默截断。
7. **R5**：树模式在 adaptive spec / SPS / STS 启用，或 planner 会把 verify 宽度裁到 `1 + budget`
   以下时被拒绝。per-position accept 估计器被禁用，或其位置索引被 clamp 到 `budget`（vllm 在这里
   撞到 `IndexError`，因为上游假设恰好有 `num_spec_tokens` 个位置）。
8. **R6**：混合 prefill+decode batch 永不进入树 verify gear；它跑 eager 或被拒绝。
9. **R1**：树路径的每个张量都预分配、静态形状；每步路径内无 `.item()`/`.tolist()`/`.cpu()`；
   捕获区内不分配。
10. 回归证明：静态 diff 审计（只应有被 guard 的 hunk）、CPU 单测用 `tree_plan=None` 把链式路径
    跑过 plan/metadata/accept，然后在 A3 上针对未修改构建执行 Gate (a)。

参考实现里发现、我们**不会**重复的问题：

- 参考的 `_tree_config is not None` 滑窗检查即使 `topk=1` 也会触发；
- 参考 hook 在 `main` 留作 `None` 的地方发布了 `speculative_dspark_block_size`；
- 参考 `resolve_runtime_config` 改变了 gamma 回退顺序（checkpoint gamma 优先于 `verify-1`），
  会把原本静默的 mismatch 变成硬错误。

---

## 8. Fail-loud 规则（不允许静默截断）

- 带有 `ancestors == -1` 或 `>= n` 的节点行进入 attention/commit → raise，不要「什么都不看」；
- `#nodes > capacity`、`depth > gamma`，或 `budget < depth` → raise；
- 请求树路径却 `topk < 2`，或 `topk` 超出 `[1,2048]` → raise（1 表示链，不是树）；
- `ori_sparse_indices` 宽度不是 128 的倍数，`ratio` 不在 `{4,128}` → raise；
- 填充/未用节点行携带完整失效元组：`token=-1`、`depth=0`、`parent=-1`、`first_child=-1`、
  `next_sibling=-1`、`visibility=false`、`position=-1`、`cache_loc=-1`，且永不被接受；部分失效的
  行就是 bug；
- 一个目标被两条搬移声明，且不在 first-writer-wins 规则内 → raise，绝不静默丢弃；
- 树与不兼容特性同开（deferred commit / compact / watermark / accept graph / adaptive spec /
  SPS / STS / KV transfer / PP>1 / CP>1 / DP-attention）→ 启动时 raise；
- 树 gear 内出现混合 prefill+decode batch → eager 或 raise；
- 捕获中的 forward 内创建任何张量 → 禁止；buffer 在 `__init__` 预分配。

---

## 9. 阶段与 Gate

| 阶段 | 工作 | 证据等级 |
|---|---|---|
| P0 | **证伪（原计划；本机无 torch/NPU 无法运行）**：A1 再压缩 vs 节点页 commit；A2 四算子 oracle；A3/A4/A5 代码断言。必须在硬件上执行 | [hw] |
| P1 | 从 `main` 建分支，推到 `unclezhou`；venv；只做 config/hook 门禁 + 校验（树不可达） | [static] |
| P2 | `dspark_tree.py` 核心（config/builder/metadata/sparse/accept/output/plan），**按 §4.5 可捕获** + 16 不变量 CPU 套件 | [cpu] |
| P3 | `dspark_tree_ops.py`（节点计划、4 个算子适配器、accept 后再压缩、SWA 搬移）+ oracle 契约测试 | [cpu] |
| P4 | `ascend_dsv4_backend.py` 路由（+ `dsv4_rope`、`memory_pool`） | [static] |
| P5 | worker/verify/config 集成 + commit 路径；**topk==1 diff 审计** | [cpu] + [static] |
| P6 | A3 Gate：(a) topk=1 完全一致（DP1 **以及** TP16/DP16）→ (b) topk=2 eager DP1 → (b2) topk=2 eager TP16/DP16 → (c) greedy 短上下文相等 → (d) accept_len vs verify 成本 → (e) 仅当 (d) 为负才做图捕获 | [hw] |

说明：

- **P0 是硬 Gate。** A1 失败 ⇒ 在 P1 之前移植 `node_commit.py`（约 +350 行）。A2 失败 ⇒ 停止；
  `sgl-kernel-npu` 里的 kernel 工作超出本仓库范围。
- (b2) 不是可选项：参考实现只在 TP16/DP16 上被证明可用；一棵在 DP1 能跑、在 DP16 不能跑的树不可
  交付。它同时演练 §5.6 的 `num_tokens_per_req` / coordination 管道。
- (d) 是决策点：vllm 的 DSV4 树只声称 FULL 图运行，而他们测得的收益包含图。如果 eager (d) 在
  verify 成本上失败，答案是 (e)，不是调 kernel。
- (e) 在 `sgl_kernel_npu` 有两个已知的 kernel 侧阻塞点（indexer 的 `.item()`/全选、project 的
  校验同步）。这些是上游 kernel 问题；框架侧必须已经可捕获。
- (a)-(d) 通过之前，不做 kernel/tiling 调优。

---

## 10. 已解决的决策与剩余 Gate

已与用户确定：

1. **CPU torch**：在 `dspark_topk/` 下建本地 venv 并安装 CPU `torch`，让 P2/P3 单测在本工作站
   运行（`[cpu]`）。NPU 仍是 A3 机器上的 `[hw]`。
2. **v1 特性门禁**：当 `SGLANG_DSPARK_DEFER_CONTEXT_KV`、compact ragged 模式、watermark 或
   sampling-accept graph 任一启用时，树模式**启动即大声失败**。绝不静默使用前缀语义。
3. **填充策略**：每个请求永远恰好输出 `1 + budget` 个候选行；未用行显式失效
   （`position=-1`、`ancestors=-1`、`cache_loc=-1`），永不被接受。因此 stride 恒定，这既保住现有
   的 uniform verify 窗口，也为未来的图路径留门。
4. **分支**：`feat/npu-dsv4-dspark-tree`，从本地 `main = 5d4a6f499e` 建，推到远端 `unclezhou`。
5. **硬件基线**：参考分支在 TP16 + DP16 验证过，所以移植必须复现该拓扑（见 §5.6），不只是 DP1。
   这取代了早前「DP 未验证」的错误假设。

剩余 Gate：P1 建分支 / venv、且尚未触碰任何 sglang 代码之前，需要用户明确放行。

---

## 11. 可追溯性

| 本设计 | 参考实现 |
|---|---|
| S1 config | `tree/config.py`；`speculative_hook.py:774-793` |
| S2 scoring/builder | `tree/scoring.py`、`tree/builder.py` |
| S3 metadata | `tree/metadata.py` |
| S4 sparse | `tree/sparse.py` |
| S5 accept/output | `tree/acceptance.py`、`tree/output.py` |
| S6 plan/commit plan | `tree/round.py`、`tree/commit.py` |
| P1 node plan | `tree/node_pool.py:26-96, 301-476` |
| P2 compressor | `tree/node_pool.py:118-205` |
| P3 indexer | `tree/indexer.py:69-229`（去掉 vendor/shadow） |
| P4 attention | `tree/attention.py:36-231`（去掉 dispatch） |
| P5 accept 后提交 | 链式 `forward_compress` + `dspark_kv_inject` + `tree/commit.py:78-101`（SWA 搬移）；节点页 freeze/commit 明确**不**移植 |
| backend 分支 | `ascend_dsv4_backend.py:528-681, 1041-1128, 2452-2659, 2144-2271` |
| R2 行搬移计划 | vllm-ascend `tree/transport.py`、`tree/kv_project.py`、`tree/kv_binding.py`（仅设计） |
| R3 提交顺序 | v5：gather hidden → 链式再压缩 → SWA 搬移；v3 的 freeze/commit 顺序只保留为回退 |
| R4 accept 契约 | vllm-ascend `tree/acceptance.py`、`tree/rejection_sampler.py` |
| R5/R6 门禁 | vllm-ascend `tree/speculator.py`（FULL、PP/PCP/DCP、KV transfer、固定 depth）、`aclgraph_utils.py`（混合 batch） |

---

## 12. 修订日志

| 版本 | 变更 | 触发 |
|---|---|---|
| v1 | 初始设计：2 个新文件、5 个修改文件、topk==1 靠 `tree_plan=None` 保持惰性、图捕获推迟到 P6 | `require.txt` + sglang 参考审计 |
| v2 (R1) | plan/accept/commit 从 P2 起必须可捕获（固定容量、静态 k topk、无 host 读回、捕获区内不分配、行数恒定） | vllm DSV4 树要求 FULL 图；`612b84ad6` 表明 host 收敛是阻塞点 |
| v2 (R2) | KV 行搬移计划不变量：每组一份计划复用、只排索引、两阶段 gather/scatter、first-writer-wins、`path >= 0`、字节保真 | `b7cd632f2` + 其 21 例 NPU 测试 |
| v2 (R3) | 明确提交顺序 freeze → relocate → commit → compact hidden | `2fe76a173` / `full_graph.py` |
| v2 (R4) | 四值 accept 契约（`accept_index`、`num_sampled` 含 bonus、`num_accepted` 只算 draft、`endpoint_node`） | vllm accept 语义 |
| v2 (R5) | 树 ⇒ adaptive spec/SPS/STS 关闭、verify 宽度不裁剪、accept 估计器禁用或 clamp | sglang 有 vllm 没有的宽度裁剪机制；vllm 撞到 per-position 统计的 `IndexError` |
| v2 (R6) | 混合 prefill+decode batch → eager 或拒绝 | vllm `aclgraph_utils.py` 混合 batch guard |
| v2 (budget) | 生产 1,950-2,550；含测试 2,400-3,200 | R2/R4/R5/R6 增量 |
| v3 (DP16) | 树按 DP rank 的本地 batch 构建；verify 宽度必须流入 `num_tokens_per_req` 和 prefill coordination plan；空闲 rank 仍要加入按 token 计数的 collective；要求 `--enable-dp-lm-head` | 用户报告：参考在 TP16/DP16 验证 |
| v3 (TP16) | 树需要 full-vocab base logits 与未分片 markov bias；`attn_tp_size == 1` 使 `vocab_shard` 不存在，否则必须关 shard 开关；每提议一次 full-vocab gather + 每步 `[bs]` draft 广播；树 accept 无广播 | 参考分支 TP 处理（`DSparkMarkovScorer` 拒绝 shard、`gather_and_crop_vocab`、`SpecTpSync`） |
| v3 (gates) | 启动 Gate：树 + 非 static ragged / sampling accept / folded 提议 / 捕获 decode / `attn_tp>1` 且 markov 分片 / DP 无 dp-lm-head / 非 INT8 indexer / KV transfer / PP>1 / CP>1 | §5.6 |
| v3 (budget) | 生产 2,030-2,610；含测试 2,480-3,260 | DP/TP 管道 |
| v4 (EAGLE falsified) | B（复用 `build_tree_kernel_efficient` + `verify_tree_greedy`）被拒：均匀 topk 叉 + 无 ancestor closure；Triton accept 只支持链；产出不同的树 | `build_tree_kernel.cpp:151,181,163-165`；`verify_tree_greedy.py:51` |
| v4 (compressed KV) | 「跳过当前块压缩 KV」被拒：边界残块（`prefix % ratio != 0`）需要节点 KV | `tree_ops_semantics.md`；`tree_reference.py:539-557` |
| v5 (architecture) | **C**：精简 torch 树核心 + 四个算子 + 已实现的节点页提交；accept 后复算推迟；不复用 EAGLE | plan agent + ultrabrain reviewer，两者都把 C 排第一 |
| v5 (commit) | commit = accept → gather 被接受 hidden → 链式 `forward_compress`（类似 prefill，≤ depth+1 行）→ SWA 行搬移；节点页 freeze/commit 丢弃（约 300 行） | 链式 compressor 是第一个写入者；树算子只读 |
| v5 (budget) | 生产 1,950-2,150；测试 900-1,100（16 条不变量）或约 1,790（全部 78）；backend +400-450 强制 | 两个独立估算 |
| v5.2 (trim) | 删除参考专用 helper（`build_chain_*`、`apply_tree_commit`、`plan_tree_round`）和 ops 重复 commit plan；生产约 2,790，测试 66 | CPU 调用图审计；66/66 通过 |
| v5 (P0) | P0 是硬证伪 Gate：A1 再压缩一致性、A2 四算子 oracle；A1 失败 ⇒ 移植 `node_commit.py`（约 +350） | reviewer |

---

## 13. 风险登记（v5）

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| 1 | `verify_num_draft_tokens` 同时是深度、窗口宽度、KV stride、commit 长度和 mamba step 预算；树把它拆成 `depth` 和 `1 + budget` | 高 | 在 `dspark_config.py` 里单点定义；启动断言；topk==1 关系不动 |
| 2 | 如果 verify 宽度没有反映到 `num_tokens_per_req` / coordination，DP 按 token 计数的 collective 会死锁 | 高 | 任何性能工作之前先过 (b2) TP16/DP16 Gate；显式宽度管道 |
| 3 | 如果 base logits 被分片或 draft 广播被关闭，树会在 TP rank 间静默分叉 | 高 | scorer 拒绝 `vocab_shard`；断言 full-vocab 宽度；保留 `DSPARK_DRAFT` 广播 |
| 4 | kernel 侧图阻塞点（indexer `.item()`/全选、project 校验同步） | 中 | 框架侧做成可捕获；kernel 修复是独立的上游改动 |
| 5 | per-position accept 统计假设恰好 `depth` 个位置；`budget > depth` 会破坏它们 | 中 | 树模式禁用或 clamp 估计器 |
| 6 | Mamba/KDA step 映射（`commit_lens - 1`）是链式专用 | 中 | 显式选择最后一个被接受节点行；加断言 |
| 7 | 填充约定不匹配：`sgl_kernel_npu` 用 `-1`，vllm 用 safe-root + invisible | 中 | 遵循 kernel 契约；对部分失效行 fail loudly |
| 8 | 混合 prefill+decode batch 无法重放固定的树 gear | 低/中 | 按 batch eager 或拒绝 |
| 9 | 仓库内没有测试覆盖 TP>1 + topk>1 | 中 | P6 (b2) 是强制硬件 Gate；为宽度管道补 CPU 测试 |
| 10 | 行数超出 v5 估算 | 低 | 生产 1,950-2,150；backend +400-450 是强制项；报告净 diff，区分移植与新逻辑 |
| 11 | **A1**（仅后续优化）：accept 后链式再压缩与已验证的节点页 commit 发散 | 低（当前路径不受影响） | 保留节点页 commit；仅在启用可选复算前运行 A1 |
| 12 | **A2**：`sgl_kernel_npu.tree` 算子与 `tree_reference.py` oracle 发散 | **critical** | P0：在 A3 上运行 Repo C 的 `test_tree_*` + oracle；失败 ⇒ kernel 工作，停止 |
| 13 | 边界残块（`prefix % ratio != 0`）丢弃或错误聚合一个压缩 key | 高 | 测试 `prefix=9, ratio=4`；检查 `(k+1)*ratio-1-prefix` 步骤和 paged+node 聚合 |
| 14 | State-ring 损坏 / 双写 / bonus-token state | 高 | 两步循环，把 `state_cache` 与纯链式基线对比；断言 bonus 只计一次 |
| 15 | `verify_tree_greedy_native` 的 flat-row → node-id 映射差一，破坏 `dest_slots` | 高 | 在有分支的树上做 native vs `greedy_tree_accept` 的 golden 测试；断言 root 在第 0 列 |
| 16 | DP16 按 token 计数的 collective 在本地 batch 不等时少计 / 挂起 | 高 | 静态断言 `num_tokens_per_req == 1+budget`；用不匹配 local bs 的 CPU mock；P6 (b2) |

---

## 14. 评审记录（子代理轮次）

| 轮次 | 评审者 | 范围 | 结论 |
|---|---|---|---|
| R1 | `explore` ×4 | EAGLE 复用面；accept 后再压缩可行性；仓库约定 / GPU 边界；NPU 原生 EAGLE 算子 ABI | 发现 EAGLE builder/accept 不匹配和 NPU 原生算子集 |
| R1 | `Oracle` | A/B/C 对抗性评审 | **不可用**（`Cannot connect to API`，两次） |
| R1 | `Artistry` | 三个非传统的更小设计 | 提出 SWA-covers-verify-block 捷径；后因未证明被拒 |
| R2 | `plan` | 决策完整的架构 + 计划 | **C**；B 被拒；生产约 1,980；backend +400 强制 |
| R2 | `ultrabrain`（Oracle 替代） | 尝试证伪 C/B | **C > A > B**；B 被 kernel 语义证伪；跳过压缩 KV 被拒；6 个不变量测试 + 5 个失败模式 |

参考证据基数：47 个 commit（其中 33 个是行为修复）、45 个文件、+5,024/-94；`tree/` 2,697 行；
CPU 测试 1,337 行 / 78 个测试。

两个决定性证伪的证据锚点：

- B：`build_tree_kernel.cpp:151,181`（`parent = selected_index // topk`）、`:163-165`（孤儿
  `continue`）、`verify_tree_greedy.py:51`（链式专用 accept）、参考 `tree/builder.py:47-49,99-129`
  （ancestor-closed）。
- 跳过压缩 KV：`tree_ops_semantics.md`（树映射）、`tree_reference.py:539-557`、边界
  `prefix % ratio != 0` 部分块。

尚未修改任何 sglang 代码。P0 是第一个可执行步骤，且只在明确批准之后进行。
