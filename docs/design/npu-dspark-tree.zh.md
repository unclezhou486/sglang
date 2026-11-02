# DSV4 + DSpark 树形投机解码（topk > 1）Ascend NPU 架构

> **状态：v5.3 已实现，CPU 75/75 通过，A3 硬件关卡待执行。**
> 分支：`feat/npu-dsv4-dspark-tree-v5`，基线 `main = 5d4a6f499e`，最新代码提交 `52a388a233`，已推送到 `unclezhou`。v5.3 是文档修订号，分支名保留创建时的 `-v5`。
> 运行时后端：`vendor`（vllm-ascend `_C_ascend`，默认）或 `sgl_kernel_npu`（仓内 Triton），由 `SGLANG_DSPARK_TREE_BACKEND` 选择。
> 规模：17 文件 = 2 个新文件 + 12 个生产修改 + 3 个测试文件；生产代码仍约 2,800 行。最新 backend/cache 更新没有新增文件。
> CPU 验证：`python3 -m pytest -q test/registered/unit/spec/test_dspark_tree_core.py test/registered/unit/spec/test_dspark_tree_ops.py test/registered/unit/spec/test_dspark_tree_host_opt.py test/registered/unit/spec/test_dspark_tree_rope.py`（75 passed）。

---

## 1. 背景与目标

### 1.1 问题

`main` 里的 DSpark 只实现了链式投机解码（`topk == 1`）：每个请求每步只验证一条候选链，接受长度受限于单链命中。把候选扩成一棵树（`topk > 1`）可以提高接受长度，但 DSV4 的 decode 状态是混合的：SWA 原始 KV、压缩 KV 池（ratio 4/128）、lightning indexer。树上的每个节点都需要自己的拓扑、绝对位置、压缩 KV、indexer top-k 和 ancestor 注意力 mask。只加 mask 或 sparse-index 补丁不够，必须补齐这些按节点的输入。

### 1.2 目标

交付生产级树，支持任意 `topk` / `depth` / `budget`，不是固定配置的 MVP。

1. 树开关关闭时，输出与 accept length 与 `main` 逐位一致。
2. 树开启时，真实请求端到端跑通，无 device assert、无 host index error。
3. Greedy（temperature 0）短上下文下，`topk=2` 与 `topk=1` token 一致。
4. 低延迟 workload（小 batch，1k/4k/8k 输入，256 输出）下，accept length 的收益大于 verify 成本。
5. 所有结论标注 hardware-verified / CPU-only / static-only。

### 1.3 术语

| 术语 | 含义 |
|---|---|
| `topk` | 每层保留的候选宽度（分支因子） |
| `depth` | 树的深度，等于 draft horizon `gamma` |
| `budget` | 每个请求保留的非根节点数；verify 宽度是 `1 + budget` |
| ancestor closure | 一个节点被保留时，它的整条祖先链都必须保留 |
| node-page commit | 把被接受节点的压缩 KV/state 先写入暂存页，再 freeze/commit 到持久化池 |
| SWA | sliding window attention，最近窗口的原始 KV |
| indexer | DSV4 的 lightning indexer，对压缩历史做 top-k |
| V1 / V2 | 两个证伪实验：V1 是 accept 后复算与节点页提交的等价性，失败不影响当前节点页路径；V2 是四算子与 CPU oracle 的一致性，失败则树不可用 |
| A3 | Ascend A3 机器，硬件关卡在这里执行 |

---

## 2. 架构结论

### 2.1 方案 C

**后端中立的纯 torch 树核心 + 四个树算子适配器（vendor / Triton 双后端）+ 参考实现已验证的节点页提交。**

- `dspark_tree.py`：拓扑、ancestor closure、metadata、sparse slots、accept/output、commit plan。只依赖 torch 和标准库，可在 CPU 上测试，也可在 GPU 上复用。
- `dspark_tree_ops.py`：唯一的设备边界。NodePlan、四个树算子、节点页提交、SWA 行搬移，以及运行时 backend 切换。
- `ascend_dsv4_backend.py`：只做路由。当 `tree_plan is not None` 时，把 compressor、indexer、SWA、compressed attention 分派给 `dspark_tree_ops`。

topk=1 完全不经过树：`tree_plan` 默认为 `None`，所有树分支短路，走 `main` 原有的链式路径。

### 2.1.1 运行时 backend

四个算子有两个实现：

| backend | 实现 | 选择方式 |
|---|---|---|
| `vendor`（默认） | vllm-ascend `_C_ascend` AscendC binding | `SGLANG_DSPARK_TREE_BACKEND=vendor` + `SGLANG_DSPARK_TREE_OPS=/path/a.so:/path/b.so:/path/c.so` |
| `sgl_kernel_npu` | 仓内 Triton package | `SGLANG_DSPARK_TREE_BACKEND=sgl_kernel_npu`（别名 `triton` / `sgl`） |

vendor binding 由 `torch.ops.load_library` 显式加载；SGLang 不 import vllm。选中的 backend 不可用时直接报错，不静默回退。

算子映射：

| 公共 API | vendor binding |
|---|---|
| `compressor_tree_project` | `torch.ops._C_ascend.compressor_tree_project` |
| `compressor_tree` | `torch.ops._C_ascend.compressor_tree` |
| `quant_lightning_indexer_tree` | `torch.ops._C_ascend.quant_lightning_indexer_tree` |
| `sparse_attn_sharedkv_tree` | `torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree` |

vendor 的 indexer / attention 需要真实调度 metadata：

- indexer：`npu_quant_lightning_indexer_v2_metadata`
- attention：`npu_sparse_attn_sharedkv_metadata`

Triton 实现只校验 metadata ABI、不读内容，所以 zero metadata 只适用于 `sgl_kernel_npu`。

`NodePlan` 上缓存 per-step 结果：`cu_seqlens_cache`、`max_seqlen_cache`、`attn_metadata_cache`、`indexer_lengths_cache`、`max_kv_cache`、`indexer_metadata_cache`、`select_all_cache`、`residue_cache`，避免同一 verify step 的每一层重复构造 vendor schedule 和 host readback。

### 2.2 架构落地：复用、修改、新增

**复用（既有逻辑不改）**

| 组件 | 为什么能复用 |
|---|---|
| `DSparkWorkerV2` 调度骨架 | 链式和树共用入口，树只在 verify/commit 分支 |
| `dspark_draft.py` 的 draft forward 和 markov block（仅 +5 行暴露 `base_logits`） | 树只消费 `base_logits`，不需要第二次 draft forward |
| `dspark_planner.py` | `VerifyWindow` 已有 `positions_2d` / `verify_cache_loc_2d`；树把宽度设为 `1+budget` |
| `TargetVerifyExecutor` | 复用 target forward、ragged verify；只加可选 `tree_plan` |
| `DFlashDraftInputV2` / `DFlashVerifyInput` | 复用 draft/verify 输入；`DFlashVerifyInput` 加一个 `tree_plan` 字段 |
| `ascend_dsv4_backend` 链式路径 | 树分支全部 `if tree_plan is not None`，链式代码不动 |
| DP/TP、ragged verify、KV/cache | 基础设施不变 |

**新增（2 个文件）**

| 文件 | 职责 | 边界 |
|---|---|---|
| `dspark_tree.py` | 拓扑、ancestor closure、metadata、sparse slots、accept/output、commit plan | 只 import torch 和标准库，不 import 设备 |
| `dspark_tree_ops.py` | NodePlan、四个树算子、节点页提交、SWA 搬移、backend 切换 | 唯一设备边界，支持 vendor / `sgl_kernel_npu` |

**修改（12 个生产文件 + 3 个测试文件）**

| 文件 | 预期怎么改 | 不影响 |
|---|---|---|
| `speculative_hook.py` | 新增 `--speculative-dspark-tree-enable` / `--speculative-dspark-tree-budget`；tree_enable 时 topk 默认 2；调用 `resolve_dspark_widths`；树模式关闭 decode graph；不兼容特性尽早报错（compact ragged 启动，grammar/watermark 运行时报） | 未开树时 topk 仍强制 1 |
| `dspark_config.py` | 把 `gamma`（深度）和 `verify_num_draft_tokens`（宽度）解耦；topk=1 仍是 `gamma+1` | 非树 DSpark 行为不变 |
| `dspark_worker_v2.py` | init 解析树配置；proposal 后 `plan_tree_pre`；把 `tree_plan` 传给 `run_non_compact`；target verify 后 `plan_tree_post`；accept/commit 走树分支；`commit_hidden` 传 `accept_index/tree_positions/tree_commit` | `tree_plan=None` 时走原链式路径 |
| `dspark_verify.py` | `run_non_compact(..., tree_plan=None)`；树时用 `tree_plan.metadata.positions`；`DFlashVerifyInput(tree_plan=...)`；`commit_hidden` 加树分支 | 链式 commit 不变 |
| `dflash_info.py` | `DFlashVerifyInput.tree_plan: Optional[object] = None` | 默认 None |
| `dspark_draft.py` | `DraftProposal` 增加 `base_logits` 字段并在 eager proposal 赋值（+5 行） | 既有 draft 逻辑不改 |
| `ascend_dsv4_backend.py` | `tree_plan is not None` 时：compressor 走 `compress_nodes`；indexer 走 `tree_indexer`；SWA-only 走 base `sparse_attn_sharedkv`；compressed 层走 `attend_nodes`；节点 KV/scale 挂在 `tree_plan` | 链式 attention 路径不动 |
| `dsv4_rope.py` | padding 的 `position=-1` clamp 到 0 | 非负位置不变 |
| `deepseek_v4_memory_pool.py` | c4 ring 按 `verify_width+6` 取 next pow2（最小 16） | 小宽度仍是 16 |
| `fields/spec.py` | 增加两个 tree 参数 | 无 |
| `field_order.py` | 注册两个参数 | 无 |
| `spec_info.py` | draft 宽度取 `speculative_dspark_block_size`，不再从 verify 宽度推导 | topk=1 等于 `num_draft-1` |
| 3 个测试文件 | 75 个 CPU 用例：core、ops、host-opt、rope | 不涉及 NPU |

**不改**：`dspark_planner.py`、`dspark_kv_inject.py`、`dspark_deferred_commit.py`、`base_spec_worker.py`。`dspark_draft.py` 仅 +5 行暴露 `base_logits`，既有逻辑不改。

**调用点**

```text
worker._forward_decode
  proposal                        # 复用
  plan_tree_pre                   # 新增：base_logits -> TreePrePlan
  run_non_compact(tree_plan=...)  # 复用 executor，新增参数
    DFlashVerifyInput.tree_plan
    backend tree branches         # compressor / indexer / SWA / compressed
  plan_tree_post                  # 新增：accept / output / commit plan
  commit: move_swa_kv + prepare_nodes/commit_nodes + commit_hidden
```

### 2.2.1 PCTree 建树（本分支实际用的算法）

树不是按 marginal base logits 直接取 topk，而是 **PCTree**（Parent-Conditioned Drafting Tree）的 level-wise 变体：

- 每个深度共享 backbone 的 `base_logits[:, d]`；对 frontier 里每个具体 parent `p` 单独调用 `markov_head.step_bias(p)`，得到 `L_d + Markov(p)`。
- 每个 parent 取 local top-`width` 子节点，按累计 joint score 逐层保留 top-`width` frontier。
- 跑完 `depth` 层后，在已选节点里保留 top-`budget` 个非根节点；因为子节点分数不高于父节点，保留集天然 ancestor-closed。
- `topk=1` 退化为原链。

与论文 Algorithm 1 的差异：论文保留每层所有 local top-k 子节点到候选池，再做 global top-N；本分支沿用 vllm-ascend 的 level-wise builder，每层只写 top-`width`，最后在 `width × depth` 个已选节点里裁剪。两者都保持 ancestor closure，但不是逐节点等价。详见从零讲解 §4.3。

### 2.3 每步数据流

| 阶段 | 链（topk=1） | 树（topk>1） |
|---|---|---|
| draft proposal | 1 次 draft forward，`bs*(gamma 或 gamma+1)` 行 | 相同，不改 |
| 规划 | 无 | `plan_tree_pre`：拓扑、metadata、sparse slots、NodePlan |
| target verify | `[anchor, d1..dγ]` | `[root, n1..nN]`，`bs*(1+budget)` 行 |
| 压缩 / indexer / attention | 连续块 | 逐节点，四个树算子 |
| accept | 标量 `correct_len` | 节点路径 + bonus + commit plan |
| commit | 前缀门 `col < commit_lens` | SWA 行搬移 + 节点页提交 |
| 每步 forward 次数 | 2 | 2 |

树模式没有第二次 draft forward。Draft 仍然只产出 `base_logits [bs, depth, vocab]`，树由 markov scorer 在这些 logits 上逐层扩展；节点的 target hidden 来自 target verify，这也正是节点 KV 的来源。

### 2.4 四个算子的管线

```text
compressor_tree_project(x, wkv, wgate, nodes, ape, positions, cu)
    -> nodes fp32 [T+1, 1, 2*c*d]（row 0 保留）

compressor_tree(state, nodes, norm, cos, sin, table, prefix,
                ancestors, depths, requests, ratio, rope_dim, epsilon)
    -> node_kv bf16 [T, d]（非压缩边界节点为零行）

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

契约要点：

- `compressor_tree_project` 是唯一 in-place 的算子，`nodes` 必须预分配。
- `metadata` 是 int32[1024]。Triton backend 校验后忽略，传静态零即可；vendor backend 需要 `npu_sparse_attn_sharedkv_metadata` / `npu_quant_lightning_indexer_v2_metadata` 生成的真实调度 metadata。
- `cu` 只用于断言“一 query 一节点”。
- `seqused_kv = prefix + depth + 1`。
- `ori_sparse_indices` 宽度必须是 128 的倍数。
- SWA-only 层不使用树注意力算子，使用 base `torch.ops.npu.sparse_attn_sharedkv` 加 `metadata_host`。

### 2.5 提交路径

```text
verify 阶段：
  compressor_tree_project + compressor_tree
      -> 计算所有树节点的临时 compressed node KV
      -> 记录每个节点的 NodeRecord

accept 之后：
  move_swa_kv
      -> 把被接受的 full/SWA 行搬到连续历史位置
  prepare_nodes + commit_nodes
      -> 节点页 freeze/commit，把 compressed KV/state 写进持久化池
  commit_hidden
      -> 按 accept_index 和目标深度顺序提交 hidden
```

这是参考实现已经在 TP16/DP16 上验证过的路径。

accept 后链式复算（V1）没有启用：它会在 accept 之后用现有 chain compressor 重新压缩接受路径，可以省掉节点页提交，但必须先证明它与节点页提交逐位等价。V1 只能在 A3 上做，因此当前保留已验证路径。

---

## 3. 为什么不是其他方案

### 3.1 不采用方案 A：整体移植参考树层

参考实现的 `tree/` 包是 19 个文件（18 个 `.py` + README），当前 checkout 共 2,573 行 Python，还带着 vendor `_C_ascend`、shadow、backend switch、节点页提交脚手架。我们不复制整个包，但 `dspark_tree_ops.py` 接入了 `vendor` / `sgl_kernel_npu` 两个 backend，并去掉了 shadow。

### 3.2 不采用方案 B：复用 EAGLE 树算子

EAGLE 的 `build_tree_kernel_efficient` 和 `verify_tree_greedy` 不能直接用于 DSpark：

- `build_tree_kernel.cpp:151,181` 用 `selected_index // topk` 推导父节点，是均匀 topk 叉、parent-major 的候选空间。
- `build_tree_kernel.cpp:163-165` 在父节点未被选中时直接跳过，不保证 ancestor closure。
- 参考 DSpark builder 是 ancestor-closed、每层全局 top-width、重新编号的树，两者不是同一棵树。
- 当前 EAGLE 的 `verify_tree_greedy` 使用 `retrieve_next_token` / `retrieve_next_sibling`，本身支持树 accept；不能复用的原因是 EAGLE 的树均匀 topk、parent-independent、不保证 ancestor closure，而且不处理 DSV4 的 per-node 压缩 KV/indexer。

复用它需要写一个和参考 builder 一样大的适配器，还会产出不同的树，放大静默 orphan 和 accept 错误。

### 3.3 不跳过 verify 时的当前块压缩 KV

树注意力会把压缩槽 `k >= prefix // ratio` 解析到 `tree_ancestors` 对应的 node pool。当 `prefix % ratio != 0` 时，边界残块必须把 prefix 之前的 token（paged state）和 draft 节点（node pool）聚合起来。跳过当前块压缩 KV 会静默丢失一个 key。因此四个树算子全部保留。

---

## 4. 约束与不变量

### 4.1 topk=1 完全一致

1. `speculative_eagle_topk` 默认是 1；代码显式门禁是 DSpark 路径 + 显式开树开关，NPU/DSV4 是隐式前提（树分支只存在于 `ascend_dsv4_backend`）。
2. `topk=1` 时 `verify_num_draft_tokens == gamma + 1`。
3. 所有树入口都接受 `tree_plan=None`，所有树分支都是 `if tree_plan is not None`。
4. 不新增环境变量；`dspark_planner.py` 不改；`dspark_draft.py` 只新增 `DraftProposal.base_logits` 字段（+5 行），既有逻辑不改。
5. `DFlashVerifyInput.tree_plan` 默认是 `None`。
6. 树模式与 deferred commit、compact ragged、watermark、sampling accept graph 互斥。compact ragged 在启动时报错；grammar / watermark 在每步验证时报错（fail-loud）。
7. 树模式关闭 adaptive spec、SPS、STS，不裁剪 verify 宽度。
8. 混合 prefill+decode batch 不进树 gear。
9. 树路径的张量 buffer 预分配、形状固定；但树规划（`plan_tree_pre` / `plan_tree_post`）仍有 `.item()` 读回和按请求的 Python 循环（源码里标注 `# HOST-SYNC`），这是当前只能走 eager 的原因之一。
10. 回归方式：静态 diff 审计加 CPU 测试，最后在 A3 上与未修改版本逐 token 对齐。

### 4.2 Fail-loud

- `ancestors == -1` 或越界的节点行进入 attention/commit，报错。
- `#nodes > capacity`、`depth > gamma`、`budget < depth`，报错。
- 请求树路径但 `topk < 2`，报错。
- `ori_sparse_indices` 宽度不是 128 的倍数、`ratio` 不在 `{4,128}`，报错。
- 未使用节点必须完整失效：`token=-1`、`depth=0`、`parent=-1`、`first_child=-1`、`next_sibling=-1`、`visibility=false`、`position=-1`、`cache_loc=-1`。
- 树模式与不兼容特性同开，启动报错。
- 混合 prefill+decode batch 在树 gear 内，走 eager 或报错。
- 捕获区内不允许创建 tensor，buffer 在 `__init__` 预分配。

### 4.3 DP16 / TP16

参考实现是在 TP16 + DP16 上验证的，这个拓扑对应 `attn_tp_size = tp_size // attn_dp_size = 1`。

- 树按 DP rank 的本地 batch 构建，不是全局 batch。
- DP 要求 `--enable-dp-lm-head`。
- verify 宽度必须流入 `num_tokens_per_req`，每 rank 的计数是 `local_bs * (1 + budget)`。
- 空闲 DP rank 仍要进入 draft 和 target forward，加入按 token 计数的 collective。
- 树要求 `attn_tp_size == 1`；否则 `DSparkMarkovScorer` 会拒绝 vocab-sharded markov。
- 树模式强制 static ragged，关闭 planner 的动态预算广播和 DP tier all-gather。

### 4.4 图捕获现状

树路径当前是 **eager only**，不是 capture-safe：

- hook 对 `topk>1` 显式 `disable_decode_cuda_graph`；
- `TreeBuilder` 每步分配 token/parent/path buffer；
- `build_tree_metadata` / `build_tree_ori_sparse_indices` / `plan_tree_commit` 有 `.item()` / `int()` 读回和逐请求循环；
- `build_tree_accept_outs` 有 `bool(mask.any())`；
- node commit / `move_swa_kv` 有 host descriptor 和循环；
- 树 accept 长度导致 commit shape 动态。

因此 `tree_plan=None` 的 topk=1 仍可走原 decode graph；topk>1 树路径必须 eager。入图计划见 `dspark-tree-roadmap.zh.md`。

当前已知的 kernel 侧图阻塞点：indexer 的 `lengths.max().item()` 和 select-all 分支，`compressor_tree_project` 的校验同步。这些在 `sgl-kernel-npu` 内，不在本仓库。

---

## 5. 实现清单

### 5.1 新文件

| 文件 | 行数 | 职责 |
|---|---:|---|
| `dspark_components/dspark_tree.py` | 1,134 | 设备无关的 tree 核心 |
| `dspark_components/dspark_tree_ops.py` | ~1,100 | NodePlan、backend 切换、四个树算子、节点页提交、SWA 搬移、per-step cache |

`dspark_tree.py` 分成 config、scoring/builder、layout、metadata、sparse、accept/output、commit plan、round 八节。`dspark_tree_ops.py` 分成 node plan、compressor、indexer、attention、node commit 五节。

### 5.2 修改文件

| 文件 | 改动 |
|---|---|
| `ascend_dsv4_backend.py` | +478 / −11：compressor、indexer、SWA、compressed attention 的树分支 |
| `dspark_worker_v2.py` | +132 / −2：tree config、plan pre/post、accept/commit |
| `dspark_verify.py` | +66 / −12：`tree_plan` 参数、树提交分支 |
| `dspark_draft.py` | +5：`DraftProposal.base_logits` |
| `dspark_config.py` | +45 / −16：gamma 与 verify 宽度解耦 |
| `speculative_hook.py` | +55 / −39：树参数、宽度解析、门禁 |
| `dflash_info.py` | +2：`DFlashVerifyInput.tree_plan` |
| `dsv4_rope.py` | +5 / −2：负位置 clamp |
| `deepseek_v4_memory_pool.py` | +7 / −1：c4 ring 覆盖树宽度 |
| `fields/spec.py`、`field_order.py`、`spec_info.py` | +26 / −1：tree 参数与 draft 宽度 |

### 5.3 明确不做

- 不新建 `tree/` 文件夹。
- 不复制 SGLang。
- 不新增第二个 DSpark worker、算法枚举、`SpecInputType` 或 `ForwardBatch` 字段。
- 不引入 `shadow`；支持 `vendor` 和 `sgl_kernel_npu` 两个 backend。
- 不在框架层重写四个算子。
- 不在 A3 的 (a) 到 (d) 通过前调 tiling 或 kernel。

---

## 6. 验证状态

### 6.1 已验证

- CPU：75/75 通过，覆盖 `test_dspark_tree_core.py`、`test_dspark_tree_ops.py`、`test_dspark_tree_host_opt.py`、`test_dspark_tree_rope.py`。
- `py_compile` 和 `compileall` 全部改动文件通过。
- 生产代码 backend-neutral：`dspark_tree.py` 不含设备 backend；`dspark_tree_ops.py` 持有 vendor/Triton 切换。
- AST 检查：跨模块 import 和调用签名没有缺失。
- 测试覆盖树拓扑、ancestor closure、metadata、sparse slots、accept/output、commit plan、NodePlan、算子调用契约、RoPE clamp、宽度配置。

### 6.2 未验证

- 四个算子在 vendor / `sgl_kernel_npu` 两个 backend 下的真实数值（V2）。
- 在 A3 上加载三个 vendor binding `.so` 和 vendor metadata op。
- backend 和 worker 的真实运行时集成。
- A3 eager、greedy 一致性、accept length 与 verify 成本。
- TP16/DP16。
- 图捕获。

### 6.3 A3 关卡

按顺序执行：

1. (a) 关树，topk=1，记录输出和 accept length 作为基线。
2. (b) 开树，`topk=2`，`budget >= depth`，`SGLANG_RAGGED_VERIFY_MODE=static`，eager。
3. (c) greedy 短上下文：topk=2 token 与 topk=1 一致。
4. (d) 同一 workload：`accept_len(topk=2)/accept_len(topk=1) > verify_cost(topk=2)/verify_cost(topk=1)`。
5. (e) 在 TP16/DP16 上重复 (a) 到 (c)。
6. (f) 如果 (d) 为负（eager 下 accept 收益覆盖不了 verify 成本），下一步才做图捕获；如果 (d) 为正，不进入图捕获。

V2（四算子对 `tree_reference.py` oracle）必须在 A3 通过后才能签收。

---

## 7. 风险

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| 1 | 四个算子与 CPU oracle 不一致（V2） | high | A3 上运行 `test_tree_*` 和 oracle；失败即停止 |
| 2 | backend/worker 运行时集成有 shape/dtype/顺序问题 | high | A3 (a) 到 (d) 逐步验证，失败即修 |
| 3 | DP token 计数不匹配导致 collective 挂死 | high | 断言 `num_tokens_per_req == 1+budget`；TP16/DP16 关卡 |
| 4 | markov head 分片或 `attn_tp > 1` | high | `DSparkMarkovScorer` 拒绝 vocab_shard；树要求 attn_tp=1 |
| 5 | mamba + tree | medium | `_commit_target_mamba_states_after_verify` 保留 topk 断言；DSV4 非 mambaish |
| 6 | grammar / watermark + tree | medium | 当前 fail loudly；需要时再适配 |
| 7 | 图捕获被 kernel 侧和树规划侧 host sync 阻塞 | medium | 当前树走 eager；设备化与图计划见 roadmap |
| 8 | V1 复算与节点页提交不等价 | low（当前路径不受影响） | 只作为 V1 通过后的可选优化 |
| 9 | 规模高于精简估算 | low | 当前生产约 2,790；需要更小只能等 V1 通过后替换提交路径 |

---

## 附录 A：追溯

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
| P3 indexer | `tree/indexer.py:69-229` |
| P4 attention | `tree/attention.py:36-231` |
| P5 node commit | `tree/node_commit.py`、`tree/commit.py:78-101` |
| backend branches | `ascend_dsv4_backend.py:528-681, 1041-1128, 2452-2659, 2144-2271` |

## 附录 B：证据锚点

- 方案 B 被证伪：`build_tree_kernel.cpp:151,181`（parent = selected_index // topk）、`:163-165`（orphan 直接 continue）、参考 `tree/builder.py:47-49,99-129`（ancestor-closed）；EAGLE 的 `verify_tree_greedy` 虽支持树 accept，但它的树不含 DSpark 需要的 ancestor-closed / per-node DSV4 状态。
- 跳过压缩 KV 被证伪：`tree_ops_semantics.md`、`tree_reference.py:539-557`，以及 `prefix % ratio != 0` 的边界残块。
- DP/TP：`runtime_context.derive_attn_tp_size:73-76`、`fields/parallel.py:203-205`。

## 延伸阅读

- 从零讲解（适合第一次接触投机解码 / DSV4 的读者）：[npu-dspark-tree-explained.zh.md](npu-dspark-tree-explained.zh.md)
- 后续开发路线（入图、mixed batch、P0-P4 规划）：[dspark-tree-roadmap.zh.md](dspark-tree-roadmap.zh.md)
- 既有文件修改说明（12 个公共文件逐文件讲解）：[dspark-tree-modified-files.zh.md](dspark-tree-modified-files.zh.md)
