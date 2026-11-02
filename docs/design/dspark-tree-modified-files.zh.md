# DSV4 + DSpark 树形投机解码：既有生产文件修改说明（公共部分）

> **范围**：只讲 12 个既有生产文件的修改，不展开新增的 `dspark_tree.py` / `dspark_tree_ops.py`，也不逐行讲 3 个测试文件。  
> **基线**：`main = 5d4a6f499e`；实现分支 `feat/npu-dsv4-dspark-tree-v5 @ 52a388a233`。
> **用途**：给评审和 onboarding 看的“改了什么、为什么改、topk=1 有没有受影响”。  
> **配套文档**：
> - 设计与差异：[dspark-tree-topk-design-and-diff.zh.md](dspark-tree-topk-design-and-diff.zh.md)
> - 从零讲解：[npu-dspark-tree-explained.zh.md](npu-dspark-tree-explained.zh.md)
> - 后续路线：[dspark-tree-roadmap.zh.md](dspark-tree-roadmap.zh.md)
>
> 代码片段是节选，省略了参数、日志和断言；完整 diff 见 `git diff main...HEAD`。

**阅读路径**：第 1 节总览 → 第 3 节配置层 → 第 4 节接线层 → 第 5 节后端层 → 第 6 节修复/适配层 → 第 7 节 topk=1 影响矩阵。

---

## 0. 一句话总结

这 12 个文件的修改可以分成四层：

```text
配置层    fields/spec.py, field_order.py, speculative_hook.py, dspark_config.py
接线层    dspark_draft.py, dflash_info.py, dspark_verify.py, dspark_worker_v2.py
后端层    ascend_dsv4_backend.py
修复/适配层 dsv4_rope.py, deepseek_v4_memory_pool.py, spec_info.py
```

核心思路：

- 新增配置：树开关、budget；
- 把 `gamma` 和 `verify_width` 解耦；
- 让 draft 暴露 `base_logits`；
- 把 `tree_plan` 从 worker 传到 verify/backend；
- 在 backend 里加四条树路由 + 两个 helper；
- 修两个树路径暴露出来的边界问题：负 position、c4 ring 宽度；
- 所有树分支都以 `tree_plan is not None` 或 `topk > 1` 为门，topk=1 走原路径。

---

## 1. 总览

| # | 文件 | 精确 +/− | 角色 |
|---|---|---:|---|
| 1 | `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py` | +478 / −11 | 四条树路由 + 两个 helper、算子接线、absolute page table |
| 2 | `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` | +132 / −2 | worker 生命周期接线：plan pre/post、accept/commit |
| 3 | `python/sglang/srt/speculative/dspark_components/dspark_verify.py` | +66 / −12 | verify 接收 tree_plan、tree commit_hidden |
| 4 | `python/sglang/srt/arg_groups/speculative_hook.py` | +55 / −39 | CLI、宽度解析、关图、门禁 |
| 5 | `python/sglang/srt/speculative/dspark_components/dspark_config.py` | +45 / −16 | gamma 与 verify width 解耦 |
| 6 | `python/sglang/srt/arg_groups/fields/spec.py` | +16 / −1 | 两个新 CLI 字段 |
| 7 | `python/sglang/srt/speculative/spec_info.py` | +8 / −0 | draft 宽度从 block_size 取 |
| 8 | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | +7 / −1 | c4 ring 按 verify 宽度扩容 |
| 9 | `python/sglang/srt/speculative/dspark_components/dspark_draft.py` | +5 / −0 | `DraftProposal.base_logits` |
| 10 | `python/sglang/srt/hardware_backend/npu/dsv4/dsv4_rope.py` | +5 / −2 | padding position=-1 clamp |
| 11 | `python/sglang/srt/arg_groups/field_order.py` | +2 / −0 | 注册新字段 |
| 12 | `python/sglang/srt/speculative/dflash_info.py` | +2 / −0 | `DFlashVerifyInput.tree_plan` |

合计：**+821 / −84**。  
再加上 2 个新文件（`dspark_tree.py` 1,134 + `dspark_tree_ops.py` ~1,100），才是生产新增约 3,050 行（净增约 2,970）。

下面按“配置层 → 接线层 → 后端层 → 修复/适配层”讲，和上面的文件编号顺序不同。

---

## 2. 调用链先看一遍

```text
speculative_hook
  └─ resolve_dspark_widths
       ├─ resolve_tree_config
       └─ 决定 verify_width / gamma / 是否关图

dspark_draft
  └─ DraftProposal.base_logits

dspark_worker_v2
  ├─ resolve_tree_for_server
  ├─ plan_tree_pre
  ├─ run_non_compact(tree_plan=...)
  │    └─ dspark_verify → backend → dspark_tree_ops
  ├─ plan_tree_post
  └─ tree accept / move_swa_kv / prepare_nodes / commit_nodes
```

---

## 3. 配置层

### 3.1 `arg_groups/fields/spec.py`（+16 / −1）

**改了什么**

新增两个 CLI 字段：

- `speculative_dspark_tree_enable`
- `speculative_dspark_tree_budget`

同时更新 `speculative_dspark_block_size` 的说明：链式 verify 是 `gamma + 1`，树是 `1 + budget`。

**关键片段**

```python
speculative_dspark_tree_enable: A[
    bool,
    "DSPARK only. Enable draft-tree topk>1 verification. Requires "
    "--speculative-eagle-topk > 1 and --speculative-dspark-tree-budget; "
    "the verify width is 1 + budget and the tree runs eager.",
] = False

speculative_dspark_tree_budget: A[
    Optional[int],
    "DSPARK tree only. Number of non-root tree nodes. It may exceed gamma "
    "up to topk * gamma; the verify width is 1 + budget. topk=1 (chain) "
    "still requires budget == gamma.",
] = None
```

**为什么**

树需要两个用户可见的旋钮：是否开树、树保留多少非根节点。预算不能从 `gamma` 反推，因为树可以 `budget > gamma`。

**topk=1 影响**

默认 `False` / `None`，不影响。

**验证点**

- 参数能出现在 `--help`；
- `tree_enable=False` 时现有 DSpark 启动不变；
- `tree_enable=True` 但缺少 budget 时启动报错。

---

### 3.2 `arg_groups/field_order.py`（+2）

**改了什么**

把两个新字段注册进 `POSITIONAL_FIELD_ORDER`。

```python
"speculative_dspark_block_size",
"speculative_dspark_tree_enable",
"speculative_dspark_tree_budget",
"speculative_dspark_sps_table_path",
```

**为什么**

保证参数顺序、哈希和序列化稳定。

**topk=1 影响**

无。

---

### 3.3 `arg_groups/speculative_hook.py`（+55 / −39）

**改了什么**

1. `tree_enable` 时，`speculative_eagle_topk` 默认 2；
2. 不再手写 `gamma + 1`，读取 ragged mode 后改调 `resolve_dspark_widths`；
3. 自动声明 `speculative_num_draft_tokens = widths.verify_width`；
4. 树 `topk > 1` 时关闭 decode CUDA graph；
5. 清掉用户显式设置的 `cuda_graph_bs_decode/prefill`；
6. 保留宽度 >= 2 的检查。

**关键片段**

```python
tree_enable = bool(getattr(cfg, "speculative_dspark_tree_enable", False))
if tree_enable:
    if cfg.speculative_eagle_topk is None:
        declare_resolution(..., speculative_eagle_topk=2)
elif cfg.speculative_eagle_topk is None:
    declare_resolution(..., speculative_eagle_topk=1)
```

```python
widths = resolve_dspark_widths(
    explicit_block_size=cfg.speculative_dspark_block_size,
    checkpoint_gamma=draft_config.resolve_gamma(default=None) if draft_config else None,
    num_draft_tokens=cfg.speculative_num_draft_tokens,
    tree_enable=tree_enable,
    tree_topk=cfg.speculative_eagle_topk,
    tree_budget=cfg.speculative_dspark_tree_budget,
    ragged_mode=ragged_mode_value,
    default_gamma=DEFAULT_DSPARK_GAMMA,
)
declare_resolution(..., speculative_num_draft_tokens=widths.verify_width)
```

```python
if widths.tree is not None and widths.tree.topk > 1:
    logger.warning("DSpark tree runs eager; disabling decode CUDA graphs.")
    declare_resolution(..., disable_decode_cuda_graph=True)
```

**为什么**

- 树宽度由 `1 + budget` 决定，不能再用 `gamma + 1`；
- 当前树路径有 host sync，不能进 decode graph；
- 必须在参数解析阶段就把宽度、gamma、图开关定下来，worker 不再猜。

**topk=1 影响**

非树路径仍然 `verify_width = gamma + 1`；topk=1 不会被关图逻辑命中（`widths.tree.topk > 1` 为假）。

**验证点**

- 非树 DSpark 启动参数与 main 一致；
- 树模式 `--speculative-num-draft-tokens` 与 `1+budget` 不一致时报错；
- 树模式日志出现 “disabling decode CUDA graphs”。

---

### 3.4 `dspark_components/dspark_config.py`（+45 / −16）

**改了什么**

`resolve_runtime_config` 新增 `gamma` 和 `verify_num_draft_tokens` 两个参数，明确：

- `gamma` 是 draft horizon；
- `verify_num_draft_tokens` 是 target verify 行数；
- 树：`verify = 1 + budget`；
- 链：`verify = gamma + 1`；
- 只有 legacy fallback 才用 `verify - 1` 反推 gamma。

**关键片段**

```python
if verify_num_draft_tokens is not None:
    verify_width = int(verify_num_draft_tokens)
    resolved_gamma = int(gamma) if gamma is not None else draft_config.resolve_gamma(default=None)
    if resolved_gamma is None:
        resolved_gamma = verify_width - 1
    if verify_width < resolved_gamma + 1:
        raise ValueError(...)
```

**为什么**

树打破了“verify 宽度 = gamma + 1”的旧假设。如果 worker 继续从 verify 宽度反推 gamma，draft 会少跑/多跑。

**topk=1 影响**

非树路径 `verify = gamma + 1`，数值不变。

---

## 4. 接线层

### 4.1 `dspark_components/dspark_draft.py`（+5）

**改了什么**

`DraftProposal` 增加 `base_logits` 字段；eager proposal 把它填进去。

```python
class DraftProposal(...):
    verify_ids: Optional[torch.Tensor] = None
    # [bs, gamma, vocab] pre-Markov logits for tree scoring; None on the folded
    # graph path (tree v1 requires the eager path).
    base_logits: Optional[torch.Tensor] = None
```

```python
base_logits = None
...
return DraftProposal(..., base_logits=base_logits)
```

**为什么**

PCTree 要在 draft 的逐深度 logits 上展开；树不增加 draft forward，必须复用这份 `base_logits`。

**topk=1 影响**

默认 `None`，原逻辑不改。

**验证点**

- 非树 proposal 的 `base_logits is None`；
- 树模式 `proposal.base_logits` 非空；
- folded proposal 下树模式报错（worker 检查）。

---

### 4.2 `speculative/dflash_info.py`（+2）

**改了什么**

`DFlashVerifyInput` 增加：

```python
tree_plan: Optional[object] = None
```

**为什么**

树计划要跟着 verify input 传到 backend；DFlash 非树时默认 `None`。

**topk=1 影响**

默认 `None`，无影响。

---

### 4.3 `dspark_components/dspark_verify.py`（+66 / −12）

**改了什么**

1. `run_non_compact(..., tree_plan=None)`；
2. 树模式用 `tree_plan.metadata.positions` 覆盖原 positions；
3. `DFlashVerifyInput(tree_plan=tree_plan)`；
4. `commit_hidden` 增加 `accept_index` / `tree_positions` / `tree_commit`；
5. tree commit 时按 `accept_index` 重排 hidden，用 `tree_commit.dest_positions/dest_slots` 提交；
6. `accept_index` 路径用 `gather_accepted_rows` 把 positions/cache_loc 按接受路径收拢；
7. SWA ring 下，tree commit 用 `req_pool_indices` 和 `max_rows` 重建 `state_slot`。

**关键片段**

```python
def run_non_compact(..., tree_plan=None):
    positions_2d = verify_window.positions_2d
    if tree_plan is not None:
        positions_2d = tree_plan.metadata.positions.reshape(verify_ids_2d.shape)
    ...
    DFlashVerifyInput(..., tree_plan=tree_plan)
```

```python
if tree_commit is not None:
    max_rows = int(tree_commit.dest_slots.shape[1])
    path = torch.zeros((bs, max_rows), dtype=torch.long, device=hidden.device)
    if max_rows > 1:
        node_ids = accept_index[:, : max_rows - 1] % hidden.shape[1]
        valid = accept_index[:, : max_rows - 1] >= 0
        path[:, 1:] = torch.where(valid, node_ids, torch.zeros_like(node_ids))
    hidden = torch.gather(hidden, 1, path.unsqueeze(-1).expand(...))
    positions = tree_commit.dest_positions
    cache_loc_2d = tree_commit.dest_slots
```

**为什么**

树 verify 的 position 不是连续的 `prefix + i`，而是每个节点的 `prefix + depth`；commit 也不是简单前缀门，而是 winner 路径重排。

**topk=1 影响**

`tree_plan=None`、`tree_commit=None` 时走原 positions / 原 commit。

**验证点**

- 链式 verify 的 positions/commit 与 main 一致；
- 树 commit 的 `commit_lens/new_seq_lens` 与 accept 路径一致；
- `accept_index` 为 `-1` 的 padding 列被 commit gate 丢掉。

---

### 4.4 `dspark_components/dspark_worker_v2.py`（+132 / −2）

**改了什么**

1. init 把 `gamma` / `verify_num_draft_tokens` 传给 `resolve_runtime_config`；
2. init 解析 `resolve_tree_for_server`（要求 static ragged）；
3. 校验 `verify_num_draft_tokens == tree.verify_width`；
4. 校验 `sliding_window > 0`（从 `hf_config.sliding_window` / `window_size` 取）；
5. proposal 后调 `plan_tree_pre`，传 `page_table` / `full_to_swa_mapping` / `draft_slots`；
6. 把 `tree_plan` 传给 `run_non_compact`；
7. verify 后调 `plan_tree_post`；
8. tree accept / commit 分支；
9. `commit_hidden` 传 `accept_index/tree_positions/tree_commit`；
10. 树分支拒绝 folded proposal / grammar / watermark；
11. tree accept 后 ngram update 用 `accept.out_tokens`，而不是 `verify_ids_2d`；
12. tree accept 的 `cap_trim_lens` 置零，不做 cap trim。

**关键片段**

```python
self._tree_config = resolve_tree_for_server(...)
if self._tree_config is not None:
    if self.verify_num_draft_tokens != self._tree_config.verify_width:
        raise RuntimeError(...)
    if self._tree_sliding_window <= 0:
        raise ValueError(...)
```

```python
tree_pre = None
if self._tree_config is not None and self._tree_config.topk > 1:
    tree_pre = plan_tree_pre(
        base_logits=proposal.base_logits,
        root_token_ids=draft_block_ids[:, 0],
        prefix_lens=prefix_lens,
        block_table=page_table,
        block_size=1,
        window=self._tree_sliding_window,
        config=self._tree_config,
        scorer=DSparkMarkovScorer(self.draft_model.markov_head, self._tree_config.topk),
        ...
    )
    verify_ids_2d = tree_pre.metadata.verify_ids.reshape(bs, -1)
```

```python
tree_post = plan_tree_post(pre=tree_pre, target_predict=..., ...)
...
if tree_post is not None:
    accept = AcceptOuts(
        correct_len=tree_post.outputs.num_accepted,
        bonus=tree_post.acceptance.bonus,
        cap_trim_lens=torch.zeros_like(tree_post.outputs.num_accepted),
        commit_lens=tree_post.commit.commit_lens,
        new_seq_lens=tree_post.commit.new_seq_lens,
        out_tokens=tree_post.outputs.out_tokens,
    )
    move_swa_kv(pool, tree_post.commit)
    if records:
        prepared = prepare_nodes(...)
        commit_nodes(prepared)
```

**为什么**

worker 是唯一知道“proposal 完成、target verify 完成、该 commit”的地方；树必须在这三个时点插入 pre/post 计划。

**topk=1 影响**

`_tree_config is None` 或 `topk <= 1` 时 `tree_pre/tree_post` 都不生成，走原 accept/commit。

**已知缺口（不在本次修改内）**

当前树分支没有 `sampling_info.is_all_greedy` 门禁，非 greedy 会静默走 argmax；这是 roadmap 的 P0，不是这次已修内容。

---

## 5. 后端层

### 5.1 `hardware_backend/npu/attention/ascend_dsv4_backend.py`（+478 / −11）

**改了什么**

新增树路由和辅助函数：

| 位置 | 作用 |
|---|---|
| `_tree_swa_ops` | 选支持 `ori_sparse_indices` 的 base `sparse_attn_sharedkv` |
| `_build_full_state_page_table` | 给 tree compressor/commit 构造绝对 state page table |
| compressor 树分支 | 逐节点压缩、记录 NodeRecord、写 `tree_node_kv` |
| `_forward_tree_indexer` | 逐节点 c4 top-k，支持 select-all 快捷路径 |
| `_forward_tree_swa` | SWA-only 层：base 算子 + 树 slot |
| `_forward_tree_attention` | 压缩层：`attend_nodes` + node KV |

另外还有几处容易漏掉的接线：

- `_op_supports_kwarg`：用 op schema 反射判断 `ori_sparse_indices` / `ori_topk` 是否被当前 `sgl-kernel-npu` 支持，避免不同 kernel 版本签名不同；
- tree verify 的 `ori_win_left` 扩到 `window + query_width - 2`：让 root 刚写入的 KV 落在 SAS workspace 窗口内；
- `_compute_kernel_metadata` 在 tree 下传 `ori_topk = ori_sparse_indices.shape[-1]`，链式传 0；
- tree compressor 用 `rope.ensure_tables`（按绝对 `start+depth` 索引），非 tree 用 `get_cos_sin`；
- compressor 树分支：非 arch35 用 `_build_full_state_page_table` 重建绝对 state table（滑窗表列 0 是 `start_pos-history`，tree kernel 按绝对 position 索引会错位）；`compress_nodes` 后把 `node_kv` 挂到 `tree_plan.tree_node_kv[layer_id]`；indexer compressor 额外 `quantize_indexer_nodes`，并记录 `NodeRecord`；
- indexer 树分支：短历史走 `tree_indexer_select_all`，否则走 `tree_indexer`。

**关键片段**

```python
def _tree_swa_ops():
    sgl_attn = getattr(torch.ops.npu, "sparse_attn_sharedkv", None)
    sgl_meta = getattr(torch.ops.npu, "sparse_attn_sharedkv_metadata_host", None)
    if (
        sgl_attn is not None
        and sgl_meta is not None
        and _op_supports_kwarg(sgl_attn, "ori_sparse_indices")
    ):
        return sgl_meta, sgl_attn
    return None
```

```python
if tree_plan is not None:
    from sglang.srt.speculative.dspark_components.dspark_tree_ops import (
        compress_nodes,
        get_node_plan,
    )
    plan = get_node_plan(tree_plan)
    node_kv, _ = compress_nodes(...)
    tree_plan.tree_node_kv[compressor.layer_id] = node_kv
```

```python
def _forward_tree_indexer(self, c4_indexer, tree_plan, ...):
    nodes = tree_plan.tree_node_kv.get(c4_indexer.layer_id)
    node_scale = tree_plan.tree_node_scale.get(c4_indexer.layer_id)
    ...
```

**为什么**

树在 verify 期间破坏了“position 唯一 KV”的假设：

- compressor 要按节点生成压缩 KV；
- indexer 要按祖先选块；
- attention 要用 `tree_ancestors` 解析 node pool；
- SWA-only 层没有压缩 KV，只能复用 base 算子。

**topk=1 影响**

所有树路由都在 `tree_plan is not None` 下；链式 attention/compressor/indexer 路径不改。

**验证点**

- 链式 `tree_plan=None` 时 `_forward_tree_*` 不被调用；
- 树模式 compressor/indexer/attention 的 shape/dtype 与 ops 契约一致；
- 非 arch35 的 absolute state page table 构造正确。

---

## 6. 修复/适配层

### 6.1 `hardware_backend/npu/dsv4/dsv4_rope.py`（+5 / −2）

**改了什么**

```python
safe_positions = positions.clamp_min(0).to(torch.long)
cos = cos_cache.index_select(0, safe_positions)
sin = sin_cache.index_select(0, safe_positions)
```

**为什么**

树的 padding 行可能带 `position=-1`，`index_select` 不接受负下标。这些行 downstream 会被丢弃，但 RoPE 不能先崩。

**topk=1 影响**

非树 positions 非负，clamp 是 no-op。

---

### 6.2 `mem_cache/deepseek_v4_memory_pool.py`（+7 / −1）

**改了什么**

```python
if is_speculative:
    if compress_ratio == 4:
        needed = int(num_draft_tokens) + 6
        return 1 << max(4, (needed - 1).bit_length())
    return 256          # c128, speculative
# 非 speculative: 8 (c4) / 128 (c128)
```

**为什么**

c4 ring 原来的固定 16 写不下更宽的 tree verify。当前按 `verify_width + 6` 取下一个 2 的幂，最小 16。

**topk=1 影响**

- 宽度 <= 10 仍是 16；
- 宽度 >= 11（例如 gamma >= 10 的宽链式）会从 16 变 32，只影响显存，不影响输出。

---

### 6.3 `speculative/spec_info.py`（+8）

**改了什么**

DSpark draft 宽度优先从 `speculative_dspark_block_size` 取：

```python
if self.is_dspark() and is_draft_worker:
    gamma = get_spec().speculative_dspark_block_size
    if gamma is not None:
        return int(gamma)
    return num_draft_tokens - 1
```

**为什么**

树模式下 draft horizon 是 `gamma`，verify width 是 `1 + budget`，不能再从 verify width 反推 draft。

**topk=1 影响**

链式下两者相等，结果不变。

---

## 7. topk=1 影响矩阵

| 改动 | 是否影响 topk=1 | 保证方式 |
|---|---|---|
| 两个新 CLI | 否 | 默认 False/None |
| 树宽度解析 | 否 | 非树 `verify_width=gamma+1` |
| draft `base_logits` | 否 | 默认 None，原逻辑不变 |
| verify `tree_plan` | 否 | 默认 None，原 positions/commit |
| backend 树路由 | 否 | `if tree_plan is not None` |
| worker tree pre/post | 否 | `_tree_config is None or topk<=1` 不生成 |
| rope clamp | 否 | 只影响负 position |
| c4 ring 扩宽 | **可能** | 宽链式（gamma>=10）会多占显存 |
| hook 自动声明 num_draft_tokens | **可能** | 用户不填时默认行为变化 |

---

## 8. 验证状态

- CPU：76/76 通过（含 core / ops / host-opt / rope）；
- `py_compile` / 静态检查通过；
- A3 eager、greedy 一致性、accept length、TP16/DP16：**未跑**；
- V2 四算子对 oracle：**未跑**；
- 非 greedy 树路径：**未拒绝**，roadmap P0；
- 树图捕获：**未做**，当前 eager only。

---

## 9. 本文没覆盖什么

- `dspark_tree.py` / `dspark_tree_ops.py` 的内部实现：见 `dspark-tree-topk-design-and-diff.zh.md` §4；
- 三个测试文件：见设计文档 §4.3；
- 入图、mixed batch、设备化的具体计划：见 `dspark-tree-roadmap.zh.md`；
- 非 greedy fail-loud、compact 断言：属于待补的 P0，不是本次已改内容。
