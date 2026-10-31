# 从零讲清楚：这个分支做了什么

> 主题：DSV4（DeepSeek V4 系列模型）+ DSpark（本仓库的投机解码特性）+ PCTree（Parent-Conditioned Drafting Tree），在 Ascend NPU 上支持 topk>1 的树形投机解码。
> 读者：没接触过投机解码、DSpark、DSV4 内部结构的人。
> 目标：读完之后能说清楚「以前是什么、现在多了什么、为什么这么做、代码改在哪、怎么验证」。

---

## 读前先记住

下面这些词会在全文反复出现。先扫一眼，记不住也没关系，第 1、2 节会展开。

- **DSV4**：DeepSeek V4 系列模型。它的 decode 不是只有一份 KV cache，而是 SWA、压缩 KV、indexer 三套状态混在一起。
- **DSpark**：本仓库的投机解码特性，负责 draft 和 target 的调度。
- **draft**：小模型，先快速猜接下来几个 token。
- **target**：大模型，一次验证 draft 的多个猜测。
- **topk**：draft 每层保留几个候选。
- **depth**：树有几层，也就是一次 draft 最多往前猜几个 token。
- **budget**：整棵树保留多少个非根节点。
- **prefix**：已经生成并写入 KV cache 的历史。
- **verify / accept / commit**：验证候选、选出接受路径、把接受路径写回 cache。
- **ancestor closure**：保留一个节点，就必须保留它从 root 到自己的整条祖先链。
- **PCTree**：Parent-Conditioned Drafting Tree，父节点条件化建树。同一个深度上，每个 parent 用自己的 Markov bias 算子节点，而不是所有 parent 共用一份子分布。
- **frontier**：当前层参与扩展的父节点集合。
- **joint score**：从 root 到某个节点的累计 log 概率。

完整术语表见第 11 节。

---

## 0. 三分钟版本

**以前**：DSpark 在 `main` 上只有 `topk=1` 的链式投机解码。每一步只猜一条候选链，target 模型一次验证这条链，接受长度有限。

**现在**：这个分支增加了 `topk>1` 的树形验证，具体用的是 **PCTree**（Parent-Conditioned Drafting Tree，父节点条件化建树）的工程化变体。draft 每一步可以保留多个候选，形成一棵树；target 一次验证整棵树；accept 时从中选出一条被接受的路径。`topk=1` 仍然走原来的链式路径，行为不变。

**做了什么**：

- 新增 2 个文件：`dspark_tree.py`（纯 Python/torch 的树核心）和 `dspark_tree_ops.py`（NPU 设备层）。
- 修改 12 个生产文件、增加 3 个测试文件。
- 只使用 `sgl_kernel_npu` 的四个树算子，不使用 vendor `_C_ascend`（厂商算子库）、shadow（影子实现）、backend switch（可切换后端的开关）。
- 提交路径沿用参考实现在 16 卡张量并行 + 16 路数据并行（TP16 + DP16）上验证过的节点页提交（node-page commit）。

**当前状态**：

- CPU 66/66 测试通过。
- `py_compile`、静态检查、禁用项检查通过。
- A3 硬件上的 eager、greedy 一致性、accept length、TP16/DP16 还没有跑。
- 所以：**代码完成，CPU 验证完成，硬件验证待做**。

---

## 1. 投机解码是什么

### 1.1 普通 decode 为什么慢

大模型生成文本是逐 token 的：每生成一个 token，都要把整个模型跑一遍。哪怕只要下一个词，也要等一次完整 forward。

假设要生成 100 个 token，就要跑 100 次大模型。每次 forward 的耗时基本固定，所以 decode 很慢。

### 1.2 投机解码的基本想法

投机解码引入一个小的 draft 模型（草稿模型）：

1. draft 先快速猜出接下来的几个 token；
2. target 大模型一次 forward，同时验证这几个猜测；
3. 猜对的接受，猜错的丢掉，并在第一个错的位置补一个 target token（bonus）；
4. 下一次从新的位置继续。

比喻：学生先做几道题，老师一次批改。批改比逐题讲解快，学生做对越多，整体越快。

### 1.3 链和树

`topk=1` 是链：

```text
prefix -> A -> B -> C -> D
```

每步只猜一个 token，只有一条路径。

`topk>1` 是树：

```text
             A
           /   \
          B     C
         / \     \
        D   E     F
        |
        G
```

每层保留多个候选。target 一次验证整棵树，accept 时从 root 往下走，找一条和 target 预测一致的路径。

本分支用的具体树算法叫 **PCTree**（Parent-Conditioned Drafting Tree），关键在「parent-conditioned」：同一个深度上，不同的父节点会得到不同的子节点分布。4.3 会展开。

树的优点是：一条路径没猜中，可能另一条分支猜中，接受长度更高。代价是 verify 的行数变多，target forward 更贵。

### 1.4 三个参数

| 参数 | 含义 | 例子 |
|---|---|---|
| `topk` | 每个节点向下扩展几个候选 | `topk=2` 表示每个节点最多两个孩子 |
| `depth` | 树有几层，等于 draft 的 horizon（一次 draft 最多猜几个 token，代码里记作 `gamma`） | `depth=3` 表示最多往下猜 3 层 |
| `budget` | 整棵树保留多少个非根节点 | `budget=4` 表示 root 之外保留 4 个节点 |

verify 的行数是 `1 + budget`：1 行是 root/anchor，剩下是候选节点。

`topk=1` 是树的一个特例：每个节点只有一个孩子，`budget = depth`，verify 行数是 `depth + 1`，也就是原来的链。

### 1.5 接受长度和 verify 成本

- **accept length**：一次 verify 平均能推进多少个 token。越大越好。
- **verify cost**：target 一次 verify 的计算成本。verify 行数越多越贵。

树只有在「accept length 的提升」大于「verify 成本的增加」时才划算。这就是验收里要比较两个比值的原因。

---

## 2. DSV4 为什么特殊

### 2.1 DSV4 的 decode 状态是混合的

普通 Transformer 的 KV cache 只有一份。DSV4 不是：

1. **SWA（sliding window attention）**：最近一段窗口的原始 KV，保真度最高。
2. **压缩 KV 池**：更早的历史按 ratio 4 或 128 压缩，降低显存和计算。
3. **lightning indexer**：对压缩历史做 top-k，挑出当前 query 最相关的若干条压缩 KV。

可以理解为：最近几页看原文（SWA），更早的内容看摘要（压缩 KV），indexer 负责从摘要里挑最相关的几条。

### 2.2 树上每个节点都需要自己的状态

链式 decode 只有一个位置在往前走。树上不同节点有不同的父节点、深度和祖先链，所以每个节点都需要：

- 自己的 `position`（绝对位置）；
- 自己的 `depth`（深度）；
- 自己的 `ancestors`（从 root 到自己的祖先链）；
- 自己的 `seqused_kv`（能看到多少历史）；
- 自己的压缩 KV；
- 自己的 indexer top-k；
- 自己的 attention mask（只能看祖先，不能看兄弟）。

### 2.3 为什么只加一个 mask 不够

如果只是给 attention 加一个树 mask，压缩 KV 仍然是按「连续块」算的。但树的分支不在同一个连续位置上：节点 A 的压缩边界和节点 B 的压缩边界不同，祖先链也不同。所以要补的不是一个 mask，而是整套按节点的数据：拓扑、位置、祖先、压缩 KV、indexer、commit。

---

## 3. 总体方案

### 3.1 一句话

**后端中立的纯 torch 树核心 + 四个 `sgl_kernel_npu` 树算子适配器 + 参考实现已验证的节点页提交。**

### 3.2 三层结构

先看层次，不用记文件名；`tree_plan`、`NodePlan` 的细节在第 4 节。

```text
┌──────────────────────────────────────────────┐
│ worker / verify（复用 main 的调度骨架）        │
│   proposal、target forward、accept、commit    │
└───────────────────┬──────────────────────────┘
                    │ tree_plan
┌───────────────────▼──────────────────────────┐
│ dspark_tree.py（新增，纯 torch）              │
│   拓扑、ancestor closure、metadata、          │
│   sparse slots、accept/output、commit plan    │
└───────────────────┬──────────────────────────┘
                    │ NodePlan
┌───────────────────▼──────────────────────────┐
│ dspark_tree_ops.py（新增，设备边界）           │
│   NodePlan、四个树算子、节点页提交、SWA 搬移   │
└───────────────────┬──────────────────────────┘
                    │ 算子调用
┌───────────────────▼──────────────────────────┐
│ ascend_dsv4_backend.py（修改，只做路由）       │
│   compressor / indexer / attention 的树分支   │
└──────────────────────────────────────────────┘
```

读法：worker 决定「做什么」，`dspark_tree.py` 决定「树的形状和接受谁」，`dspark_tree_ops.py` 把决定翻译成 NPU 算子调用，`ascend_dsv4_backend.py` 只负责路由。

### 3.3 为什么不直接复制参考分支

参考分支 `feat/npu-dsv4-dspark-topk` 能跑，但：

- 19 文件的 `tree/` 包，测试前约 3,619 行；
- 默认走 vendor `_C_ascend`，保留 shadow 和 backend switch；
- 与本次「基于最新 main、只复用 sgl_kernel_npu、最小 delta」的要求冲突。

所以不是复制，而是在 `main` 上重新实现同样的语义。

### 3.4 为什么不用 EAGLE 的树算子

EAGLE 是另一种投机解码。它的树算子和 DSpark 要的树不是一回事：

- EAGLE 的树是均匀 topk 叉，父节点用 `selected_index // topk` 推导；父节点没被选中时直接跳过，不保证 ancestor closure。
- EAGLE 的接受算子只支持链式，不使用 `retrive_next_token` / `retrive_next_sibling`。

DSpark 需要 ancestor-closed 的树（定义见开头），所以复用 EAGLE 反而要写一个和参考 builder 一样大的适配器，还会产出不同的树。这里的 kernel 名字可以跳过，不影响理解。

### 3.5 提交路径为什么用节点页

树在 verify 时会给所有节点算一份临时的 compressed KV。但大部分节点会被拒绝，不能直接写进持久化池。

当前做法：

1. verify 时记录每个节点的行（`NodeRecord`）；
2. accept 后，把被接受路径的行搬到连续历史位置（`move_swa_kv`）；
3. 把接受节点的压缩 KV/state 通过节点页 `freeze/commit` 写进持久化池（`prepare_nodes` / `commit_nodes`）；
4. 按 `accept_index` 和目标深度提交 hidden。

另一种做法是 accept 后用现有链式 compressor 重新压缩接受路径（V1）。它可以省掉节点页提交，但必须先证明「复算结果与节点页提交逐位一致」。V1 只能在 A3 上做，所以当前保留参考实现已验证的节点页路径。

---

## 4. 一步 decode 的完整流程

### 4.1 起点

请求已经有 prefix（已经生成并写入 KV 的历史），以及一个 anchor/root token。树从 root 开始长。

### 4.2 draft proposal（复用，不改）

draft 模型跑一次 forward，得到每个深度的 `base_logits [bs, depth, vocab]`（`bs` 是本地 batch size）。它表示「如果沿着链往下走，每个位置下一个 token 的分布」。

树模式没有再跑第二次 draft forward。树是在这些 `base_logits` 上长出来的。

### 4.3 plan_tree_pre：PCTree 把树长出来

这个分支用的树算法叫 **PCTree**（Parent-Conditioned Drafting Tree，父节点条件化建树）。名字里的 parent-conditioned 是关键：同一个深度上，不同的父节点会得到不同的子节点分布，而不是所有父节点共用一份「下一个 token 的概率」。

#### 4.3.1 DSpark 原来怎么走链

DSpark 的 backbone 一次 forward 给出每个深度的 `base_logits`，记作 `L_d`。原来只取一条链：

```text
z_d = L_d + Markov(y_{d-1})
y_d = argmax softmax(z_d)
```

`Markov(y_{d-1})` 是一个小 head，根据「具体的前一个 token」给 logits 加偏置。`topk=1` 时每个深度只保留 argmax，所以是一条链。

#### 4.3.2 PCTree 的关键：每个 parent 单独算 Markov

PCTree 不增加 backbone forward，仍然复用同一份 `L_d`；但它对当前 frontier 里的每个具体父节点 `p` 分别算：

```text
z_d(p) = L_d + Markov(p)
logπ_d(·|p) = log_softmax(z_d(p))
```

也就是说：

- `L_d` 是共享的，backbone 只跑了一次；
- `Markov(p)` 是每个 parent 自己的，所以叫 parent-conditioned；
- 子节点的分数是「在这个 parent 条件下」的 log 概率，不是全树共享的 marginal 概率。

为什么这很重要？如果所有 parent 共用一份子分布，那么「单个 token 很常见」不代表「这条路径条件概率高」，可能拼出一条条件上很不可能的路径。PCTree 用 parent 自己的 Markov bias 打分，保证每个分支都是「在这个父节点之后」合理的续写。

一个示意：

| parent | 共享的 `L_d` | 自己的 `Markov(p)` | 该 parent 下的 top-2 孩子 |
|---|---|---|---|
| p1 | L | bias1 | A, B |
| p2 | L | bias2 | C, D |

同一个 `L_d`，两个 parent 的孩子可以完全不同。这就是和「按 marginal logits 建树」的区别。

#### 4.3.3 逐层扩展、候选池、frontier

设：

- `k = topk`：每个 parent 的局部分支数；
- `B = depth`：block size，也是最大层数；
- `N`：verify 节点预算，**包含 root**；代码里 `N = 1 + budget`；
- `F_{d-1}`：第 d 层使用的 parent frontier，`F_{-1} = {root}`。

每一层 d：

1. 把 frontier 里的所有 parent 批量送进同一个 Markov head；
2. 每个 parent 取 local top-`width` 个子节点（正常 `width = k`），加入候选池；
3. 子节点累计分数：`s(child) = s(parent) + logπ_d(child | parent)`；
4. 在本层候选里按累计分数保留 top-`width`，作为下一层的 frontier。

注意：不是每个 parent 都保留 top-k 个孩子。每个 parent 先产生 top-k 个候选，但一层结束时只留全局 top-`width` 个节点。所以 frontier 的宽度始终 ≤ k，不会指数爆炸。

论文把本层所有 local children 放进候选池，等最后再做 global top-N；本实现为了省内存，在每层就只把 top-`width` 写进 `state`。所以 4.3.4 说的「已选节点」是这些被写下来的节点，差异见 4.3.6。

#### 4.3.4 全局预算与 ancestor closure

跑完 B 层后，对所有已选节点按累计分数排序；root 分数为 0，永远保留；非根节点保留 top-`budget` 个。

因为 `s(child) = s(parent) + logπ ≤ s(parent)`，父节点的分数永远不低于子节点，所以保留集天然是 ancestor-closed（前缀闭包）：保留一个节点，它的祖先一定也在里面。分数相同时，论文的规则是「浅层优先，再按稳定节点 id」；代码用稳定排序，节点 id 按深度优先编号，效果基本一致。

#### 4.3.5 和代码的对应

| PCTree 里的概念 | 代码里 |
|---|---|
| `k` | `topk` |
| `N`（含 root） | `1 + budget` |
| `B` | `depth` / `gamma` |
| `L_d` | `base_logits[:, d]` |
| `Markov(p)` | `head.step_bias(frontier_tokens, hidden_states=None, state=None)` |
| `logπ_d(·\|p)` | `select_candidates` 的 log_softmax + topk |
| `s(c)` | `path_scores` |
| frontier | `TreeBuilder.build` 里的 `frontier` |
| 候选池 | `state.tokens` / `state.path_scores`，只存每层 top-`width` |
| 全局 top-N | `keep_count = min(budget, num_nodes)` 的最终裁剪 |

代码里的 `width = min(topk, vocab, budget)` 是安全上限：正常情况下 `width = topk`；当 vocab 很小或 budget 小于 topk 时，宽度会被压小。

#### 4.3.6 本分支的实现和论文 Algorithm 1 有一个差异

论文的 Algorithm 1 会把**每个 parent 的 local top-k 子节点都放进候选池**，最后在完整候选池上做 global top-N。本分支沿用 vllm-ascend 的 level-wise builder：每层只把全局 top-`width` 个节点写进 `state`，最后在 `width × depth` 个已选节点里做 top-`budget`。

两者都保证 ancestor closure，也都用 parent-conditioned Markov 打分，但不是逐节点等价：如果以后要严格复现论文的候选池，需要把每层所有 local children 都保留下来，并把候选池规模从 `width × depth` 放宽到论文的 `k + (B-1)k²` 量级。当前实现是已经上机验证过的 vllm-ascend 版本，所以分支先用它。

`topk=1` 时，每层只有一个孩子，PCTree 退化成 DSpark 原来的链（最多到全局 budget）。

为什么要 ancestor closure？如果保留了一个节点，但它的父节点被丢掉，这个节点就没有从 root 到它的路径，target 无法验证它，accept 也走不到它。

### 4.4 树的元数据

长完树后，为每个节点计算：

| 字段 | 含义 |
|---|---|
| `position` | `prefix + depth` |
| `depth` | 节点在树里的深度 |
| `ancestors` | 从 root 到自己的祖先节点 id，root-first，`-1` 填充 |
| `seqused_kv` | `prefix + depth + 1`，表示这个节点能看到多少历史 |
| `verify_ids` | `[root, 节点1, 节点2, ...]`，target verify 的 token 行 |
| `ori_sparse_indices` | SWA 窗口槽加祖先节点槽，宽度按 128 对齐 |

### 4.5 NodePlan：节点 id 和 KV slot 是两件事

`NodePlan` 把树节点的逻辑 id 映射成算子需要的行布局：

- `n`：扁平节点行数，等于 `local_batch × (1 + budget)`；`max_q`：单请求 verify 宽度，等于 `1 + budget`；
- `start` / `depths`：每个节点的 prefix 和深度；
- `ancestors`：按请求加了 offset 的祖先 id；
- `req_ids`：每个节点属于哪个请求；
- `query_cu`：长度为 `n+1` 的前缀和，表示第 i 行 query 对应第 i 个节点；
- `hidden_ids` / `token_index`：每个节点用自己的 hidden 行。

注意：**节点 id 不是 KV slot**。节点 id 是树里的坐标，KV slot 是缓存里的物理位置。commit 时要把节点行搬到正确的 slot。

### 4.6 target verify：一次验证整棵树

target 模型对 `[root, 节点1..节点N]` 这些行做一次 forward。对 DSV4 的每一层，下面三块分别对应 2.1 的三套状态：压缩层对应压缩 KV，indexer 层对应 indexer，attention 里的 SWA-only 对应 SWA。

1. **压缩层**：
   - `compressor_tree_project`：把每个节点的 hidden 投影成 KV 和 gate 输出（gate 是压缩时每个 token 的权重系数），写入临时 `nodes`。
   - `compressor_tree`：沿每个节点的祖先链聚合成压缩 KV `node_kv`。
   - 压缩后的 `node_kv` 会做一次动态量化（`npu_dynamic_quant`，一种把浮点 KV 压成 int8 的实现细节，可以先跳过），再交给 indexer 和 attention。
2. **indexer 层**：
   - 对每个节点，在「前缀压缩历史 + 祖先节点」上做 top-k，得到 `cmp_sparse_indices`。
3. **attention**：
   - SWA-only 层：用 base `sparse_attn_sharedkv`，配合 `ori_sparse_indices` 只允许看祖先。
   - 压缩层：用 `sparse_attn_sharedkv_tree`，同时读原始 KV、前缀压缩 KV 和节点 KV。

这一步的输出是每个节点的 target logits 和 hidden。

### 4.7 accept：从树里走出一条路径

target 给每个节点一个 argmax token。accept 从 root 开始：

1. 看 root 的 target argmax；
2. 在 root 的孩子里找 token 相同的节点；
3. 找到就接受，继续看这个孩子的孩子；
4. 找不到就停止；
5. 在停止的位置输出一个 bonus token。

结果是一条被接受的节点路径、接受节点数、bonus、以及把节点行映射到 verify 行的 `accept_index`。

### 4.8 commit：把接受路径写回缓存

1. `move_swa_kv`：把被接受节点的 full/SWA 行搬到连续历史位置。
2. `prepare_nodes` / `commit_nodes`：把接受节点的压缩 KV/state 通过节点页 freeze/commit 写进持久化池。
3. `commit_hidden`：按 `accept_index` 和目标深度顺序提交 hidden。

### 4.9 下一步

下一次 decode 从新的 prefix 继续。整个一步只有 2 次模型 forward：draft 一次，target 一次。

---

## 5. 一个具体例子

假设：

- prefix 长度 100；
- root/anchor 是 A；
- `depth=3`，`topk=2`，`budget=4`；
- 树长成这样（C 分支被预算丢掉）：

```text
             A (root, depth 0)
           /   \
          B     C        (C 被丢掉)
         / \
        D   E
        |
        H
```

（为了图简洁，深度 3 的另一个节点没有画出来；它和 C 一样被预算丢掉。）

保留节点是 B、D、E、H，正好 `budget=4`。ancestor closure 成立：H 的祖先 D、B、A 都在。（C 被丢而 H 保留，是因为 H 这条路径的累计分数高于 C；这不是树结构的必然，而是这个分数分布下的结果。）

verify 行是 `[A, B, D, E, H]`（按节点 id 排序），共 5 行，等于 `1 + budget`。

| 行 | 节点 | depth | position | seqused_kv |
|---|---|---:|---:|---:|
| 0 | A | 0 | 100 | 101 |
| 1 | B | 1 | 101 | 102 |
| 2 | D | 2 | 102 | 103 |
| 3 | E | 2 | 102 | 103 |
| 4 | H | 3 | 103 | 104 |

accept 假设走 A -> B -> D -> H，bonus 是 X：

- 接受节点：B、D、H；
- `commit_lens = 3 + 1 = 4`（root A 行加三个接受节点）；
- 新 prefix 长度是 `100 + 4 = 104`；
- 输出 token 是 B、D、H、X（接受节点 token 加 bonus；root 不重复输出）。

根节点 A 的 KV 在这一步才补进 cache，所以算进 commit；bonus X 这一步只输出、暂不写 KV，下一步再为它计算，所以不出现在 commit 里。

topk=1 的对照：链只有 A -> B -> D -> H 这一条，verify 4 行，`budget=depth=3`，输出也是 B、D、H、X。树在同样的 depth 下多了候选 E，可能换一条路径，但 verify 从 4 行变成 5 行。这就是为什么必须比较 accept length 和 verify 成本。

---

## 6. 代码怎么改

### 6.1 复用（既有逻辑不改）

| 组件 | 为什么能复用 |
|---|---|
| `DSparkWorkerV2` 调度骨架 | 链式和树共用入口，树只在 verify/commit 分支 |
| `dspark_draft.py` 的 draft forward 和 markov block | 树只消费 `base_logits`，不需要第二次 draft forward |
| `dspark_planner.py` | `VerifyWindow` 已有 `positions_2d` / `verify_cache_loc_2d`，树把宽度设为 `1+budget` |
| `TargetVerifyExecutor` | 复用 target forward、ragged verify，只加可选 `tree_plan` |
| `DFlashDraftInputV2` / `DFlashVerifyInput` | 复用输入流程，`DFlashVerifyInput` 只加 `tree_plan` 字段 |
| `ascend_dsv4_backend` 链式路径 | 树分支全部 `if tree_plan is not None`，链式代码不动 |
| DP/TP、ragged verify、KV/cache | 基础设施不变 |

### 6.2 新增 2 个文件

| 文件 | 行数 | 职责 |
|---|---:|---|
| `dspark_tree.py` | 1,134 | 拓扑、ancestor closure、metadata、sparse slots、accept/output、commit plan。只 import torch 和标准库。 |
| `dspark_tree_ops.py` | 835 | NodePlan、四个树算子、节点页提交、SWA 搬移。唯一设备边界，直连 `sgl_kernel_npu`，无 vendor 回退。 |

### 6.3 修改 12 个生产文件 + 3 个测试文件

这张表按文件列改动。读不懂某一行不影响主线，重点看最后一列「不影响」；基础设施行（`fields/spec.py`、`field_order.py`、`dsv4_rope.py`、`deepseek_v4_memory_pool.py`）可以最后再看。

| 文件 | 预期怎么改 | 不影响 |
|---|---|---|
| `speculative_hook.py` | 新增 `--speculative-dspark-tree-enable` / `--speculative-dspark-tree-budget`；tree_enable 时 topk 默认 2；解析宽度；树模式关闭 decode graph；不兼容特性尽早报错（compact ragged 启动，grammar/watermark 运行时报） | 未开树时 topk 仍强制 1 |
| `dspark_config.py` | `gamma`（深度）与 verify 宽度解耦；topk=1 仍是 `gamma+1` | 非树 DSpark 行为不变 |
| `dspark_worker_v2.py` | init 解析树配置；proposal 后 `plan_tree_pre`；把 `tree_plan` 传给 `run_non_compact`；target verify 后 `plan_tree_post`；accept/commit 走树分支；`commit_hidden` 传 `accept_index/tree_positions/tree_commit` | `tree_plan=None` 时走原链式路径 |
| `dspark_verify.py` | `run_non_compact(..., tree_plan=None)`；树时用 `tree_plan.metadata.positions`；`DFlashVerifyInput(tree_plan=...)`；`commit_hidden` 加树分支 | 链式 commit 不变 |
| `dflash_info.py` | `DFlashVerifyInput.tree_plan: Optional[object] = None` | 默认 None |
| `dspark_draft.py` | `DraftProposal` 增加 `base_logits`（+5 行），既有逻辑不改 | draft 行为不变 |
| `ascend_dsv4_backend.py` | `tree_plan is not None` 时：compressor 走 `compress_nodes`；indexer 走 `tree_indexer`；SWA-only 走 base `sparse_attn_sharedkv`；compressed 层走 `attend_nodes`；节点 KV/scale 挂在 `tree_plan` | 链式 attention 路径不动 |
| `dsv4_rope.py` | padding 的 `position=-1` clamp 到 0 | 非负位置不变 |
| `deepseek_v4_memory_pool.py` | c4 ring（压缩状态的环形缓冲区）按 `verify_width+6` 取 next pow2（下一个 2 的幂），最小 16 | 宽度 ≤10 仍是 16；更大时 ring 变大，只影响内存 |
| `fields/spec.py` | 增加两个 tree 参数 | 无 |
| `field_order.py` | 注册两个参数 | 无 |
| `spec_info.py` | draft 宽度取 `speculative_dspark_block_size`，不再从 verify 宽度推导 | topk=1 等于 `num_draft-1` |
| 3 个测试文件 | 66 个 CPU 用例：core 45、op 20、rope 1 | 不涉及 NPU |

### 6.4 不改

`dspark_planner.py`、`dspark_kv_inject.py`、`dspark_deferred_commit.py`、`base_spec_worker.py`。

`dspark_draft.py` 只加 `base_logits` 字段（+5 行），既有 draft 逻辑不改。

### 6.5 调用图

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

---

## 7. topk=1 的输出为什么没变

> 下面讲的是设计不变量。输出级逐位一致还要等 A3 关卡 (a) 确认；目前只有 CPU 和静态检查。

### 7.1 三个开关

1. `speculative_eagle_topk` 默认是 1。代码里显式门禁是「DSpark 路径 + 显式开树开关」；NPU 和 DSV4 是隐式前提（树分支只存在于 `ascend_dsv4_backend`），并不是启动时的显式校验。
2. 即使 tree 开关打开，如果 `topk=1`，`_tree_config.topk > 1` 为假，`plan_tree_pre` 不会被调用，`tree_plan` 保持 `None`。
3. 所有树分支都是 `if tree_plan is not None`。`None` 就是原来的链式路径。

### 7.2 不变量

- `topk=1` 时 `verify_num_draft_tokens == gamma + 1`。
- 不新增环境变量。
- `dspark_planner.py` 不改；`dspark_draft.py` 只加 `base_logits` 字段。
- `DFlashVerifyInput.tree_plan` 默认 `None`。
- `deepseek_v4_memory_pool.py` 的 c4 ring（压缩状态的环形缓冲区）按 verify 宽度算，不受 tree 开关控制；topk=1 且 `gamma ≥ 10` 时 ring 会从 16 变成 32，只影响内存占用，不影响输出。
- 树模式与 deferred commit、compact ragged、watermark、sampling accept graph 互斥。compact ragged 在启动时报错；grammar / watermark 在每步验证时报错（fail-loud）。
- 树模式关闭 adaptive spec、SPS、STS，不裁剪 verify 宽度。
- 混合 prefill+decode batch 不进树 gear。
- 树路径的张量 buffer 预分配、形状固定；但树规划（`plan_tree_pre` / `plan_tree_post`）目前仍有 `.item()` 读回和按请求的 Python 循环（源码里标注 `# HOST-SYNC:`），这是树当前只能走 eager 的原因之一。

---

## 8. DP16 / TP16 为什么能跑

### 8.1 拓扑

参考实现是在 TP16 + DP16 上验证的。因为 `attn_tp_size = tp_size // attn_dp_size // attn_cp_size`（本方案要求 CP=1），`16 // 16 // 1 = 1`。

`attn_tp_size = 1` 意味着：

- 每个 DP rank 上的 LM head 不分片，logits 是本地 full vocab；
- markov head 的 `vocab_shard` 不存在；
- 树 scorer 可以拿到完整 vocab 的 logits 和 bias。

### 8.2 本地 batch

树是按每个 DP rank 的本地 batch 构建的，不是全局 batch。每个 rank 的请求不同，树也不同，这是正确的，因为 attention 是 DP 局部的。

### 8.3 token 计数

DP 的 MoE collective（多卡之间按 token 数做集合通信）是按 token 数计数的。verify 宽度必须进入 `num_tokens_per_req`，所以每个 rank 的计数是：

```text
local_bs * (1 + budget)
```

如果这里写死成 `gamma + 1`，本地 bs 不同的 rank 会算出不一致的全局 token 数，collective 可能挂死。

### 8.4 空闲 rank

没有请求的 DP rank 仍要进入 draft 和 target forward，加入按 token 计数的 collective。树计划对空闲 rank 跳过。

### 8.5 不支持的组合

- `attn_tp_size > 1` 且 markov head 分片：scorer 会拒绝。
- DP 但没有 `--enable-dp-lm-head`：启动报错。
- 非 static ragged、混合 batch、KV transfer、PP>1、CP>1：树模式不支持或走 eager。

---

## 9. 怎么验证

### 9.1 CPU

```bash
python3 -m pytest -q \
  test/registered/unit/spec/test_dspark_tree_core.py \
  test/registered/unit/spec/test_dspark_tree_ops.py \
  test/registered/unit/spec/test_dspark_tree_rope.py
```

结果：66/66 通过。覆盖：

- 树拓扑、ancestor closure、metadata、sparse slots；
- accept/output、commit plan；
- NodePlan、算子调用契约；
- RoPE 负位置 clamp；
- 宽度配置。

这些测试不执行 NPU 算子，四个 `sgl_kernel_npu.tree` 算子只验证调用契约。

### 9.2 静态检查

- `py_compile` / `compileall` 全部改动文件通过。
- 生产代码没有 `_C_ascend`、`SGLANG_DSPARK_TREE_BACKEND`、`tree/` 包引用。
- AST 检查跨模块 import 和调用签名。

### 9.3 A3 关卡

| 关卡 | 做什么 | 证明什么 |
|---|---|---|
| (a) | 关树，topk=1，记录输出和 accept length | topk=1 没有回归 |
| (b) | 开树，topk=2，budget>=depth，static ragged，eager | 树能端到端跑，无 assert/index error |
| (c) | greedy 短上下文，topk=2 与 topk=1 比较 token | 树的正确性 |
| (d) | 同一 workload 比较 accept length 和 verify 成本 | 树是否划算 |
| (e) | TP16/DP16 重复 (a)-(c) | 部署拓扑正确 |
| (f) | 如果 (d) 为负，再做图捕获 | 性能兜底 |

### 9.4 V2

V2 是四个算子和 `tree_reference.py` CPU oracle 的一致性。它必须在 A3 上通过，否则树不可用。

### 9.5 为什么现在只有 CPU

本机没有 NPU，四个算子的真实数值、backend/worker 集成、collective、graph 都只能在 A3 上验证。因此当前状态是「CPU 通过，硬件待验」，不是「全部完成」。

---

## 10. 限制与风险

| 风险 | 说明 |
|---|---|
| V2 未验证 | 四个算子的真实数值还没和 oracle 对齐 |
| 运行时集成 | backend/worker 的 shape/dtype/顺序可能有细节问题 |
| DP token 计数 | 计数不一致可能导致 collective 挂死 |
| markov 分片 | `attn_tp > 1` 时 scorer 会拒绝 |
| mamba + tree | 当前保留 topk 断言，DSV4 非 mambaish，暂不触发 |
| grammar / watermark + tree | 当前 fail loudly，未适配 |
| 图捕获 | 受 kernel 侧和树规划侧的 host sync 阻塞，当前树走 eager |
| V1 复算 | 未启用，只有节点页提交路径可用 |

---

## 11. 术语表

| 术语 | 解释 |
|---|---|
| draft | 小模型，先猜接下来几个 token |
| target | 大模型，负责验证 draft 的猜测 |
| verify | 一次 forward 验证多个候选 |
| accept | 从候选里选出和 target 预测一致的路径 |
| commit | 把接受路径写回 KV cache |
| prefix | 已经生成并写入 cache 的历史 |
| logits | 模型对下一个 token 的分数分布 |
| topk | 每个节点向下扩展的候选数上限；实际每层还受 budget 限制 |
| depth | 树深，等于 draft horizon |
| budget | 非根节点数，verify 行数是 1+budget |
| ancestor closure | 保留节点必须保留整条祖先链 |
| PCTree | Parent-Conditioned Drafting Tree，父节点条件化建树 |
| parent-conditioned | 每个父节点用自己的 Markov bias 算子节点，不共用一份子分布 |
| frontier | 当前层参与扩展的父节点集合 |
| joint score | 从 root 到某节点的累计 log 概率 |
| SWA | 最近窗口的原始 KV |
| 压缩 KV | 按 ratio 4/128 压缩后的历史 KV |
| indexer | 从压缩历史里挑 top-k 的模块 |
| node-page commit | 把接受节点的压缩 KV/state 冻结并提交到持久化池 |
| V1 | accept 后复算与节点页提交的等价性实验 |
| V2 | 四算子与 CPU oracle 的一致性实验 |
| A3 | Ascend A3 机器，硬件验证环境 |

---

## 12. FAQ

**Q: topk=2 一定比 topk=1 快吗？**
不一定。树多了候选，accept length 可能更高，但 verify 行数也更多。必须用 (d) 比较两个比值。

**Q: PCTree 是什么？**
PCTree 是 Parent-Conditioned Drafting Tree。它不是按全树共享的 marginal 概率取 token，而是让每个 parent 单独过 Markov head：`z_d(p) = L_d + Markov(p)`，再按累计 log 概率逐层保留 top-width、最后全局 top-budget。详见 4.3。

**Q: 本分支的 PCTree 和论文完全一样吗？**
不完全。论文 Algorithm 1 保留每层所有 local top-k 子节点到候选池，再做 global top-N；本分支沿用 vllm-ascend 的 level-wise builder，每层只写 top-width，最后在已选节点里裁剪。两者都保持 ancestor closure，但不是逐节点等价。详见 4.3.6。

**Q: 为什么不用 EAGLE 的树？**
EAGLE 是均匀 topk 叉、没有 ancestor closure，accept kernel 只支持链式；复用它产出的树和 DSpark 需要的不一样。

**Q: 为什么不用参考分支的 vendor 算子？**
本次约束是只用 `sgl_kernel_npu`，不引入 vendor `_C_ascend`、shadow、backend switch。

**Q: 为什么当前用节点页提交，而不是 accept 后复算？**
节点页提交是参考实现已经在 TP16/DP16 上验证的路径；复算（V1）还没证明逐位等价，所以先保证功能正确。

**Q: 为什么只有 66 个 CPU 测试，而不是 78？**
参考分支的 78 个里，有一批只覆盖参考专用 helper（例如 `plan_tree_round`、`apply_tree_commit`、`build_chain_*`），这些 helper 不在运行时路径，已经删除；保留的 66 个覆盖运行时和不变量。

**Q: 现在可以直接上生产吗？**
不行。CPU 和静态检查通过，但 A3 的 eager、greedy 一致性、accept length、TP16/DP16、V2 都还没做。

---

## 13. 延伸阅读

- 架构文档（中文）：`docs/design/npu-dspark-tree.zh.md`
- 架构文档（英文）：`docs/design/npu-dspark-tree.md`
- PCTree 论文：From Chains to Trees: Parent-Conditioned Drafting for Semi-Autoregressive Speculative Decoding，arXiv:2608.02123，https://arxiv.org/abs/2608.02123
- 实现分支：`feat/npu-dsv4-dspark-tree-v5`
- 参考分支：`feat/npu-dsv4-dspark-topk`（用户已在 TP16 + DP16 上机验证）
