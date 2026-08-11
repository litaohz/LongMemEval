# Experiment D · 规模作自变量（结构价值 vs 记忆库规模）

> Fork of `xiaowu0162/LongMemEval` · 实验负责：Tao
> 上游 commit 见 `git log`。本目录是**新增**的，不改动上游任何文件。
>
> ⚠️ **本实验是 Experiment C（结构价值主实验）的延伸，不独立。**
> 主实验在 `litaohz/graphiti` 分支 `ssv/structure-value` 的 `ssv_structure/`。
> 宿主是 **Graphiti**；本 repo 提供**数据与规模档位**。

## 0. 一句话

LongMemEval 的三档（`oracle` / `s` / `m`）**同源同题，只有干扰 session 数不同**
⇒ 记忆库规模是一个**干净的自变量**。
测**结构价值随库规模的变化曲线** —— ⭐ **这条曲线本身就是卖点。**

## 1. 核心假设

> **结构价值随库变大而增长。小库上图结构是纯开销，大库上才回本。**

形式化：设 `Δ₁ = φ(L1) − φ(L0)`（图结构价值）、`Δ₂ = φ(L2) − φ(L1)`（层级价值），
则假设为 **`Δ₁`、`Δ₂` 随库规模单调递增，且在小规模处可能为负**。

⭐ **为什么这个结论非平凡**：
如果只报「加了结构分数更高」，审稿人会说「这就是 GraphRAG 消融，早有人做了」。
**真正非平凡的是那个「负→正」的穿越点**：

> **建 Community 在小库上是负价值**（占预算、引入摘要误差），
> **过了某个库大小才转正 —— 而现有系统全都是无条件建。**

⇒ 落点是：**结构不是免费的，它有代价，而代价什么时候被覆盖没人量过。**

## 2. 为什么这个数据能当自变量（已实测，非转述）

**`longmemeval_oracle`（15MB，500 题）证据 session 数分布**：

| 类型 | 题数 | 证据 session 数分布 |
|---|---|---|
| **multi-session** | **133** | **`{2:84, 3:26, 4:17, 5:6}`** ← ⭐ **49 题 ≥3，主战场** |
| knowledge-update | 78 | `{2:78}` 全部恰好 2 个 |
| temporal-reasoning | 133 | `{1:20, 2:88, 3:15, 4:2, 5:5, 6:3}` |
| single-session ×3 | 156 | 全部 `{1: ...}` |

全库 `{1:176, 2:250, 3:41, 4:19, 5:11, 6:3}` → **≥2 占 64.8%（324 题）**

⭐ **`multi-session` 是结构价值的主战场**：这类题**必须跨 session 聚合才答得出**，
正是 Entity/Community 层该发挥作用的地方。单 session 题上结构价值先验就该是 0
—— **这构成一个天然的负对照**，可用来验证我们的 φ 没有系统性高估结构。

**三档规模（实测）**：

| 档 | 规模 | 相对 |
|---|---|---|
| `oracle` | 每题中位 **2 session / ~6.6k tok**，总 3.3M | 1× |
| `s` | 均值 **47.7 session / ~122k tok**，总 61M | **18.5×** |
| `m` | ≈500 session，推断 ~603M tok | ~180× |

⭐ **题目本身不变，只有干扰 session 数变** —— 这是极难得的实验条件。

## 3. 实验设计

### D1 · 🔴 可重放性验证（go/no-go，必须先做）

⚠️ **全仓库 `seed` 零命中**（实测 `grep -rn "seed" src/` = 空），
只有 `temperature: 0`（`run_generation.py:210,366` / `evaluate_qa.py:108`）。
**temperature=0 在托管 API 上不等于确定性** —— batching、路由、浮点非结合性都会漏噪声。

🔴 **为什么这是硬门槛**：Shapley 是**重叠子集之间做差**。
若重跑噪声与真实边际贡献同量级，**两者在数学上无法区分**，所有 φ 作废。

`replay_check.py` 测 flip rate。判据：
- `0` → GO
- `≤1%` → GO，但必须证明 φ 效应量远超此噪声底，且写进 limitation
- `>1%` → **NO-GO**，先补 seed 再测

### D2 · 三档 × 三级消融（主结果）

对 `oracle` / `s` / `m` 各跑 Experiment C 的四级阶梯
（`L0_episodic_only` / `L0plus_dense` / `L1_entity_edge` / `L2_community`），
画 **`Δ₁`、`Δ₂` vs 库规模** 曲线。

⚠️ **必须带上 `L0plus_dense` 对照臂**：
上游 `EpisodeSearchMethod` **只有 bm25，没有 cosine_similarity**
（`graphiti_core/search/search_config.py:44-45` 实见）
⇒ 朴素的 L1−L0 差值**混淆了「加了结构」和「加了稠密检索」两件事**。
**`φ(L1) − φ(L0plus_dense)` 才是结构价值的保守估计。**
只报朴素差值 = 高估自己的结果。

### D3 · 负对照
**single-session 156 题**：结构价值先验应 ≈ 0。
若这里也测出显著正的 `Δ₁`，说明我们的 φ 有系统性偏置，**必须回头查**。

## 4. 工程

### 已核实的上游情况（源码级）

| 项 | 结论 | 出处 |
|---|---|---|
| ✅ **可拔除，0 行改 harness** | 注入内容完全由 `haystack_dates / haystack_session_ids / haystack_sessions` 三个平行 list 决定 ⇒ **数据层预处理即可** | `run_generation.py:75-77, 92-96` |
| ⚠️ **可重放存疑** | 全仓 `seed` 零命中，只有 `temperature: 0` | 实测 grep |
| ❌ **无 checkpoint** | 输出文件名带时间戳，异常静默 `continue` | `run_generation.py:376-378` |
| judge 很脆 | `'yes' in resp.lower()` 会把 "yes, but actually no" 判对 | `evaluate_qa.py:24-43` |
| memory 后端 | 只有 long-context 直灌 + BM25/contriever/stella/gte，**零第三方 memory 系统** | `src/retrieval/` |

⭐ **正因为上游没有 memory 后端，我们才需要 Graphiti 当宿主** —— 两者互补，不冲突。

⚠️ **官方已迁至 `longmemeval-cleaned`（2025/09 清洗）** —— 早前统计基于旧 repo，
**s/m 文件可能已不同，跑之前需重核**。

📌 **可选**：`mem0ai/memory-benchmarks` 的 `benchmarks/longmemeval/run.py`（1413 行）
自带 seed、续跑判断、异步并发，**比官方 harness 完善**。

### 我们要写的（全部在 `ssv_scale/` 下，零侵入）
```
ssv_scale/
  PLAN.md            ← 本文件
  replay_check.py    ⭐ D1：flip rate（go/no-go），已写
  export_for_graphiti.py  三档 -> Graphiti ingest 格式（一个问题 = 一个 group_id）
  scale_curve.py     D2 主图：Δ₁/Δ₂ vs 库规模
```

⭐ **φ 引擎与消融阶梯都在 graphiti fork 那边**，本 repo 只负责数据与规模。

## 5. 已知风险（不藏）

1. 🔴 **D1 不通过整条线悬空** —— 先做
2. **成本**：`s` 档 61M token、`m` 档 ~603M，而 Shapley 要跑多个子集 ⇒ **乘数很大**。
   **对策：φ 只在受控子样本上跑**（如 multi-session 的 49 题），不是全库。
   且 Graphiti ingest 还要额外过 LLM 抽实体 —— **这是三个实验里最贵的一个**
3. **judge 很脆** —— soft 分数不能直接用上游 judge，需自己实现并报告一致率
4. ⚠️ **上下文预算混淆项** —— 「删了 session 成绩反而更好」有两种解释：
   **它本身有害** vs **上下文短了模型表现更好**。
   **对策：SSV 双算子** `ρ_del`（真删）/ `ρ_pad`（占位置空），`φ_pad − φ_del` 分离内容价值与占位成本
5. **Community 在小库上可能建不起来** —— 聚类退化成一个巨类或全是单点类。
   **必须报告每档的社区数与大小分布**，否则 `Δ₂` 无法解释

## 6. 下一步

- [ ] **Step 0**：`replay_check.py` 测 flip rate（**go/no-go**）
- [ ] Step 1：`export_for_graphiti.py`，oracle 档 multi-session 49 题
- [ ] Step 2：接 graphiti fork 的 `ssv_structure/`，跑四级阶梯
- [ ] Step 3：D2 规模曲线（oracle → s → m）
