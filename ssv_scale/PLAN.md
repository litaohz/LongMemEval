# Experiment B · 记忆库规模 × 统计代理偏差

> Fork of `xiaowu0162/LongMemEval` · 实验负责：Tao
> 上游 commit 见 `git log`。本目录是**新增**的，不改动上游任何文件。

## 0. 一句话

把**记忆库规模**当自变量（oracle → s → m），用反事实 φ 当 ground truth，
测量**廉价统计代理**（召回频次 / 保鲜度 / 命中率）的偏差**随库增大如何变化**。

## 1. 核心主张

统计信号**便宜、可在线持续更新、但有偏**；反事实 φ **无偏、能表达集合级干扰、但贵**。

> **真问题：统计信号在多大程度上能逼近反事实价值？偏差有多大？什么条件下能替代？**

🔴 **统计代理的硬伤：召回频次由检索器决定，不由价值决定。**
检索器用相似度 ⇒「常被召回」≈「embedding 位置居中、跟很多 query 都沾边」= **曝光偏差不是效用**。
泛而无用的记忆频次高；关键但冷门的记忆频次为零。
更糟的是会**自我实现**：高频 → 判定有价值 → 排序靠前 → 更高频
（= Jiang et al. `1902.10730` 推荐系统 degenerate feedback loop，**该线在 LLM agent memory 文献零引用**）。

⭐ **这把工作从「又发明一个打分」变成「校准一个已有的打分」，不易被说成方法找问题。
而且自带可做的实验 —— 没人做过这个校准，因为没人有 φ。**

⚠️ **「无偏」必须加限定语**：Shapley 消除的是**曝光偏差**（只看在场/不在场，检索器污染不到）。
但 φ 完全取决于所选效用函数与任务集合 ⇒ 准确说法是「**相对于给定任务分布无偏**」，
不是「客观真值」。**不写这个限定，第一个审稿人就会问「相对于什么无偏」。**

## 2. 为什么选这个数据（已核实，非转述）

**实测 `longmemeval_oracle`（15MB，500 题）证据 session 数分布**：

| 类型 | 题数 | 证据 session 数分布 |
|---|---|---|
| **knowledge-update** | **78** | ⭐ **`{2:78}` 全部恰好 2 个，无一例外** |
| **multi-session** | 133 | `{2:84, 3:26, 4:17, 5:6}` ← 49 题 ≥3 |
| temporal-reasoning | 133 | `{1:20, 2:88, 3:15, 4:2, 5:5, 6:3}` |
| single-session ×3 | 156 | 全部 `{1: ...}` |

全库 `{1:176, 2:250, 3:41, 4:19, 5:11, 6:3}` → **≥2 占 64.8%（324 题）**，≥3 占 14.8%。

**三点强过 LoCoMo**：
1. ⭐ **证据是 session 级 + 稳定 ID**（`answer_xxx_1` / `_2`）—— **天然可拔除单元，不必自定义**
2. **有 78 题明确的知识更新**（LoCoMo 一条都没有）
3. **多证据比例更高**（64.8% vs 26.6%）

⭐ **决定性理由：三档同源同题，只有干扰 session 数不同**
⇒ **记忆库规模是一个干净的自变量，题目本身不变。** 这是极难得的实验条件。

| 档 | 规模（实测） |
|---|---|
| `oracle` | 每题中位 **2 session / ~6.6k tok**，总 3.3M |
| `s` | 均值 **47.7 session / ~122k tok**，总 61M（**18.5×**） |
| `m` | ≈500 session，推断 ~603M tok |

## 3. 实验设计

### B1 · 可重放性验证（🔴 go/no-go，必须先做）
⚠️ **全仓库 `seed` 零命中**（实测 `grep -rn "seed" src/` = 空），
只有 `temperature: 0`（`run_generation.py:210,366` / `evaluate_qa.py:108`）。

- 同一输入重跑 N 次，测 **flip rate**
- 若 flip rate 非零 → **必须自加 `seed=` 参数并重测**
- ❌ **flip rate 压不下去，整个 φ 方案在这个宿主上悬空** —— 这一步不通过就不要往下做

### B2 · 单元级 φ（oracle 档）
- 单元 = session（用官方 `session_ids`）
- **多证据题 324 条**为主战场；单证据题 φ 退化为 LOO，闭式算掉不采样
- 采样预算全压到 ≥3 证据的 74 题

### B3 · ⭐ 主结果：代理偏差 vs 规模
对每个单元同时记录：
- `φ`（反事实，ground truth）
- `recall_freq`（被检索器召回次数）
- `recency`（保鲜度）
- `hit_rate`（召回后答对率）← ⚠️ 这条**性质不同**，它带 outcome 反馈，是 observational 版本的 φ

**测三档（oracle / s / m）上 `corr(proxy, φ)` 与偏差方向**，画成曲线。

⭐ **预期结论（若成立即为主卖点）**：
**代理与 φ 的相关性随库增大而衰减** —— 也就是说，
**统计代理在小库上还凑合，恰恰在最需要它的大库上失效。**
这比「代理有偏」这个静态结论有力得多。

### B4 · 成对对照
78 题 knowledge-update **全部恰好 2 个证据** ⇒ 天然的成对对照组。
⭐ **成对场景下我们的判据应与现有方法一致，集合级场景（multi-session 49 题 ≥3）才拉开差距。**
这比单纯多跑一个数据集有说服力得多。

## 4. 工程

### 已核实的上游情况（源码级）

| 项 | 结论 | 出处 |
|---|---|---|
| ✅ **可拔除** | 注入内容完全由 `haystack_dates / haystack_session_ids / haystack_sessions` 三个平行 list 决定 ⇒ **数据层预处理即可，0 行改 harness** | `run_generation.py:75-77, 92-96` |
| ⚠️ **可重放存疑** | **全仓 `seed` 零命中**，只有 `temperature: 0` | 实测 grep |
| ❌ **无 checkpoint** | 输出文件名带时间戳，异常静默 `continue` | `run_generation.py:376-378` |
| judge | 按 question_type 分 5 套模板 + abstention 单独模板；后处理 `'yes' in resp.lower()` **很脆** | `evaluate_qa.py:24-43` |
| memory 后端 | 只有 long-context 直灌 + BM25/contriever/stella/gte，**零第三方 memory 系统** | `src/retrieval/` |

仓库规模：`src/` 共 **1937 行 / 15 个 py**。

⚠️ **官方已迁至 `longmemeval-cleaned`（2025/09 清洗）** —— 早前统计基于旧 repo，
**s/m 文件可能已不同，跑之前需重核**。

📌 **可选**：`mem0ai/memory-benchmarks` 的 `benchmarks/longmemeval/run.py`（1413 行）
自带 seed、续跑判断、异步并发，**比官方 harness 完善**，可考虑直接抄。

### 我们要写的（全部在 `ssv_scale/` 下，零侵入）
```
ssv_scale/
  PLAN.md            ← 本文件
  replay_check.py    ⭐ B1：flip rate 测量（go/no-go）
  build_units.py     session → Unit（id 用官方 session_id，稳定）
  backend.py         backend(text, tag, tasks) -> {tid: {"hard","soft"}}
  proxies.py         recall_freq / recency / hit_rate 埋点
  run_phi.py         调 SSV 估计器
  analyze_bias.py    B3 主图：corr(proxy, φ) vs 库规模
```

### ⭐ φ 引擎无需修改
接口同 Experiment A：`backend(text, tag, tasks) -> {task_id: {"hard": float, "soft": float}}`。
`RenderCachingEvaluator` 只做 render→hash→cache→backend。

### ⚠️ 必须自建缓存
上游无 checkpoint，而 Shapley 要反复评估重叠子集 ⇒
**`ScoreCache` 必须落盘持久化，否则成本失控。**

## 5. 已知风险（不藏）

1. 🔴 **B1 不通过整条线悬空** —— 见上，先做。
2. **代理字段上游没有** —— `recall_freq / last_used / hit_count` 现有 harness **普遍缺失**，
   需自行埋点，**应计入工程量**。
3. **成本**：`s` 档 61M token，`m` 档 ~603M。Shapley 要跑多个子集 ⇒ **乘数很大**。
   对策：φ 只在**受控子样本**上跑（比如 multi-session 的 49 题），不是全库。
4. **judge 很脆** —— `'yes' in resp.lower()` 会把 "yes, but actually no" 判对。
   **soft 分数不能直接用上游 judge，需自己实现并报告一致率。**
5. ⚠️ **上下文预算混淆项** —— 「删了 session 成绩反而更好」有两种解释：
   **它本身有害** vs **上下文短了模型表现更好**。不控制会把大量无辜记忆判成有害。
   **对策：用 SSV 现成的双算子** `ρ_del`（真删）/ `ρ_pad`（占位但内容置空），
   两者之差分离**内容价值**与**占位成本**。**这套在 memory 上比在 skill 上更有必要。**

## 6. 下一步

- [ ] **Step 0**：`replay_check.py`，测 flip rate（**go/no-go**）
- [ ] Step 1：`build_units.py` + oracle 档 multi-session 49 题的 φ
- [ ] Step 2：`proxies.py` 埋点（需要跑一遍检索器才有频次）
- [ ] Step 3：B3 主图
