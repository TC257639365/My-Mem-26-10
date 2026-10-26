# Relation Vector Lab

## 1. 这个目录做什么

`relation_vector_lab` 是第一阶段关系向量化实验的统一代码目录。它不是一套新的完整记忆系统，而是
一台可逐步增加模块的实验仪器：同一份数据、同一个 Encoder、同一套评分代码下，依次打开普通检索、
简单改写、独立 Factor、共享 Relation Pattern 和软并集。

目录名没有使用 `stage1`，避免与项目过去的 stage1 实验、当前文档中的“第一阶段”以及 T-Mem 自身
的 stage 编号混淆。

直接指导文档：

```text
docs/stage3/5-第一阶段最小实现安排.md
```

背景规划：

```text
docs/stage3/4-构建与检索的最小实现大致规划2.md
```

## 2. 它在整体工作流中的位置

整体方案希望把 Memory 表达成潜在影响来源，把 Query 表达成目标需求，再通过可复用关系找到普通
语义检索容易遗漏的候选。这个目录只负责 Candidate Discovery：

```text
Memory/Query 数据
    -> 多种无泄漏表示
    -> 固定 Encoder
    -> raw / Factor / Registry 多路检索
    -> 候选、分数与解释 trace
    -> 审核感知评测和误差地图
```

它不判断实际影响是否成立，不执行 Evidence Verification，也不生成最终答案。

与整体规划的对应关系：

| 整体阶段 | 本目录当前实现 |
|---|---|
| S0：数据、基线与统一评测 | 已接入 auxiliary_v2、预检、keyword/dense、统一结果与指标 |
| S1：固定 Encoder 双侧文本 | 已建立 P1/P2 profile 和多 view max 匹配 |
| S2：固定 Registry 与软路由 | 已建立 P3 automatic/oracle/ablation 和 P4 rank union |
| S3 以后：训练式模型 | 尚未实现，只预留稳定的数据与 trace 边界 |

## 3. 第一阶段实验变体

这些不是多套互不相干的系统，而是一条流水线上的 profile：

| Profile | 增加的能力 | 当前实现状态 |
|---|---|---|
| P0-K `p0_keyword` | 原始 Memory 与 Query 的关键词匹配；不使用 Encoder | 已实现 |
| P0-V `p0_vector` | 原始 Memory 与 Query 的向量检索基线 | 已实现 |
| P1-S `p1_subject` | 只把 Memory 换成简短 subject，观察压缩是否有帮助 | 已实现 |
| P1-R `p1_rewrite` | 只把 Memory 换成普通改写，作为角色化表示的控制组 | 已实现，405条 Memory 完整覆盖 |
| P1-Q `p1_query_expansion` | 只扩展 Query，观察一般性补充信息的作用 | 已实现，80个 Query 完整覆盖 |
| P1-T `p1_prospective_trigger` | trigger flat 对照 | 已实现 profile，当前不强制补数据 |
| P2 `p2_factor` | 独立的 Source Factor 与 Target Requirement 双侧匹配 | 已实现，双侧辅助表示完整覆盖 |
| P3 `p3_registry_auto` | 两侧自动路由到共享 Relation Pattern 后匹配 | 已实现，含合法 abstain 和条件 trace |
| `p3_registry_oracle_m` | 只向 Memory 侧提供正确 Pattern | 已实现，用于定位 Memory 构建瓶颈 |
| `p3_registry_oracle_q` | 只向 Query 侧提供正确 Pattern | 已实现，用于定位 Query 激活瓶颈 |
| `p3_registry_oracle_both` | 两侧都提供正确 Pattern，估计 Registry 上限 | 已实现，仅作诊断 |
| `p3_registry_swap` | 交换 Pattern 的 Source/Target 描述 | 已实现，用于检查非对称角色是否生效 |
| `p3_registry_shuffle` | 打乱 Pattern 描述 | 已实现，用于检查结构是否优于偶然文本相似 |
| P4 `p4_union` | 合并 raw、Factor、Registry 的候选名次，观察互补召回 | 已实现 |

P4会自动计算它依赖的 P0-V、P2和P3结果，但最终仍分别保存 route rank，不学习融合权重。

## 4. 当前代码结构

```text
relation_vector_lab/
├── contracts.py       # Memory、Query、view、Binding、结果 trace 的公共契约
├── datasets.py        # stage3_test1、auxiliary_v2 与旧格式兼容适配
├── text.py            # 确定性多语分词和 view 序列化
├── embeddings.py      # hashing smoke backend 与 OpenAI-compatible Encoder
├── profiles.py        # P0～P4 的声明式配置
├── preflight.py       # 数据覆盖检查和辅助信息补齐 worklist
├── scoring.py         # keyword、dense、Registry、rank union
├── evaluation.py      # audit 指标、profile rank delta、反事实 margin 和 P4互补性
├── runner.py          # 统一运行及 artifact 输出
├── cli.py             # preflight / run 入口
└── tests/             # 本目录自身的轻量测试
```

## 5. 数据隔离与默认行为

代码固定三条输入边界：

```text
Memory-only -> Source Factor / Binding / Memory index
Query-only  -> Requirement / Target Pattern / Query vector
Pair-aware  -> counterfactual label / evaluation only
```

Relation pair 中的 Pattern 和 conditions 不会被数据 loader 回填到 Memory 索引。Memory 辅助表示通过
`source_memory_id -> canonical_memory_id` 映射到唯一正文，避免同一 Memory 因多次出现而重复构建。

默认运行会检查 profile 所需辅助信息是否完整：

- P0可直接运行；
- P1～P3覆盖不足时直接报错，并给出缺失数量；
- `--allow-incomplete` 只用于诊断，会将缺失表示记录为 abstain，不能作为正式结果。

## 6. 当前数据与代码就绪状态

默认路径现已指向 `results/stage3_test1/auxiliary_v2`。预检和端到端测试得到：

```text
Controlled Pool 唯一 Memory：405
Memory-side representation / Binding：405 / 405
全部 Query：80
Query Requirement / Binding：80 / 80
Relation Pattern：10
legacy mixed condition：0
```

loader会保留输入隔离、审核状态、entity/time scope、hard/open condition 和 Registry 的 Join
Constraint。P3 trace 明确把 Join Constraint 标为“仍待联合验证”，不会把 Candidate Discovery 写成
影响已经成立。装载时还会检查 Memory/Query 内容哈希、重复辅助记录和未知 Pattern 引用，避免正文或
Registry 变化后静默复用旧表示。

代码侧已经修正两个会影响实验结论的问题：

1. `score=None` 的 abstain 候选不再进入 Hit/Recall/MRR 的 Top-k，避免按 ID 排序产生虚假命中；
2. P4 的 readiness 会递归检查 P0-V、P2、P3，不再出现“P4显示 ready、依赖项却缺数据”的情况。

当前自动测试包含 auxiliary_v2 覆盖检查、P0～P4端到端运行、P3条件 trace 和全
abstain 指标检查。hashing backend 下13个本轮必需 profile 已完成80个 case 的烟雾运行；这只证明代码
链路可运行，不代表模型效果。

后续正式数据按“Memory、Query及辅助文本全部为英文”处理。这让
`bge-small-en-v1.5` 和 `all-MiniLM-L6-v2` 可以作为有意义的英文 Encoder 对照；每次重建冻结数据后仍
应先检查 validation report，不能只依赖这一语言假设。

## 7. 默认实现决策及原因

### 固定平面检索

当前候选规模很小，优先使用精确评分，不引入 ANN 或向量数据库。这样可以把变化归因于表示，而不是
索引近似。

### 多 view 使用 max

P2对预算内的 Memory 与 Query view 计算全部组合，取最高相似度，并在 trace 中保存命中的具体两端。
runner默认每侧只取1个 view，使 P1/P2/P3 的首轮比较具有相同预算；随后可用 `--view-budget 3` 做
预算曲线，或用 `--view-budget 0` 诊断全部 view。预算内采用 max，是“少数重要关系不应被平均冲淡”
的最简默认版本。

### Registry 使用两侧适配的瓶颈分数

P3分别计算 Memory 对 Pattern Source、Query 对 Pattern Target 的适配度，再取较低一侧；多个共享
Pattern 取最高分。它容易解释，但只是第一版近似，不代表最终关系公式已经确定。

### 允许 abstain，不设 hard gate

没有 Pattern 或缺少 view 时，relation route 可以不出分，但不能删除 raw 候选。T-Mem 实验中
Entity/Bridge 的0.85 gate几乎不触发，因此本实现不预设类似门槛。

### P4只做无训练 rank union

不同 route 的分数分布并不相同，第一版不直接相加或相乘。P4只取各 route 的 Top-r 并集，以最佳
reciprocal rank 排序，同时保留所有 route rank。

### hashing只用于 smoke test

默认CLI使用 deterministic hashing，目的是让任何环境都能验证代码链路。正式实验必须使用该轮指定
并冻结的真实 Encoder 服务，不能将 hashing 指标写入方法结论。

## 8. 正式实验工作流

模型生命周期分成两层：Encoder 独立启动并在多轮实验间复用；每条实验命令只启动一个指定的 Qwen
基座模型，依次跑完全部13个 profile，然后自动关闭该 Qwen。当前 P0～P4的分数仍只由 Encoder
产生，Qwen 仅用于连通性检查和保持与 T-Mem 实验组织一致，不能把三个 Qwen 目录视作三个独立统计
样本。

### 8.1 离线预检

以下命令检查冻结数据、全部 profile 以及模型路径，不启动模型：

```bash
bash My-Mem/results/stage3_test1/My-Mem-test/scripts/run_first_stage_local.sh \
  --loop 1 --encoder bge-m3 --base-model qwen3-4b \
  --encoder-gpu 2 --llm-gpu 0 --validate-only
```

### 8.2 启动并复用一个 Encoder

例如在 GPU 2 启动 BGE-M3：

```bash
bash My-Mem/results/stage3_test1/My-Mem-test/scripts/manage_encoder.sh start \
  --encoder bge-m3 --gpu 2 --port 8001
```

可选 Encoder 为 `bge-m3`、`bge-small-en-v1.5`、`all-minilm-l6-v2`、
`qwen3-embedding-0.6b` 和 `gte-modernbert-base`。查看状态或发送一次真实向量请求：

```bash
bash My-Mem/results/stage3_test1/My-Mem-test/scripts/manage_encoder.sh status \
  --encoder bge-m3 --port 8001

bash My-Mem/results/stage3_test1/My-Mem-test/scripts/manage_encoder.sh probe \
  --encoder bge-m3 --port 8001
```

本机实测 `bge-small-en-v1.5` 与 `all-minilm-l6-v2` 均可由同一管理脚本启动，并能通过
OpenAI-compatible `/embeddings` 接口返回384维向量；临时测试服务已正常关闭。测试记录保存在
`My-Mem-test/services/connectivity/`。

### 8.3 一个基座模型跑完整轮次

一条命令会让指定基座模型连续运行全部 profile，完成或失败后都尝试关闭该基座模型；Encoder 保持
运行。不同基座模型建议使用不同 loop，使轮次语义和恢复操作保持简单：

```bash
bash My-Mem/results/stage3_test1/My-Mem-test/scripts/run_first_stage_local.sh \
  --loop 1 --encoder bge-m3 --base-model qwen3-4b \
  --encoder-gpu 2 --llm-gpu 0

bash My-Mem/results/stage3_test1/My-Mem-test/scripts/run_first_stage_local.sh \
  --loop 2 --encoder bge-m3 --base-model qwen3-8b \
  --encoder-gpu 2 --llm-gpu 0

bash My-Mem/results/stage3_test1/My-Mem-test/scripts/run_first_stage_local.sh \
  --loop 3 --encoder bge-m3 --base-model qwen3-4b-instruct-2507 \
  --encoder-gpu 2 --llm-gpu 0
```

结果位于 `My-Mem-test/runs/<encoder>/loop_<NNN>/<base_model>/`。按 Encoder 分区后，更换 Encoder
不会与 BGE-M3结果混在一起。此前已完成的 `runs/loop_001/p0_keyword/` 保留为旧流程历史结果；新流程
不会覆盖或读取它，并会在完整轮次中重新计算 P0-K。

失败后需要复用同一个目录时可增加 `--resume`。它会重跑尚未形成 `_SUCCESS.json` 的整轮，不会把旧
的单 profile 结果拼接进来。

### 8.4 查看进度与报错

当前轮次、Encoder、基座模型、profile、处理 case 和大致比例会同时打印到终端并写入文件。另一个
终端可以运行：

```bash
python My-Mem/results/stage3_test1/My-Mem-test/scripts/monitor_progress.py \
  --encoder bge-m3 --loop 1 --base-model qwen3-4b
```

也可以直接查看 `progress.log`。Python异常和 traceback 保存在 `errors.jsonl`，模型服务输出保存在
`service_logs/`。

### 8.5 关闭 Encoder

全部轮次结束后显式关闭：

```bash
bash My-Mem/results/stage3_test1/My-Mem-test/scripts/manage_encoder.sh stop \
  --encoder bge-m3 --port 8001
```

管理脚本只会关闭自己记录且进程命令仍匹配的服务，不会结束占用端口的未知进程。

## 9. 开发与烟雾测试

无需模型服务时，可用 hashing backend 检查代码链路：

```bash
cd My-Mem
python -m relation_vector_lab.cli run \
  --profiles p0_keyword,p0_vector,p1_subject,p1_rewrite,p1_query_expansion,p2_factor,p3_registry_auto,p3_registry_oracle_m,p3_registry_oracle_q,p3_registry_oracle_both,p3_registry_swap,p3_registry_shuffle,p4_union \
  --embedding-backend hashing
python -m pytest -q relation_vector_lab/tests
```

正式启动器固定使用 `model-env`。`--allow-incomplete` 只用于数据缺失诊断，正式运行不要开启。

## 10. 运行产物

每次运行生成：

```text
run_manifest.json       输入 hash、Encoder、profile、运行时间与实际向量化文本数
preflight_report.json   数据覆盖、view 数量、审核状态、legacy condition 和数据警告
retrieval_results.jsonl 每个 Query 的完整排序与 route trace
metrics_summary.json    audit-aware 指标、rank delta、反事实 margin 和 P4互补性
```

## 11. 尚未完全决断的问题

以下问题采用了默认简练版本，并应在第一轮结果出来后重新审查：

1. **Registry compatibility 公式**：当前使用 `min(source_fit, target_fit)`；若 oracle 也无法区分正反例，
   应先修表示或条件结构，而不是直接增加复杂打分器。
2. **多 view 聚合**：当前为 max；需通过 trace 检查是否经常被错误 Factor 劫持，再决定是否测试 mean
   或 log-sum-exp。
3. **Pattern oracle 范围**：Deep 没有 Gold Pattern，oracle profile主要解释 Relation 40，不应把 Deep
   的 automatic fallback 当 oracle 结果。
4. **P2的独立性**：现有 Factor 文本没有 Pattern ID/name，但生成时看过公开 Registry。当前对比只能
   作为探索性证据；若要做严格归因，需要另生成一版完全不读取 Registry 的 Factor。
5. **人工审核**：auxiliary_v2 的 Binding 仍是 `reviewed_candidate_pending_human`。代码会保留并输出该
   状态，但无法代替 Gate C 人工审核。
6. **改写质量变量**：即使统一为英文，部分 rewrite 仍可能与原文相同。正式报告应区分
   identity/non-identity；当前 evaluator 尚未自动输出该切片。
7. **规模实验**：当前仓库只有 Controlled Pool。代码使用平面精确评分，能够读取候选数不同的同形
   test case，但100/1,000/5,000候选池尚未构建，无法完成规模退化结论。
8. **T-Mem对照**：第一轮引用已有 T-Mem Scene/Horizon；只有需要同索引、同向量预算比较时，才补齐
   `p1_prospective_trigger`。
9. **训练接口**：普通双编码器、Type投影和 latent Factor 尚未实现。当前契约刻意保存 side、view、
   Pattern 和 trace，方便后续在不改评测器的情况下接入。

## 12. 下一步

1. 冻结 auxiliary_v2、Registry、BGE-M3模型版本和运行参数；
2. 使用 BGE-M3复现 P0并对齐 T-Mem flat；
3. 按 P1→P2→P3→P4依次运行，不同时改变多个模块；
4. 根据逐候选 trace 形成人工审核队列和误差地图；
5. 若 Controlled Pool 出现稳定信号，再构建扩大候选池和严格无 Registry 的 P2 Factor；
6. 再决定是否补训练数据并进入普通双编码器与非对称 Type/Factor 投影实验。

