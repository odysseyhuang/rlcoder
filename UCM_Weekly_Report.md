# UCM Group Meeting Weekly Report

Date: 2026-07-09

## 1. 本周工作目标
进一步优化 UCM 统一候选池构建实验，并分析实验结果，明确下一步方向。

## 2. 当前实现状态

### 2.1 已实现部分

当前代码已经完成了 UCM 路线中的多路径候选池增强：

- 支持从不同视角构造检索 query，包括基础上下文、标识符、import/API、路径信息，以及可选 draft generation。
- 支持多路径 BM25 召回、候选合并、去重、source 标记和候选池大小控制。
- 支持增强 BM25，将代码内容、文件路径、代码块描述信息一起纳入检索，并对 identifier/camel case 等代码 token 进行更细粒度切分。
- 支持 UCM retrieval trace，能够记录每个样本不同来源的召回数量、合并后候选数量、最终 RLRetriever 选中的来源分布等。
- 支持简化版 context gate，主要用于限制 auxiliary/path-only 候选，以及在最终 retrieved blocks 中处理早停块。
- 已有 DeepSeekCoder-6.7B 条件下的完整评测结果，覆盖 CrossCodeEval Python/Java 与 RepoEval Line/API。

### 2.2 尚未实现或尚不完整部分

Plan 中最能体现 "Unified Context Modeling" 的图结构建模部分目前还没有真正落地：

- 尚未实现轻量上下文图 expansion。
- 尚未加入 same-file neighbor、identifier overlap、import/path relation 等图边。
- 尚未将图扩展候选与多路径候选统一交给 RLRetriever 重排。
- 尚未实现 source-aware prompt budget allocation。
- 当前 gate 是较简单的启发式过滤，没有实现 Plan 中更完整的 source-aware gate、graph-ratio 控制或 prompt token budget 分配。

因此，当前 UCM 实现更准确地说是：

> Enhanced BM25 + multi-path candidate recall + pretrained RLRetriever rerank.

它还不是完整的：

> Multi-path recall + lightweight context graph expansion + selective source-aware context packing.

## 3. 实验设置说明

### 3.1 公共实验条件

本次横向对比均使用同一主实验条件：

- generator: DeepSeekCoder-6.7B
- retriever: pretrained RLRetriever
- inference type: unixcoder_with_rl
- generator cross-file context budget: 1536
- generator total context length: 2048
- evaluated datasets:
  - CrossCodeEval Python
  - CrossCodeEval Java
  - RepoEval Line-level
  - RepoEval API-level

### 3.2 参数命名解释

下表中的配置名含义如下：

- `default`: 默认 UCM 小召回设置，基本对应 base query 使用原 RLCoder topK，auxiliary query 使用较小 topK。
- `t10/p120`: auxiliary query topK 为 10，候选池上限为 120。
- `t20/p140`: auxiliary query topK 为 20，候选池上限为 140。
- `base60/aux30/pool140`: base query topK 为 60，identifier/import 等 auxiliary query topK 为 30，候选池上限为 140。
- `path10/pool150`: 开启 path query，path query topK 为 10，候选池上限为 150。
- `enh`: 使用 enhanced BM25。
- `legacy`: 使用原始 BM25 tokenization 和 code-only index。
- `gate`: 开启当前简化版 context gate。

## 4. 多组实验结果横向对比

下表为 EM 结果，括号中为相对 Baseline RLCoder 的涨点。

| 配置 | CCE-python | CCE-java | Repo-line | Repo-api | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline RLCoder | 30.2439 | 26.1337 | 21.2500 | 28.1250 | 26.4382 |
| base+id, default | 30.5066 (+0.2627) | 26.1337 (+0.0000) | 21.3750 (+0.1250) | 28.3125 (+0.1875) | 26.5819 (+0.1438) |
| base+import, default | 30.5066 (+0.2627) | 26.1337 (+0.0000) | 21.3125 (+0.0625) | 28.3125 (+0.1875) | 26.5663 (+0.1282) |
| base+id+import, default | 30.5066 (+0.2627) | 26.1337 (+0.0000) | 21.3750 (+0.1250) | 28.3125 (+0.1875) | 26.5819 (+0.1438) |
| base+id+import+path, default | 30.5066 (+0.2627) | 26.1805 (+0.0468) | 21.1875 (-0.0625) | 28.3750 (+0.2500) | 26.5624 (+0.1242) |
| id+import, t10/p120 | 30.5066 (+0.2627) | 26.1337 (+0.0000) | 21.3750 (+0.1250) | 28.3125 (+0.1875) | 26.5819 (+0.1438) |
| id+import+path, t10/p120 | 30.5066 (+0.2627) | 26.1805 (+0.0468) | 21.1875 (-0.0625) | 28.3750 (+0.2500) | 26.5624 (+0.1242) |
| id+import, t20/p140 | 30.6191 (+0.3752) | 26.5545 (+0.4208) | 21.7500 (+0.5000) | 28.2500 (+0.1250) | 26.7934 (+0.3552) |
| id+import+path, t20/p140 | 30.5816 (+0.3377) | 26.6012 (+0.4675) | 21.7500 (+0.5000) | 28.2500 (+0.1250) | 26.7957 (+0.3575) |
| id+import, base60/aux30/pool140/enh | 32.0826 (+1.8387) | 27.2090 (+1.0753) | 21.6875 (+0.4375) | 28.5000 (+0.3750) | 27.3698 (+0.9316) |
| id+import+path, base60/aux30/path10/pool150/enh | 32.0450 (+1.8011) | 27.1622 (+1.0285) | 21.6250 (+0.3750) | 28.6250 (+0.5000) | 27.3643 (+0.9261) |
| id+import, base60/aux30/pool140/legacy | 29.3058 (-0.9381) | 25.5259 (-0.6078) | 21.5625 (+0.3125) | 28.7500 (+0.6250) | 26.2860 (-0.1521) |
| id+import+path, default+gate | 30.3565 (+0.1126) | 26.1805 (+0.0468) | 21.3750 (+0.1250) | 28.0625 (-0.0625) | 26.4936 (+0.0555) |
| test1_worse | 28.6679 (-1.5760) | 23.8429 (-2.2908) | 21.6250 (+0.3750) | 27.8750 (-0.2500) | 25.5027 (-0.9354) |

## 5. 重点配置的宏平均指标

下表选取 baseline、较强 UCM 配置、legacy 对照和 gate 对照。括号中为相对 baseline 的变化。

| 配置 | EM | ES | ID_EM | ID_F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline RLCoder | 26.4382 | 63.1038 | 34.7892 | 62.9487 |
| id+import, t20/p140 | 26.7934 (+0.3552) | 63.3921 (+0.2884) | 35.1904 (+0.4012) | 63.3769 (+0.4282) |
| id+import+path, t20/p140 | 26.7957 (+0.3575) | 63.4006 (+0.2968) | 35.1615 (+0.3723) | 63.3822 (+0.4335) |
| id+import, base60/aux30/pool140/enh | 27.3698 (+0.9316) | 63.8909 (+0.7872) | 35.7929 (+1.0037) | 63.9128 (+0.9641) |
| id+import+path, base60/aux30/path10/pool150/enh | 27.3643 (+0.9261) | 63.9113 (+0.8075) | 35.7945 (+1.0053) | 63.9142 (+0.9655) |
| id+import, base60/aux30/pool140/legacy | 26.2860 (-0.1521) | 63.2191 (+0.1154) | 34.4593 (-0.3299) | 63.0026 (+0.0540) |
| id+import+path, default+gate | 26.4936 (+0.0555) | 63.1269 (+0.0231) | 34.8032 (+0.0140) | 62.9478 (-0.0009) |

## 6. 改写后的关键趋势

### 6.1 多路径候选池有效，但默认设置收益有限

默认多路径配置只能带来约 +0.12 到 +0.14 的 macro EM 提升。这说明 identifier/import/path 等 query view 本身是有价值的，但如果召回池太小，RLRetriever 能选择的候选空间没有被充分打开。

### 6.2 主要收益来自 enhanced BM25 与更大的候选池

`base60/aux30/pool140` 与 `base60/aux30/path10/pool150` 两组 enhanced BM25 配置都能带来约 +0.93 的 macro EM 提升，并且 ID_EM、ID_F1 也同步提升约 +1.0。当前收益的核心不是继续增加 query 类型，而是把候选池做得更充分、更干净，让 pretrained RLRetriever 有足够好的候选可重排。

### 6.3 Path query 的收益具有数据集差异

加入 path query 后，RepoEval API 的 EM 有提升，但 CrossCodeEval 和 RepoEval Line 未表现出稳定优势。整体看，path query 更适合作为可选增强，而不是当前主配置的必要组成。

### 6.4 Enhanced BM25 是关键组件

同样是较大的召回设置，legacy BM25 的 macro EM 反而低于 baseline。这说明多路径路线不能只理解为“召回更多候选”，还依赖更适合代码场景的检索文本构造和 tokenization。文件路径、代码块描述、identifier/camel case 分词对候选质量有明显影响。

### 6.5 当前简化 gate 没有形成有效增益

现有 gate 配置只有约 +0.06 macro EM，低于不加 gate 的默认多路径配置，也明显低于扩大召回的 enhanced BM25 配置。当前 gate 更像保守过滤规则，还没有学到或规则化出稳定的上下文选择优势。

## 7. 关键结论

### 7.1 多路径候选池不建议继续作为主要死磕方向

从结果看，多路径候选池已经跑出了清晰趋势：

- 小召回设置有小幅收益。
- 扩大召回后收益明显。
- enhanced BM25 是必要条件。
- path query 不稳定。
- 当前 gate 不明显。

因此，继续围绕 `id/import/path/topK/pool/gate` 做大量排列组合，预计边际收益有限。

### 7.2 推荐固定的主配置

建议将后续主线实验固定在：

```bash
--enable_ucm
--enable_multi_path_retrieval
--ucm_base_topk 60
--ucm_topk_per_path 30
--ucm_candidate_pool_size 140
```

并保持：

- 启用 identifier query。
- 启用 import/API query。
- 不启用 path query 作为主配置。
- 使用 enhanced BM25。
- 不启用当前版本的 context gate。

对应实验配置可概括为：

```text
id+import, base60/aux30/pool140/enh, no path, no gate
```

该配置在四个公开数据集上的 macro EM 最高，且 CrossCodeEval Python/Java 表现最强，适合作为下一阶段图结构建模的固定底座。

### 7.3 Path 配置可作为补充对照

如果后续更关注 ES、ID_F1 或 RepoEval API，可以保留带 path 的强配置作为补充对照：

```text
id+import+path, base60/aux30/path10/pool150/enh, no gate
```

该配置 macro EM 与 no-path 配置几乎持平，但 ES、ID_EM、ID_F1 的宏平均略高。

## 8. 后续路线：从多路径候选池转向图结构建模

### 8.1 为什么可以结合图结构建模

当前 UCM 已经完成了一个较强的候选池底座，但它本质上仍是 query-based recall。它能找到 lexical/identifier/import 相似的候选，却不一定能补全结构上相邻或依赖相关的候选。

图结构建模正好补这个空缺：

- 多路径召回负责找到高相关 seed candidates。
- 图扩展负责补充 seed 周围的结构邻居。
- RLRetriever 负责对原始候选与图扩展候选统一重排。

这与 Plan 中的 UCM 叙事更一致：不同来源的上下文统一成候选池，再由 pretrained RLRetriever 评分选择。

### 8.2 建议优先实现的图边

第一阶段建议只做轻量、可控、低风险的图扩展：

1. same-file previous/next edge
   - 对召回到的代码块，补充同文件上一个和下一个自然代码块。
   - 目标是补齐被 natural block 切分打断的上下文。

2. identifier-overlap edge
   - 对共享稀有 identifier 的代码块建立关联。
   - 目标是补充同一 symbol、API、class、function 周围的相关实现。

3. import/path relation edge
   - 根据当前文件或候选块中的 import/module/path 信息，连接可能被依赖的文件或代码块。
   - 目标是提升跨文件 API、类、工具函数相关任务表现。

### 8.3 建议图扩展参数

建议初始参数保守设置，避免候选池噪声过大：

```bash
--enable_context_graph
--ucm_graph_max_seed 20
--ucm_graph_max_neighbors_per_seed 2
--ucm_graph_max_expanded 40
```

整体流程建议为：

```text
fixed UCM multi-path recall
-> candidate merge/dedupe
-> graph expansion from top seed candidates
-> merge/dedupe again
-> pretrained RLRetriever rerank
-> generator evaluation
```

### 8.4 建议下一阶段实验矩阵

建议下一阶段不要再大规模 sweep 多路径参数，而是固定候选池参数后做图结构 ablation：

| 实验 | 说明 |
| --- | --- |
| A0 | Baseline RLCoder |
| A1 | Fixed UCM multi-path: id+import, base60/aux30/pool140/enh |
| A2 | A1 + same-file previous/next graph |
| A3 | A2 + identifier-overlap graph |
| A4 | A3 + import/path relation graph |
| A5 | A4 + optional path-query comparison |

主要观察指标：

- EM
- ES
- ID_EM
- ID_F1
- stop/no-context rate
- average merged candidate count
- graph-expanded candidate hit rate
- RLRetriever final selected source distribution
- retrieval latency

### 8.5 预期收益位置

图结构建模更可能优先提升：

- RepoEval API
- RepoEval Line
- ID_EM
- ID_F1

原因是这些任务更依赖跨文件符号、API 调用、定义位置和相邻上下文。CrossCodeEval Python/Java 也可能提升，但预计主要收益会体现在 identifier/API 相关指标上。

## 9. 风险与控制

### 9.1 主要风险

- 图扩展可能引入噪声，稀释高质量 BM25/RLRetriever 候选。
- import/path 关系如果规则太宽，容易引入弱相关文件。
- 候选池过大可能增加 RLRetriever 延迟。
- 如果图候选没有清晰 source trace，后续难以分析涨跌原因。

### 9.2 控制策略

- 固定当前最强多路径底座，不再同时改变多路径参数和图参数。
- 图扩展从 top seed candidates 出发，而不是全仓库全图扩展。
- 严格控制 max seed、neighbors per seed 和 expanded candidates。
- 每一步都保留 source trace，区分 original multi-path candidates 与 graph-expanded candidates。
- 保持图扩展发生在 RLRetriever 前，让 pretrained RLRetriever 统一裁决候选质量。

## 10. 组会汇报口径

本周完成了 RLCoder + UCM 代码和实验结果的梳理。当前实现已经完成 multi-path candidate recall、enhanced BM25、trace 和简化 gate，但还没有实现 Plan 中的 graph expansion 与 source-aware prompt budget。

实验上，多路径候选池证明是有效的，但收益主要来自 enhanced BM25 和更大的候选池。默认多路径只有小幅提升，当前 gate 没有明显收益，path query 不是稳定主增益。最推荐的主配置是 `id+import, base60/aux30/pool140/enh, no path, no gate`，该配置四数据集 macro EM 提升约 +0.93。

下一阶段不建议继续大量调多路径参数，而应固定当前最强候选池底座，转向 lightweight context graph expansion。预期重点提升 RepoEval 和 identifier/API 相关指标，让 UCM 从 query-based retrieval augmentation 进一步推进到真正的结构化上下文建模。

## 11. 本轮后续更新：轻量图结构建模初版

在上述分析之后，已按推荐路线补充了轻量 context graph 的初版实现。当前实现将多路径候选池固定在主配置：

```text
id_import_nopath_base60_aux30_pathk5_pool140_enhbm25_nogate
```

并在多路径召回、候选合并之后，RLRetriever 重排之前，增加 graph expansion 阶段。初版支持：

- same-file previous/next edge，默认随 `--enable_context_graph` 开启；
- identifier-overlap edge，通过 `--ucm_graph_enable_identifier_edges` 开启；
- import/path relation edge，通过 `--ucm_graph_enable_import_edges` 开启；
- graph 扩展候选数量控制，包括 seed 数量、每个 seed 的邻居数、每个样本最大扩展候选数；
- graph trace 字段，用于记录每个样本扩展出的候选数量、graph source 分布和 graph 后候选池大小。

建议下一轮实验从以下配置开始：

```bash
--enable_ucm
--enable_multi_path_retrieval
--ucm_base_topk 60
--ucm_topk_per_path 30
--ucm_candidate_pool_size 140
--enable_context_graph
--ucm_graph_max_seed 20
--ucm_graph_max_neighbors_per_seed 2
--ucm_graph_max_expanded 40
```

然后逐步加入：

```bash
--ucm_graph_enable_identifier_edges
--ucm_graph_enable_import_edges
```

目前该 graph 初版已完成静态编译和无模型烟测，但还没有完成 DeepSeekCoder-6.7B 全量 GPU 评测。后续需要按 A1/A2/A3/A4 消融矩阵验证 graph expansion 是否能在 RepoEval、ID_EM 和 ID_F1 上带来进一步收益。
