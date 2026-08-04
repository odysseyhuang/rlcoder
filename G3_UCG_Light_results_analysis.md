# G3-UCG-Light Results Analysis

拉取状态：`git pull` 返回 `Already up to date.`，当前 `develop` 已是 GitHub 远端最新版本。

结果来源：`RLCoder/result_infer/G3_A*`。

## 实验结果

baseline 为未进行任何改动的 RLCoder；每行最高值加粗，并列最高值同时加粗。最后一列按 `max(A0...A5) - baseline` 计算。

| 数据集 | 指标 | baseline（RLCoder） | A0 base_ucm | A1 g2_best | A2 import | A3 api | A4 full | A5 weighted | 最优结果 - baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cceval_java | em | 26.1337 | 27.2090 | 27.2557 | 27.2557 | **27.3492** | **27.3492** | 27.3025 | +1.2155 |
|  | es | 66.1309 | 66.5680 | 66.7312 | 66.7429 | 66.7359 | **66.7461** | 66.6914 | +0.6152 |
|  | id_em | 35.3436 | 36.2319 | 36.1851 | 36.1851 | **36.2786** | **36.2786** | 36.2319 | +0.9350 |
|  | id_precision | 62.1821 | 62.7361 | 62.9800 | 62.9909 | 62.9874 | **63.0107** | 62.9117 | +0.8286 |
|  | id_recall | 59.8501 | 60.3009 | 60.4630 | **60.4864** | 60.4435 | 60.4669 | 60.4318 | +0.6363 |
| cceval_python | em | 30.2439 | 32.0826 | 32.1201 | **32.1576** | 32.0450 | 32.0826 | 32.0450 | +1.9137 |
|  | es | 73.5797 | 74.8675 | **74.8859** | 74.8795 | 74.8124 | 74.8128 | 74.8660 | +1.3062 |
|  | id_em | 41.3133 | 43.3771 | 43.6398 | 43.6773 | 43.7148 | **43.7523** | 43.5272 | +2.4390 |
|  | id_precision | 70.0833 | 71.5932 | 71.8428 | **71.8741** | 71.8293 | 71.8668 | 71.7148 | +1.7908 |
|  | id_recall | 67.3607 | 68.8719 | 69.0580 | 69.0955 | 69.0774 | **69.1150** | 68.9599 | +1.7543 |
| repoeval_api | em | 28.1250 | 28.5000 | 28.5000 | 28.5000 | **28.5625** | **28.5625** | 28.5000 | +0.4375 |
|  | es | 62.0938 | **63.1419** | 63.0706 | 63.0844 | 62.9744 | 62.9850 | 63.0250 | +1.0481 |
|  | id_em | 32.7500 | **33.3125** | **33.3125** | **33.3125** | **33.3125** | **33.3125** | 33.1875 | +0.5625 |
|  | id_precision | 65.2131 | **66.2896** | 66.1876 | 66.2004 | 66.1189 | 66.1316 | 66.1570 | +1.0765 |
|  | id_recall | 70.1672 | 71.4421 | **71.4856** | 71.4078 | 71.4296 | 71.3519 | 71.3941 | +1.3184 |
| repoeval_line | em | 21.2500 | 21.7500 | 22.0000 | 22.0000 | **22.0625** | 21.9375 | 21.6250 | +0.8125 |
|  | es | 50.6106 | 51.0075 | 51.4675 | 51.4706 | **51.5363** | 51.4406 | 50.9712 | +0.9257 |
|  | id_em | 29.7500 | 30.3125 | 30.6250 | 30.6250 | **30.6875** | 30.6250 | 30.1250 | +0.9375 |
|  | id_precision | 46.4984 | 47.0625 | 47.6685 | 47.6555 | **47.7708** | 47.6781 | 46.9648 | +1.2724 |
|  | id_recall | 65.8463 | 66.6735 | 67.3488 | 67.3238 | **67.4628** | 67.4378 | 66.4933 | +1.6165 |

baseline 数据来源：`RLCoder/result_infer/RLCoder_deepseekcoder_7b_crossfile_1536_infile_512/<dataset>/results.json`。EM、ES 只展示并参与计算括号前的主数值。

## Retrieval Diagnostics

按全部 8,004 个任务加权统计，A1/A2/A3/A4/A5 的 `graph coverage` 分别为 28.91%/29.82%/34.90%/35.54%/19.48%，`selected graph/task` 分别为 0.490/0.509/0.614/0.626/0.270。A3、A4 扩大了图候选覆盖，A5 则明显低于 A1；这里的 `graph coverage` 指至少有一个图候选进入最终上下文的任务比例。

## Paired Sample Analysis

总体均值的变化很小，需要看相同任务上的成败翻转。以 A1 为基线，A2/A3/A4/A5 的 EM `win/loss` 分别为 1/0、6/4、6/5、13/20，ID-EM `win/loss` 分别为 1/0、8/3、9/4、17/29，平均 ES 变化分别为 +0.0044、-0.0287、-0.0429、-0.1256。

关键条件统计：

1. A3 的 API 图块进入最终上下文的任务有 1,184 个，其中 EM 为 6 胜 4 负，净增 2 个 exact sample；但这些任务的 ES 合计下降 229，即 API 上下文更容易改变生成文本，未稳定提高整体字符串相似度。
2. A5 中 API 图块进入上下文的 207 个任务反而是 3 胜 0 负。A5 的总回退不是 API 边本身造成的，而是它牺牲了原 A1 的 same-file/identifier 覆盖。
3. 相比 A1，A5 有 850 个任务从“有图上下文”变成“无图上下文”，这些任务为 8 胜 17 负，净损失 9 个 exact sample。A5 全部任务最终净损失 7 个，主要损失几乎都来自这组覆盖退化。
4. RepoEval line 上 A5 相对 A1 是 0 胜 6 负；这 6 个损失任务全部失去了 A1 中被选中的图块。例如 `huggingface_diffusers/60` 在 A1 中选入 5 个 same-file 图块并 exact match，A5 将每个 seed 的邻居数从 2 降为 1 后没有选入任何图块，EM 从 1 变 0。
5. A4 相对 A3 只翻转 3 个任务（1 胜 2 负）。import/path 虽增加候选，但进入最终上下文的覆盖率只有 2.32%，且新候选会改变 API/same-file 候选在平坦 Top-K 中的竞争关系。

双侧精确符号检验中，A3 对 A1 的 6/4 翻转 `p=0.754`，A5 对 A1 的 13/20 翻转 `p=0.296`。RepoEval line 的 A5 方向性较强（0/6，未校正 `p=0.031`），但经过多数据集比较校正后仍不足以宣称稳定显著。当前结果应视为机制线索，而不是已经确立的总体收益。

## Why Performance Degrades

### Confirmed causes

1. **A5 并没有启用 gate。** `run_config.json` 中 `enable_context_gate=false`。A5 实际改动是 `neighbors_per_seed: 2 -> 1`、`max_expanded: 40 -> 30`、`api_call_query_only: false -> true`，并调整边权。此前把 A5 解释为“门控过强”是不准确的。
2. **A5 同时修改过多变量并削弱了有效基线边。** A1/A4 每个 seed 可取 2 个邻居，A5 只取 1 个；任务级 graph coverage 从 A4 的 35.54% 降到 19.48%，甚至低于 A1 的 28.91%。因此 A5 不能用于判断“加权是否有效”，只能说明这组联合配置破坏了覆盖。
3. **图分数没有进入最终重排。** `context_graph.py` 写入 `_ucm_graph_score`、`_ucm_graph_seed_rank` 和 `_ucm_graph_edge_distance`，但全项目没有任何其他读取位置。`Retriever` 只编码 `str(CodeBlock)`，也就是描述和代码文本。边类型、方向、距离和图分数只影响扩展池内部预筛选，进入 RLRetriever 后全部丢失。
4. **查询 API 不能直接连接 API 定义。** 当前 API 扩展必须先在 BM25 seed 中找到一个含相同 token 的块，再从这个 seed 跳到其他块；`query_api_tokens` 只用于过滤或加分，不会直接查询 API 索引。这使真正缺失、尚未 import、且没有被 seed 命中的内部 API 无法被召回。
5. **API 提取不是语法感知的。** 正则会把声明识别成调用，例如 `def load_model(path)` 和 Java 的 `public Foo build(...)` 会产生 `load_model`、`build` API token。限定名又依赖局部变量名（如 `client.load`），跨文件常不一致；退化到末尾 token（如 `load`）则过于宽泛。
6. **图扩展改变了候选集大小而没有保持固定预算。** A3/A4 将平均 retriever 输入从 77.72 增加到 96.72/97.83，但最后仍只选 Top-10。新增图块来自与 RLRetriever 训练分布不同的构造过程，且没有图特征辅助判断，容易把“词法相似但结构无关”的块排入上下文。

### Supported hypotheses

1. A3 的 `api_call_query_only=false` 在 query 与 seed API 无重合时会进行探索性扩展。它能发现未显式出现在左上下文中的 API，但也会引入与目标无关的 seed API；6 胜 4 负和 ES 下降同时出现，符合“召回提高、精度不足”。
2. import/path 边按模块名和文件路径 token 相交，缺少符号解析和实际 import target 绑定。其 2% 左右的最终覆盖说明它既没有形成强召回，也不足以单独承担跨文件语义连接。
3. same-file 边目前只按块位置取前后邻居，没有 AST scope、控制流方向或定义-使用关系。它在 A1 中总体有用，但减少邻居会直接漏掉有效块，盲目增加邻居又会引入预算竞争。

## Ten 2025+ Top-Venue Papers

以下 10 篇均为 2025 年正式发表，且任务属于代码生成或代码补全；不把 workshop 或仅有 arXiv 投稿计入这 10 篇。

| # | paper | venue | main idea | graph-related lesson for G3 |
|---|---|---|---|---|
| 1 | [Code Graph Model (CGM)](https://papers.neurips.cc/paper_files/paper/2025/hash/178ae4ba29022eb7bf509c2e27bc8ab8-Abstract-Conference.html) | NeurIPS 2025 Main | 构造 repo/package/file/class/function 等异构节点及 contains/calls/imports/extends 边；用连通子图 RAG、两阶段重排、图注意力和 Graph-to-Code 训练 | 不应把图退化为平坦候选列表；至少要保留边类型、连通性和层级，最终模型或 reranker 必须看到图信号 |
| 2 | [VerilogCoder](https://ojs.aaai.org/index.php/AAAI/article/view/32007) | AAAI 2025 | Task and Circuit Relation Graph 将任务、信号、状态转移和样例连接；按依赖执行子任务，并用 AST 波形回溯定位错误 | 图可以表示“完成目标所需的计划”，不只是代码相似性；可建立 completion intent -> API/type/definition 的任务关系图 |
| 3 | [Enhancing Project-Specific Code Completion by Inferring Internal API Information](https://ieeexplore.ieee.org/document/11096713/) | IEEE TSE 2025 | 用 API usage example 与功能语义构建知识库，再从初始 completion 推断尚未 import 的内部 API | 直接支持“先草拟、再查 API”；解决当前 query 只能经 seed 间接跳转、无法发现隐式 API 的问题 |
| 4 | [EpiCoder](https://proceedings.mlr.press/v267/wang25bi.html) | ICML 2025 | 用层次化 feature tree 合成训练数据，通过采样子树深度/宽度控制函数级到多文件级复杂度 | 可按节点层级、边类型和子图半径生成 graph-to-code/FIM 训练样本，逐步增加结构复杂度 |
| 5 | [Tree-of-Evolution](https://aclanthology.org/2025.acl-long.14/) | ACL 2025 Long | 在树上探索多条 instruction evolution 路径，并以质量反馈优化每一步 | 离线生成多种 context-plan 分支，保留能提高目标 completion 的路径作为正例，而不是手工固定边权 |
| 6 | [Alignment with Fill-In-the-Middle](https://aclanthology.org/2025.emnlp-main.419/) | EMNLP 2025 Main | 用 AST 划分结构完整的 FIM block，构造细粒度偏好对，并按 block 深度/长度做 curriculum | 将图节点从任意文本块升级到 AST 完整块；训练时可以对“选对依赖子图后补全中间块”进行局部奖励 |
| 7 | [RethinkMCTS](https://aclanthology.org/2025.emnlp-main.410/) | EMNLP 2025 Main | 在生成代码前搜索 thought tree，并用细粒度执行反馈修正错误路径 | 更适合搜索少量“上下文计划/子图”，而非穷举代码；执行或轻量语义反馈可用于修正错误边路径 |
| 8 | [AlignCoder](https://conf.researchr.org/details/ase-2025/ase-2025-papers/34/AlignCoder-Aligning-Retrieval-with-Target-Intent-for-Repository-Level-Code-Completio) | ASE 2025 Research | 生成多个候选 completion 增强 query，并用 RL 训练 retriever 对齐 target intent | 当前只看左上下文，语义缺口明显；draft 中的预测 API/type 应成为 query-to-graph 起点，重排器应由最终生成收益监督 |
| 9 | [Evaluating and Improving Framework-based Parallel Code Completion](https://conf.researchr.org/details/ase-2025/ase-2025-papers/216/Evaluating-and-Improving-Framework-based-Parallel-Code-Completion-with-Large-Language) | ASE 2025 Research | 将补全拆成插入点、框架选择、directive completion，并以 curriculum 逐步训练 | 不同边解决不同子任务；先学习 scope/插入位置，再学习 API/依赖边，最后联合，优于一次性打开全部边 |
| 10 | [SemGuard](https://conf.researchr.org/details/ase-2025/ase-2025-papers/193/SemGuard-Real-Time-Semantic-Evaluator-for-Correcting-LLM-Generated-Code) | ASE 2025 Research | 在生成过程中进行行级语义监督，发现偏离后回滚并重生成 | 可训练轻量 evaluator 判断新增图块是否让下一行语义偏离，用于软门控或重排，而非仅凭静态词频阈值 |

## Recommended Graph Design

### G3.5: low-cost, highest-priority

1. **保留 A1 的核心覆盖，只单变量加入新机制。** 固定 `neighbors_per_seed=2`、`max_expanded=40`，分别测试 rerank、draft API、direct API、connected subgraph；不要再用 A5 这种同时改 4-5 个变量的配置。
2. **让图特征进入最终分数。** 在 RLRetriever cosine score 后做归一化融合：

   `final = cosine + alpha*norm(graph_score) + beta*query_edge_match + gamma*source_prior - lambda*distance - mu*token_cost`

   `graph_score` 必须按任务、边类型归一化；先做小网格搜索，再考虑 pairwise/RL 训练。
3. **增加 direct query -> API retrieval。** API 知识库节点至少保存 signature、qualified name、defining file、功能描述和 1-3 个 usage example。用左上下文加 1-2 个廉价 draft completion 推断 expected API/type，直接检索 API 节点，不要求 seed 先命中。
4. **改为 typed connected subgraph。** 最小节点集合为 `file/class/function/block/API`，边为 `contains/calls/imports/extends/adjacent`。从 query/draft 命中的节点开始，优先保留到定义或 usage example 的最短连通路径，而不是独立追加散块。
5. **固定候选预算并给边类型配额。** 例如在总池 100 内预留 `same-file 12-16`、`API 12-16`、`identifier 4-8`、`import 2-4`，剩余给 base retrieval；未使用配额回流给 base。最终仍统一重排，但避免弱 import 候选挤掉已验证有效的 same-file 邻居。
6. **补充可归因 trace。** 保存每个最终 block 的 key、edge type、seed key、matched symbol、graph score、retriever score 和 token cost。当前只有来源计数，无法判断哪一条边真正导致生成翻转。

### G4: higher-cost research stage

1. 采用 CGM 风格的异构图输入：node encoder 压缩代码块，graph-aware attention 或 graph positional bias 保留结构。
2. 做 `Graph-to-Code/FIM` 训练：从连通子图重构 AST 完整 block，逐步增加子图深度、边类型和跨文件跨度。
3. 加入 noisy graph training：随机遗漏一个关键节点或加入无关节点，使模型学会抵抗当前实验中常见的漏召回与噪声候选。
4. 以最终 EM/ID-EM 或可执行测试反馈训练 graph reranker；如果没有测试，先用 gold target 的 identifier/API overlap 构造离线 pairwise supervision。

## Next Experiment Matrix

| run | base config | only change | purpose |
|---|---|---|---|
| G3.5-B0 | A1 | none | paired baseline |
| G3.5-R1 | A1 | graph-aware score fusion | 验证“图信号在最终重排中可见” |
| G3.5-Q1 | A1 | direct query -> API KB | 验证 seed-independent API 召回 |
| G3.5-Q2 | A1 | 1 draft + direct API KB | 验证 target-intent alignment |
| G3.5-S1 | A1 | typed connected 1-hop subgraph | 验证连通性与符号解析 |
| G3.5-F1 | A1 | R1 + Q2 + fixed edge quotas | 组合最有希望的轻量方案 |

主指标继续用四数据集 macro EM、ID-EM；同时必须报告 paired win/loss、图覆盖、各边的 selected-task win/loss 和置信区间。建议把“进入下一阶段”的最低门槛设为：macro EM 至少提升 0.10 个百分点，且四个数据集不出现超过 0.10 的单集回退；否则优先改召回和重排，不进入更重的 CFG/DDG 建模。

## Bottom Line

A3 仍是当前最好的轻量配置，但它相对 A1 只净增 2/8,004 个 exact sample，证据不足以说明 API 图已经稳定有效。A5 的退化主要来自缩减 same-file/identifier 邻居覆盖，不是 API 边，也不是 context gate。当前系统真正的瓶颈是：图只用于扩展候选，图的结构和分数没有被最终 retriever 或 generator 使用。

下一步最值得做的不是立即增加 CFG/CDG/DDG，而是保留 A1 覆盖，加入“draft 对齐的 direct API 检索 + graph-aware rerank + 固定边预算”。这同时得到 TSE 2025 的内部 API 推断、ASE 2025 AlignCoder 和 NeurIPS 2025 CGM 的直接支持，并且可以用较小改动验证核心假设。
