# G3-UCG-Light 实验设置说明书

## 1. 实验目标

G3-UCG-Light 的目标不是一步做到最强指标，而是验证当前 UCM + G2 轻量图能否扩展成更完整的 Unified Context Graph，并判断是否值得进入后续 CFG/CDG/DDG 语义图阶段。

核心问题：

1. 加入 import/path/API/call-name 等轻量结构边后，是否能提高有效上下文召回。
2. 图扩展出的候选是否真的进入最终 retrieved context。
3. 改善或退化集中在哪类任务：API 补全、line 补全、Python、Java、跨文件依赖。
4. 如果最终 EM/ES 没有明显提升，原因是图无效，还是 reranker、prompt、context budget 没有消费好图信息。

统一上下文建模叙事：

> 将文本检索上下文、同文件结构上下文、标识符上下文、导入/路径上下文、API/call-name 上下文统一建模为异构图，并用于候选扩展、重排前候选池构造和最终上下文选择。

## 2. 引入的轻量图结构

当前 G2 已有：

```text
candidate block node
same_file edge
identifier overlap edge
```

G3 建议扩展为：

```text
block node
file/path token node, optional implicit
identifier token node, optional implicit
import/api token node, optional implicit
```

实际实现可以不显式建 token node，继续沿用 inverted index，但语义上要区分边类型。

新增或保留边类型：

```text
graph_same_file
graph_identifier
graph_import_path
graph_api_call
```

边含义：

- `graph_same_file`：同一文件中目标候选块附近的前后块。保留当前 G2 逻辑。
- `graph_identifier`：候选块与 query/seed 共享低频标识符。保留当前 G2 逻辑，建议继续使用 query-only 或低 df 过滤，减少噪声。
- `graph_import_path`：根据 query 左上下文和 seed block 中的 import/package/module/path token，连接到文件路径、模块名、包名相似的候选块。
- `graph_api_call`：从 query 和 seed block 中抽取函数调用、方法调用、类名/API 名，连接到定义或使用相同 API 名的候选块。轻量版不做完整 call graph，只做 name-based API/call relation。

API/call token 示例：

```python
foo.bar(x)
BarConfig(...)
load_model(...)
```

可抽取：

```text
foo.bar
bar
BarConfig
load_model
```

Java 示例：

```java
builder.setName(...)
new FooClient(...)
FooClient.create(...)
```

可抽取：

```text
setName
FooClient
create
FooClient.create
```

## 3. 需要改动的代码位置

主要改动集中在：

```text
RLCoder/context_graph.py
RLCoder/main.py
RLCoder/scripts/submit_ucm_graph_g2_ablation.sh 或新增 G3 脚本
```

可能涉及：

```text
RLCoder/context_candidates.py
```

优先避免大改 `context_candidates.py`，先把新增能力收敛在 `context_graph.py` 和实验脚本中。

## 4. 具体代码改动设计

### 4.1 增加 API/call token 抽取

在 `context_graph.py` 新增函数：

```python
def _extract_api_call_tokens(text):
    ...
```

抽取规则保持轻量，不依赖 AST：

```text
name(
obj.name(
ClassName(
new ClassName(
ClassName.staticMethod(
```

过滤规则：

```text
长度 <= 2 丢弃
语言关键字丢弃
弱标识符丢弃
高 df token 丢弃
```

建议保留两类 token：

```text
simple token: method
qualified token: obj.method / Class.method
```

### 4.2 建 API/call index

在 `ContextGraphIndex.__init__` 增加：

```python
self.api_call_index_by_task = {}
self.api_call_df_by_task = {}
```

新增：

```python
def _ensure_api_call_index(self, task_id):
    ...
```

逻辑类似 `_ensure_identifier_index`，对每个 code block 抽取 API/call token，并建立 token 到 block index 的倒排表。

### 4.3 增加 API/call 邻居搜索

新增：

```python
def _api_call_neighbors(self, task_id, seed, max_neighbors, query_api_tokens, args):
    ...
```

排序信号：

```text
共享 API token 数
query API token bonus
低 df bonus
seed rank decay
```

source 设置为：

```text
graph_api_call
```

### 4.4 在图扩展中接入 API/call 边

新增参数：

```text
--ucm_graph_enable_api_call_edges
--ucm_graph_api_call_weight
--ucm_graph_api_call_max_df
--ucm_graph_api_call_query_only
--ucm_graph_query_api_bonus
```

在 `ContextGraphIndex.expand()` 中加入：

```python
query_api_tokens = _extract_api_call_tokens(example.left_context)

if enable_api_call:
    neighbors.extend(
        self._api_call_neighbors(...)
    )
```

trace 增加：

```text
graph_query_api_call_count
graph_sources.graph_api_call
graph_proposed_sources.graph_api_call
```

### 4.5 正式启用 import/path 边

当前已有：

```text
ucm_graph_enable_import_edges
_import_neighbors()
```

G3 要补两点：

1. 实验脚本里增加开启 import 边的实验组。
2. 确认 `_extract_import_tokens()` 和 `_path_tokens()` 对 Python/Java 都可用，必要时加强 Java package/import 解析。

### 4.6 更新 run_config 输出

`main.py` 已经会记录 graph 配置。新增 API/call 参数后，需要加入 run_config：

```python
"api_call_weight": ...
"api_call_max_df": ...
"api_call_query_only": ...
"query_api_bonus": ...
```

## 5. 实验组设计

建议 G3 先做 6 组，不要一次铺太多。

### G3_A0_base_ucm

不启用 context graph。

作用：对照 UCM 多路径检索本身。

```text
enable_ucm
enable_multi_path_retrieval
disable_context_graph
```

### G3_A1_g2_best_like

复现当前 G2 中相对合理的设置。

```text
same_file + identifier
identifier_query_only = true
identifier_max_df = 10 或 20
```

作用：G3 的图基线。

### G3_A2_import_path

在 A1 基础上加入 import/path 边。

```text
same_file + identifier + import_path
```

作用：验证 import/path 是否补足跨文件模块关系。

### G3_A3_api_call

在 A1 基础上加入 API/call-name 边。

```text
same_file + identifier + api_call
```

作用：验证 API/call 名是否对 RepoEval API、CCEVAL 有帮助。

### G3_A4_full_light

完整轻量统一图。

```text
same_file + identifier + import_path + api_call
```

作用：验证轻量统一图整体信号。

### G3_A5_full_light_gate_or_weighted

在 A4 基础上做简单自适应权重或 gate。

建议规则：

```text
如果 query 中 import/API token 多，提高 import/api 权重
如果 query identifier 少，降低 identifier 边权重
如果候选块只来自图边且无 query overlap，限制数量
```

作用：验证退化是否来自噪声候选过多。

## 6. 推荐默认参数

先用保守参数，避免图候选淹没 retriever：

```text
ucm_graph_max_seed = 20
ucm_graph_max_neighbors_per_seed = 2
ucm_graph_max_expanded = 40
ucm_graph_same_file_direction = both
ucm_graph_identifier_query_only = true
ucm_graph_identifier_max_df = 10
ucm_graph_enable_import_edges = true/false by group
ucm_graph_enable_api_call_edges = true/false by group
ucm_graph_import_weight = 1.4
ucm_graph_api_call_weight = 1.6
ucm_graph_identifier_weight = 1.2
ucm_graph_same_file_weight = 1.0
ucm_graph_query_overlap_bonus = 2
ucm_graph_query_api_bonus = 2
```

如果 A4 噪声明显，再降：

```text
ucm_graph_max_neighbors_per_seed = 1
ucm_graph_max_expanded = 30
```

## 7. 结果指标

不要只看最终 `results.json` 的 EM/ES。G3 必须加三层指标。

### 7.1 最终生成指标

继续看：

```text
EM
ES
ID-EM
ID precision
ID recall
```

按 benchmark 分开：

```text
cceval_python
cceval_java
repoeval_api
repoeval_line
github_eval
```

如果 `github_eval` 没有标准答案，则只分析 trace 和 prediction。

### 7.2 检索/图信号指标

从 `ucm_retrieval_trace.jsonl` 汇总：

```text
avg graph_expanded
graph_sources 分布
graph_proposed_sources 分布
candidate_pool_size_after_graph
retrieved_sources 分布
graph candidates selected by retriever ratio
stop_rank 分布
```

关键指标：

```text
graph_selected_ratio =
(final retrieved graph blocks) / (final retrieved non-stop blocks)

api_call_selected_ratio =
(final retrieved graph_api_call blocks) / (final retrieved non-stop blocks)

import_selected_ratio =
(final retrieved graph_import blocks) / (final retrieved non-stop blocks)
```

如果某条边 proposed 很多但 selected 很少，说明 retriever 不认可它或边噪声高。

### 7.3 Oracle recall / upper-bound 指标

建议新增或离线分析：

```text
gold target 是否出现在 BM25/UCM candidate pool
gold target 是否出现在 graph-expanded pool
gold target 是否进入 final retrieved top-k
```

分三段看：

```text
pre_graph_recall
post_graph_recall
final_retrieval_recall
```

判读：

```text
post_graph_recall 上升但 final_retrieval_recall 不升：问题在 reranker。
post_graph_recall 不升：问题在图边设计。
final_retrieval_recall 上升但 EM/ES 不升：问题可能在 prompt/context budget/generator。
```

## 8. 分析维度

### 8.1 先看是否有正信号

不要只问全局 EM 是否上涨，而要看：

```text
A2 是否改善 API/跨文件样本
A3 是否改善 API completion
A4 是否比 A1 有更多有效图候选进入 final context
A5 是否缓解 A4 的噪声退化
```

### 8.2 再看退化来源

常见退化原因：

```text
图边噪声高：post_graph pool 变大，但 final selected 或 EM 下降
上下文预算竞争：图块挤掉原本有效 base/import_api 块
retriever 不理解图结构：oracle recall 上升，但 final top-k 不升
prompt 消费差：final retrieved 有正确依赖，但生成没改善
```

### 8.3 按任务类型拆桶

建议至少拆：

```text
API-like completion
same-file local completion
cross-file completion
identifier-heavy completion
import-heavy completion
short left-context
long left-context
Python
Java
```

合理预期：

```text
import/path 对 import-heavy、cross-file 有信号
api_call 对 API-like、RepoEval API 有信号
same_file 对 local/line completion 有信号
identifier 对 identifier-heavy 有信号
```

## 9. 进入下一阶段的判据

### 9.1 可以进入 G4-SemanticGraph

满足以下任一条件：

```text
A4 或 A5 在至少两个 benchmark/任务桶上稳定优于 A1
post_graph_recall 明显提升
api/import/call 图块进入 final context 后带来局部收益
```

### 9.2 先做 G3.5 graph-aware rerank/gate

出现以下情况：

```text
post_graph_recall 提升
但 final retrieved recall 或 EM/ES 不提升
```

这说明图找到了东西，但现有 retriever/top-k/prompt 没用好。

### 9.3 暂缓继续加 CFG/DDG/CDG

出现以下情况：

```text
post_graph_recall 不提升
graph selected ratio 很低
新增边几乎不进入 final context
所有任务桶都退化
```

这时应先修图边质量、token 过滤、候选粒度，而不是加更复杂的程序分析。

## 10. 推荐最终叙事

G3 如果有信号，论文故事可以这样推进：

> G2 证明多路径检索可以扩大上下文来源，但上下文之间仍是松散候选集合。G3 将候选代码块组织为轻量异构上下文图，统一建模文件邻近、标识符、导入路径和 API 调用关系，使上下文选择从“相似文本排序”转向“关系约束下的上下文建模”。

后续自然承接 G4：

> 在轻量统一图有效后，进一步引入 GraphCoder/DraCo 风格的 CFG/CDG/DDG/call/dataflow 语义边，形成更完整的仓库级统一上下文建模。
