# UCM 参数配置说明

配置名：

```text
id_import_nopath_base60_aux30_pathk5_pool140_enhbm25_nogate_enh_id_import
```

该配置表示：使用 UCM 多路径候选池召回，候选来源包括 **base query**、**identifier query** 和 **import/API query**；不启用 path query；使用 enhanced BM25；不启用 context gate。

## 候选池成员

| 候选来源 | 是否启用 | 说明 | 召回数量 |
| --- | --- | --- | ---: |
| base query | 是 | 使用当前文件 `left_context` 的最后若干行作为基础查询 | top-60 |
| identifier query | 是 | 从 `left_context` 中抽取变量名、函数名、类名等 identifier 作为查询 | top-30 |
| import/API query | 是 | 从 `left_context` 中抽取 import 语句和 dotted API token 作为查询 | top-30 |
| path query | 否 | 不使用当前文件路径信息作为查询 | 不生效 |

## 参数拆解

| 参数片段 | 含义 |
| --- | --- |
| `id` | 启用 identifier query |
| `import` | 启用 import/API query |
| `nopath` | 不启用 path query |
| `base60` | base query BM25 召回 top-60 |
| `aux30` | 每个辅助 query 召回 top-30，即 identifier top-30、import/API top-30 |
| `pathk5` | path query 的 top-k 配置为 5，但由于 `nopath`，该值实际不生效 |
| `pool140` | 多路候选合并去重后，最多保留 140 个候选进入 RLRetriever |
| `enhbm25` | 使用 enhanced BM25，即索引 `file_path + description + code_content`，并进行 identifier/camel 分词 |
| `nogate` | 不启用 context gate |
| `enh_id_import` | 实验 tag，用于标记该实验配置，不表示额外功能 |

## 候选数量

该配置下，理论原始召回数量为：
```text
base 60 + identifier 30 + import/API 30 = 120
```
之后会根据候选块的：

```text
file_path + description + code_content
```

进行去重和合并。

合并后候选池上限为：

```text
pool140
```

由于原始召回最多约为 120 个，因此 `pool140` 通常不会造成截断，主要用于保留足够候选空间。

## 后续处理流程

候选池构建完成后，会进入 RLCoder 的 pretrained RLRetriever 进行重排：

```text
base / identifier / import_API 多路 BM25 召回
        ↓
候选去重与合并
        ↓
最多保留 140 个候选
        ↓
加入 stop block
        ↓
RLRetriever rerank
        ↓
选出最终跨文件上下文
        ↓
送入 generator 构造 prompt
```

## 总结

该配置可以概括为：

> 使用增强 BM25 构建由基础上下文、标识符信息、import/API 信息组成的多路径候选池；不使用路径查询；不使用 gate；候选池上限为 140，再交给 RLRetriever 统一重排。
