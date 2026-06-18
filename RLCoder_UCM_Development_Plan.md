# RLCoder + Unified Context Modeling Development Plan

Date: 2026-06-18

This document is the implementation plan for building a repository-level code completion method based on pretrained RLCoder and three stitched modules:

1. multi-path candidate pool augmentation;
2. lightweight context graph expansion;
3. selective context gate and prompt budget allocation.

The target research framing is:

> Based on pretrained RLRetriever, model local context, lexical/semantic retrieval context, generated-feedback context, symbol/import context, and lightweight structural context as a unified candidate pool, then select and pack the useful context for repository-level code completion.

The upstream code has been cloned to:

```text
/Users/bytedance/Documents/仓库级代码补全研究/RLCoder
```

Primary factual references:

- RLCoder paper: https://arxiv.org/abs/2407.19487
- RLCoder code: https://github.com/DeepSoftwareAnalytics/RLCoder
- GraphCoder paper: https://arxiv.org/abs/2406.07003
- DraCo paper: https://arxiv.org/abs/2405.19782
- Repoformer paper: https://arxiv.org/abs/2403.10059
- CodeRAG paper: https://arxiv.org/abs/2509.16112

## 1. Upstream RLCoder Pipeline

### 1.1 Entry

All training and inference enter through:

```text
RLCoder/main.py
```

Important CLI groups:

- generator: `--generator_model_path`, `--generator_max_crossfile_length`, `--generator_max_context_length`, `--generator_max_generation_length`;
- retriever: `--retriever_model_path`, `--retriever_query_context_length`, `--retriever_candidate_context_length`;
- method: `--inference_type`, `--enable_generation`, `--enable_repocoder`, `--rlcoder_model_path`, `--disable_stop_block`;
- output: `--output_dir`;
- training: `--epoch`, `--inner_epoch`, `--sample_number`, `--data_per_epoch`, `--weighted_keywords`.

Important practical note:

- The main RLRetriever used for final `unixcoder_with_rl` reranking is loaded from `--retriever_model_path`.
- When `--enable_repocoder` is enabled, a second retriever is loaded from `--rlcoder_model_path`; upstream uses it for the first RepoCoder-style retrieval/generation pass that augments the query.
- Therefore, to use the pretrained RLCoder retriever as the final scorer, set `--retriever_model_path nov3630/RLRetriever`. If using the same pretrained retriever for the preliminary RepoCoder pass, also set `--rlcoder_model_path nov3630/RLRetriever`.

### 1.2 Dataset Loading

Defined in:

```text
RLCoder/datasets.py
```

Data objects:

- `CodeBlock`: cross-file candidate with `file_path`, `description`, `code_content`, `language`, `_type`;
- `Example`: completion sample with `task_id`, `file_path`, `left_context`, `right_context`, `related_files`, `target_code`, `language`.

Evaluation data paths expected by upstream:

```text
RLCoder/data/cceval/python/test.parquet
RLCoder/data/cceval/java/test.parquet
RLCoder/data/repoeval/line_level/test_0.parquet
RLCoder/data/repoeval/line_level/test_1.parquet
RLCoder/data/repoeval/api_level/test_0.parquet
RLCoder/data/repoeval/api_level/test_1.parquet
```

Training data paths expected by upstream:

```text
RLCoder/data/github_repos/python/train.parquet
RLCoder/data/github_repos/java/train.parquet
```

These are downloaded from the RLCoder HuggingFace dataset `nov3630/Data4RLCoder` according to the upstream README.

### 1.3 Candidate Construction

Defined in:

```text
RLCoder/bm25.py
```

The upstream candidate construction flow is:

1. `TaskSpecificBM25(examples, args)` builds one BM25 index per dataset/task.
2. For each `Example`, it reads `example.related_files`.
3. Each related file is split by `split_into_smaller_blocks()`.
4. Default mode is RLCoder natural candidates: split by blank lines and aggregate into at most 15 non-empty lines.
5. `--enable_fixed_block` switches to fixed 12-line blocks for ablation.
6. `TaskSpecificBM25.query(task_ids, queries, topk)` returns top-k candidate `CodeBlock`s for each task.

This is where module 1 can reuse the existing BM25 index without touching the lower-level index code.

### 1.4 Retrieval Dispatch

Defined in:

```text
RLCoder/main.py
```

The central function is:

```python
retrieve_codeblocks(args, examples, bm25, retriever, dataset_name, is_training=False, inference_type=None)
```

Current upstream behavior:

1. `baseline`: returns no cross-file context.
2. `bm25`: builds a query from the last 20 non-empty lines of `left_context`, then returns BM25 candidates.
3. `unixcoder`: BM25 first recalls candidates, then `Retriever.retrieve()` reranks them.
4. `unixcoder_with_rl`: BM25 recalls more candidates, stop block is added, then the RLRetriever reranks candidates.
5. `--enable_repocoder`: before final RLRetriever reranking, upstream first runs a `unixcoder` retrieval pass, generates a draft completion, and appends that draft to the query. This implements a RepoCoder-style iterative retrieval/generation step.

This function is the main hook for all three modules.

### 1.5 Dense/RL Retriever

Defined in:

```text
RLCoder/retriever.py
```

Behavior:

1. Loads tokenizer/model from `args.retriever_model_path`.
2. Encodes query and candidate text with UniXcoder-style formatting.
3. Computes normalized mean-pooled embeddings.
4. Scores candidates by query-candidate dot product.
5. Returns top-k `CodeBlock`s.

No generator weights are trained during inference. For our plan, the pretrained RLRetriever remains the unified scorer.

### 1.6 Generator and Prompt Construction

Defined in:

```text
RLCoder/generator.py
```

Critical function:

```python
CustomDataset.construct_prompts(example, retrieved_codeblocks)
```

Current behavior:

1. Iterates through retrieved blocks until it sees an empty-path stop block.
2. Joins retained cross-file blocks into `crossfile_context`.
3. Truncates cross-file context to `generator_max_crossfile_length`.
4. Adds current file path.
5. Uses the remaining context window for the tail of `left_context`.
6. Feeds the prompt into the generator for loss evaluation or generation.

This function is the hook for module 3 if prompt budget allocation needs token-level packing.

### 1.7 Evaluation Outputs

Inference mode command shape:

```bash
python main.py \
  --eval \
  --weighted_keywords \
  --enable_generation \
  --inference_type unixcoder_with_rl \
  --retriever_model_path nov3630/RLRetriever \
  --generator_model_path deepseek-ai/deepseek-coder-6.7b-base \
  --generator_max_crossfile_length 1536 \
  --generator_max_context_length 2048 \
  --generator_batch_size_per_gpu 16 \
  --output_dir result_infer/RLCoder_deepseekcoder_7b_crossfile_1536_infile_512
```

For each dataset, upstream writes:

```text
{output_dir}/{dataset}/prediction.jsonl
{output_dir}/{dataset}/prediction_truncated.jsonl
{output_dir}/{dataset}/exact_match_idx.jsonl
{output_dir}/{dataset}/detailed_results.json
{output_dir}/{dataset}/results.json
```

Metrics include loss/PPL plus EM, ES, ID_EM, ID_F1 where applicable.

Training mode writes:

```text
{output_dir}/result_init/{dataset}/prediction.jsonl
{output_dir}/result_{epoch}/{dataset}/prediction.jsonl
{output_dir}/retriever_cpkt/result_{epoch}/
```

For this project, the first milestone should use pretrained `nov3630/RLRetriever` and avoid retriever training until the stitched inference pipeline is stable.

## 2. RLCoder Paper Innovations and Where We Attach

RLCoder's key innovations:

1. **RLRetriever trained without labeled candidate data**
   - It treats the generator/evaluator's weighted target-code perplexity as feedback.
   - It selects the candidate with best weighted PPL as the positive action and updates the retriever.
   - Our use: keep pretrained `nov3630/RLRetriever` as the unified scorer, avoiding expensive retriever training in the first version.

2. **Weighted PPL reward**
   - It gives higher weight to early target tokens and identifier/API-like tokens.
   - Motivation: repository-level failures often involve wrong identifiers or APIs.
   - Our use: for later optional training, reuse weighted PPL to evaluate whether graph-expanded or multi-path candidates are useful.

3. **Split-Aggregate natural candidates**
   - Instead of fixed windows, RLCoder splits code by blank lines and aggregates semantically continuous mini-blocks.
   - Our use: keep natural candidates as the base candidate units; our graph and multi-path modules should add and reorder candidates, not replace this unit.

4. **Stop signal**
   - It inserts an empty candidate meaning "no cross-file context needed".
   - Prompt construction keeps only blocks before this empty stop block.
   - Our use: extend this from a binary stop signal into a selective context gate and token budget allocator.

5. **Compatibility with RepoCoder**
   - Upstream supports `--enable_repocoder`, where a first generation pass is used to augment the query.
   - Our use: treat RepoCoder-style generation feedback as one query source inside the unified context model.

Our planned innovations and attachment points:

1. **Unified candidate pool augmentation**
   - Position: before `Retriever.retrieve()` in `main.retrieve_codeblocks()`.
   - Adds lexical, identifier, import/API, path, and generated-draft query sources.
   - RLCoder relationship: expands what RLRetriever can choose from.

2. **Lightweight context graph expansion**
   - Position: after BM25/multi-path recall and before `Retriever.retrieve()`.
   - Adds neighbor, same-file, import-linked, and identifier-overlap related blocks.
   - RLCoder relationship: preserves RLCoder's natural candidates but adds structural relations among them.

3. **Selective gate and budget allocation**
   - Position: after `Retriever.retrieve()` and inside/near `Generator.CustomDataset.construct_prompts()`.
   - Decides how many retrieved blocks to retain and how many tokens each source type receives.
   - RLCoder relationship: generalizes the stop signal from a single empty candidate into source-aware context selection.

## 3. Target Architecture

The final inference flow should be:

```text
Example.left_context
        |
        v
Base query from last non-empty lines
        |
        v
Module 1: multi-path candidate pool augmentation
        |   - base query
        |   - identifier/API query
        |   - import/path query
        |   - optional RepoCoder draft query
        v
BM25 multi-query recall over RLCoder natural candidates
        |
        v
Module 2: lightweight context graph expansion
        |   - same-file neighbor
        |   - import/path relation
        |   - identifier-overlap relation
        v
Candidate dedupe and cap
        |
        v
Pretrained RLRetriever reranking
        |
        v
Module 3: selective context gate and prompt budget allocation
        |
        v
Generator prompt construction
        |
        v
Prediction + metrics
```

The generator remains unchanged in the first implementation. The pretrained RLRetriever remains the central scorer.

## 4. Module 1: Multi-Path Candidate Pool Augmentation

### 4.1 Goal

Replace single-query BM25 recall with multiple query views, then merge candidates into one candidate pool for RLRetriever.

This is inspired by CodeRAG's findings that query construction and multi-path retrieval matter, but we keep the implementation lightweight and compatible with RLCoder.

### 4.2 New File

```text
RLCoder/context_query.py
```

Suggested public functions:

```python
def build_base_query(example, context_len=20) -> str:
    ...

def extract_identifiers(text: str, language: str, max_items: int = 64) -> list[str]:
    ...

def extract_import_lines(text: str, language: str, max_lines: int = 32) -> list[str]:
    ...

def build_query_bundle(example, base_query: str, draft: str | None, args) -> dict[str, str]:
    ...
```

Query bundle keys:

- `base`: current upstream query;
- `identifier`: identifiers extracted from left context;
- `import_api`: import lines plus API-like dotted names;
- `path`: current file path split into tokens;
- `draft`: optional RepoCoder first-pass generation appended to base query.

Do not use `right_context` or `target_code`; that would leak evaluation answers.

### 4.3 New File

```text
RLCoder/context_candidates.py
```

Suggested public functions:

```python
def bm25_recall_multi_path(task_bm25, examples, query_bundles, topk_per_path, pool_topk):
    ...

def merge_and_dedupe_codeblocks(candidate_lists, max_candidates):
    ...

def codeblock_key(block):
    ...
```

Implementation detail:

- Call existing `TaskSpecificBM25.query()` once per query path.
- Merge by stable priority:
  1. candidates found by more query paths;
  2. candidates found by `base` or `draft`;
  3. earlier BM25 rank;
  4. source priority.
- Store source type in `CodeBlock._type`, e.g. `multi_base`, `multi_identifier`, `multi_import_api`, `multi_path`, `multi_draft`, without changing the `CodeBlock` constructor.

### 4.4 Integration Point

Modify `main.retrieve_codeblocks()`:

- Keep the original path when `--enable_ucm` is false.
- When `--enable_ucm` and `--enable_multi_path_retrieval` are true:
  1. build `queries` exactly as upstream does;
  2. optionally run the upstream `--enable_repocoder` draft generation before final candidate recall;
  3. build query bundles;
  4. call multi-path BM25 recall;
  5. pass merged candidates to RLRetriever.

### 4.5 CLI Flags

Add to `main.py`:

```text
--enable_ucm
--enable_multi_path_retrieval
--ucm_topk_per_path default 20
--ucm_candidate_pool_size default 100
--ucm_query_identifier_limit default 64
--ucm_query_import_limit default 32
```

### 4.6 Tests

Unit tests:

- identifier extraction does not include Python/Java keywords;
- import extraction handles Python `import/from` and Java `import ...;`;
- multi-path merge dedupes identical blocks;
- no `right_context` or `target_code` appears in query bundle.

Smoke test:

```bash
python -m py_compile main.py bm25.py datasets.py retriever.py generator.py context_query.py context_candidates.py
```

## 5. Module 2: Lightweight Context Graph Expansion

### 5.1 Goal

After initial multi-path recall, expand around high-value candidates using simple repository structure. This is the most important module for the "context unified modeling" story.

This borrows the direction of GraphCoder and DraCo, but the first implementation should avoid heavy full dataflow analysis. The first version should be deterministic, language-tolerant, and testable without GPU.

### 5.2 New File

```text
RLCoder/context_graph.py
```

Suggested public API:

```python
class ContextGraphIndex:
    @classmethod
    def from_task_bm25(cls, task_bm25):
        ...

    def expand(self, task_id, seed_blocks, max_neighbors_per_seed, max_total):
        ...
```

Graph node:

- one `CodeBlock` from `TaskSpecificBM25.code_blocks[task_id]`.

Graph edges:

1. `same_file_prev_next`: adjacent natural candidates from the same file;
2. `same_file_nearby`: candidates in the same file with close line ranges parsed from `description`;
3. `import_path`: left-context imports or candidate imports match another file path/module name;
4. `identifier_overlap`: two candidates share rare identifiers.

Start with edges 1 and 4. Add import/path edges after tests pass.

### 5.3 Candidate Metadata

Do not change model input semantics. Expanded graph candidates should still be plain `CodeBlock`s.

Use `_type` values:

- `graph_neighbor`;
- `graph_same_file`;
- `graph_import`;
- `graph_identifier`.

The string representation in `CodeBlock.__str__()` already includes `description` and `code_content`, so RLRetriever can rerank expanded blocks normally.

### 5.4 Integration Point

Modify `main.retrieve_codeblocks()`:

```text
candidate_codeblocks = multi_path_or_original_recall(...)
if args.enable_context_graph:
    graph_index = get_or_build_graph_index(bm25[dataset_name])
    expanded = graph_index.expand(...)
    candidate_codeblocks = merge_and_dedupe_codeblocks([candidate_codeblocks, expanded], ...)
candidate_codeblocks = retriever.retrieve(...)
```

Cache graph indices by `dataset_name`, just like upstream caches BM25.

Suggested cache:

```python
context_graph_indices = {}
```

In the first implementation, keep graph expansion before RLRetriever. This makes pretrained RLRetriever the unified scorer of both original and graph-expanded context.

### 5.5 CLI Flags

```text
--enable_context_graph
--ucm_graph_max_seed default 20
--ucm_graph_max_neighbors_per_seed default 2
--ucm_graph_max_expanded default 60
--ucm_graph_enable_identifier_edges
--ucm_graph_enable_import_edges
```

### 5.6 Tests

Unit tests:

- graph index groups blocks by `task_id`;
- same-file neighbor expansion returns previous/next blocks;
- dedupe prevents graph expansion from duplicating seed blocks;
- max-neighbor and max-total caps are respected;
- no block from the current target file is introduced unless upstream already allows it. Upstream candidates are from `related_files`, so preserve that assumption.

## 6. Module 3: Selective Context Gate and Prompt Budget Allocation

### 6.1 Goal

Generalize RLCoder's stop signal into source-aware context selection and prompt packing.

The first implementation should not train a Repoformer-like policy. Use deterministic heuristics and expose enough trace logs for later ablation.

### 6.2 New File

```text
RLCoder/context_gate.py
```

Suggested public functions:

```python
def apply_context_gate(example, query, retrieved_blocks, args, tokenizer=None):
    ...

def allocate_prompt_budget(blocks, max_crossfile_tokens, source_budgets):
    ...

def should_keep_block(block, rank, source_counts, args):
    ...
```

Heuristic v1:

- Always respect upstream empty stop block: if `file_path == ""`, discard blocks after it.
- Keep at least `ucm_gate_min_blocks` when no stop block appears before that rank.
- Cap total blocks by `ucm_gate_max_blocks`.
- Avoid overusing one source type: if graph-expanded blocks exceed `ucm_gate_graph_ratio`, downsample lower-ranked graph blocks.
- Prefer blocks whose `file_path`/identifiers overlap with the query.
- If top-ranked block is stop block, use in-file context only.

### 6.3 Prompt Budget Allocation

Modify `generator.CustomDataset.construct_prompts()` only after retrieval-side gate works.

Current upstream allocates one flat `generator_max_crossfile_length` to all retrieved blocks. New behavior:

1. group blocks by `_type` prefix;
2. reserve minimal budget for high-rank RLRetriever candidates;
3. cap graph-expanded context so it does not crowd out original high-confidence candidates;
4. keep prompt order by final rank unless a later ablation shows grouped order is better.

Default source budgets:

```text
original/multi-path candidates: 60%
graph-expanded candidates: 25%
draft-related/path/import context: 15%
```

If implementation complexity becomes high, fallback to retrieval-side truncation only and leave token-level budget allocation for the second development pass.

### 6.4 Integration Point

Option A, simple first pass:

- Apply gate after `retriever.retrieve()` inside `main.retrieve_codeblocks()`.
- Return gated `retrieved_codeblocks`.
- Leave `generator.py` unchanged.

Option B, full pass:

- Apply retrieval-side gate in `main.py`;
- Apply token budget in `generator.CustomDataset.construct_prompts()`.

Recommended order: implement Option A first, run ablations, then implement Option B.

### 6.5 CLI Flags

```text
--enable_context_gate
--ucm_gate_min_blocks default 1
--ucm_gate_max_blocks default 5
--ucm_gate_graph_ratio default 0.4
--ucm_gate_trace
--ucm_prompt_budget_mode choices: flat, source_aware
```

### 6.6 Tests

Unit tests:

- stop block at rank 0 returns no cross-file blocks;
- blocks after stop block are removed;
- graph ratio cap is respected;
- source-aware budget never exceeds `generator_max_crossfile_length`;
- prompt construction still includes current file path and left context.

## 7. Implementation Order

### Phase 0: Baseline Reproduction

Goal: make sure upstream can run with downloaded data and pretrained retriever.

Commands:

```bash
cd /Users/bytedance/Documents/仓库级代码补全研究/RLCoder
pip install -r requirements.txt
```

Recommended first evaluation with small generator to reduce cost:

```bash
python main.py \
  --eval \
  --debug \
  --weighted_keywords \
  --enable_generation \
  --inference_type unixcoder_with_rl \
  --retriever_model_path nov3630/RLRetriever \
  --generator_model_path deepseek-ai/deepseek-coder-1.3b-base \
  --generator_max_crossfile_length 1536 \
  --generator_max_context_length 2048 \
  --generator_batch_size_per_gpu 8 \
  --output_dir result_infer/debug_RLCoder_deepseekcoder_1b
```

Expected outputs:

```text
result_infer/debug_RLCoder_deepseekcoder_1b/{dataset}/prediction.jsonl
result_infer/debug_RLCoder_deepseekcoder_1b/{dataset}/results.json
```

### Phase 1: Multi-Path Candidate Pool

Files to add:

```text
context_query.py
context_candidates.py
tests/test_context_query.py
tests/test_context_candidates.py
```

Files to modify:

```text
main.py
```

Acceptance criteria:

- Existing commands produce identical behavior when `--enable_ucm` is not set.
- `--enable_ucm --enable_multi_path_retrieval` runs on `--debug`.
- Retrieval trace shows candidate source counts.
- No evaluation sample uses `right_context` or `target_code`.

Suggested command:

```bash
python main.py \
  --eval \
  --debug \
  --weighted_keywords \
  --enable_generation \
  --enable_ucm \
  --enable_multi_path_retrieval \
  --inference_type unixcoder_with_rl \
  --retriever_model_path nov3630/RLRetriever \
  --generator_model_path deepseek-ai/deepseek-coder-1.3b-base \
  --generator_max_crossfile_length 1536 \
  --generator_max_context_length 2048 \
  --generator_batch_size_per_gpu 8 \
  --output_dir result_infer/debug_UCM_multipath
```

### Phase 2: Lightweight Context Graph

Files to add:

```text
context_graph.py
tests/test_context_graph.py
```

Files to modify:

```text
main.py
context_candidates.py
```

Acceptance criteria:

- Graph expansion can be disabled independently.
- Graph expansion never changes behavior when `--enable_context_graph` is false.
- Expanded candidates are deduped and source-tagged.
- RLRetriever receives one merged candidate pool containing both original and graph-expanded blocks.

Suggested command:

```bash
python main.py \
  --eval \
  --debug \
  --weighted_keywords \
  --enable_generation \
  --enable_ucm \
  --enable_multi_path_retrieval \
  --enable_context_graph \
  --inference_type unixcoder_with_rl \
  --retriever_model_path nov3630/RLRetriever \
  --generator_model_path deepseek-ai/deepseek-coder-1.3b-base \
  --generator_max_crossfile_length 1536 \
  --generator_max_context_length 2048 \
  --generator_batch_size_per_gpu 8 \
  --output_dir result_infer/debug_UCM_multipath_graph
```

### Phase 3: Selective Context Gate

Files to add:

```text
context_gate.py
tests/test_context_gate.py
```

Files to modify:

```text
main.py
generator.py
```

Acceptance criteria:

- Stop block semantics remain compatible with upstream.
- Gate can be disabled independently.
- Prompt length never exceeds `generator_max_context_length`.
- Trace file explains final retained context count/source distribution.

Suggested command:

```bash
python main.py \
  --eval \
  --debug \
  --weighted_keywords \
  --enable_generation \
  --enable_ucm \
  --enable_multi_path_retrieval \
  --enable_context_graph \
  --enable_context_gate \
  --ucm_gate_trace \
  --inference_type unixcoder_with_rl \
  --retriever_model_path nov3630/RLRetriever \
  --generator_model_path deepseek-ai/deepseek-coder-1.3b-base \
  --generator_max_crossfile_length 1536 \
  --generator_max_context_length 2048 \
  --generator_batch_size_per_gpu 8 \
  --output_dir result_infer/debug_UCM_full
```

### Phase 4: Full Evaluation

Use DeepSeekCoder-6.7B on A800 80G:

```bash
python main.py \
  --eval \
  --weighted_keywords \
  --enable_generation \
  --enable_ucm \
  --enable_multi_path_retrieval \
  --enable_context_graph \
  --enable_context_gate \
  --inference_type unixcoder_with_rl \
  --retriever_model_path nov3630/RLRetriever \
  --generator_model_path deepseek-ai/deepseek-coder-6.7b-base \
  --generator_max_crossfile_length 1536 \
  --generator_max_context_length 2048 \
  --generator_batch_size_per_gpu 16 \
  --output_dir result_infer/UCM_RLCoder_deepseekcoder_7b_crossfile_1536_infile_512
```

If memory is tight, reduce:

```text
--generator_batch_size_per_gpu 4 or 8
--retriever_batch_size_per_gpu 32
```

## 8. Testing Plan

### 8.1 No-GPU Tests

Run before any GPU evaluation:

```bash
python -m py_compile main.py bm25.py datasets.py retriever.py generator.py context_query.py context_candidates.py context_graph.py context_gate.py
python -m pytest tests/test_context_query.py tests/test_context_candidates.py tests/test_context_graph.py tests/test_context_gate.py
```

These tests should use fake `CodeBlock` objects and should not import or load HuggingFace models.

### 8.2 GPU Smoke Tests

After data and models are available:

1. upstream baseline on `--debug`;
2. UCM multi-path only on `--debug`;
3. UCM multi-path + graph on `--debug`;
4. UCM full on `--debug`;
5. full evaluation.

### 8.3 Regression Checks

Every phase must preserve:

- upstream behavior when all UCM flags are off;
- no use of `right_context` or `target_code` in retrieval/prompt construction;
- no mutation of original `Example` except upstream progressive generation behavior in `main.py`;
- output schema under `{output_dir}/{dataset}`.

## 9. Ablation Matrix

Run these in order:

```text
A0: RLCoder pretrained baseline
A1: A0 + multi-path candidate pool
A2: A1 + lightweight context graph
A3: A2 + selective context gate
A4: A3 + RepoCoder-style draft query, if not already enabled
```

Optional ablations:

```text
B1: graph without identifier edges
B2: graph without same-file neighbor edges
B3: gate disabled but graph enabled
B4: source-aware prompt budget vs flat upstream prompt budget
B5: pretrained RLRetriever vs microsoft/unixcoder-base scorer
```

Main metrics:

- EM;
- ES;
- ID_EM;
- ID_F1;
- PPL/loss;
- retrieval latency;
- average retained cross-file tokens;
- stop/no-context rate.

## 10. Proposed New Trace Artifacts

Add only when `--ucm_gate_trace` or `--ucm_trace_retrieval` is enabled:

```text
{output_dir}/{dataset}/ucm_retrieval_trace.jsonl
{output_dir}/{dataset}/ucm_context_stats.json
```

Each JSONL record:

```json
{
  "task_id": "...",
  "query_sources": ["base", "identifier", "import_api", "draft"],
  "candidate_counts": {"base": 20, "identifier": 20, "graph_neighbor": 15},
  "merged_candidate_count": 85,
  "retrieved_count": 10,
  "retained_count": 4,
  "stop_rank": null,
  "retained_sources": {"multi_base": 2, "graph_neighbor": 1, "multi_draft": 1}
}
```

This is useful for paper analysis and debugging.

## 11. Risk and Fallback

Risk 1: multi-path recall adds noise.

- Fallback: reduce `ucm_topk_per_path`, prioritize base/draft candidates, or use RLRetriever score-only reranking.

Risk 2: graph expansion adds too many weak candidates.

- Fallback: expand only top-N seed candidates and only previous/next same-file neighbors.

Risk 3: context gate removes useful blocks.

- Fallback: keep upstream stop signal only and use `ucm_gate_min_blocks=3`.

Risk 4: prompt budget changes reduce performance.

- Fallback: leave `generator.py` unchanged and do only retrieval-side gating.

Risk 5: one A800 is slow for full 7B evaluation.

- Fallback: use DeepSeekCoder-1.3B for development, then run only final ablations on DeepSeekCoder-6.7B with smaller batch size.

## 12. Paper Framing

The method should not be described as "just adding RAG modules." The stronger framing is:

> RLCoder learns which candidate is useful through generator feedback, but its upstream implementation still begins from a relatively narrow candidate construction and flat prompt packing process. We extend it with unified context modeling: heterogeneous context sources are normalized into a shared candidate representation, structurally expanded through a lightweight repository context graph, scored by a pretrained feedback-trained RLRetriever, and selectively packed into the generator prompt.

Contribution wording:

1. A unified context candidate pool that combines local code, lexical retrieval, symbol/import information, path information, and generation feedback.
2. A lightweight repository context graph that expands retrieved candidates with structural neighbors while retaining RLCoder natural candidate units.
3. A selective source-aware context gate that generalizes RLCoder's stop signal and controls prompt budget.
4. A practical integration with pretrained RLCoder that avoids full retriever retraining and is suitable for one A800 80G.

## 13. Development Checklist

- [ ] Download `nov3630/Data4RLCoder` to `RLCoder/data`.
- [ ] Install dependencies.
- [ ] Run upstream debug baseline.
- [ ] Add `context_query.py`.
- [ ] Add `context_candidates.py`.
- [ ] Add multi-path flags and integration in `main.py`.
- [ ] Add unit tests for query and candidate merging.
- [ ] Run py_compile and pytest.
- [ ] Run debug multi-path evaluation.
- [ ] Add `context_graph.py`.
- [ ] Add graph flags and integration in `main.py`.
- [ ] Add graph unit tests.
- [ ] Run debug multi-path + graph evaluation.
- [ ] Add `context_gate.py`.
- [ ] Add gate flags and integration.
- [ ] Add gate unit tests.
- [ ] Run debug full UCM evaluation.
- [ ] Run full ablation matrix.
- [ ] Summarize EM/ES/ID_F1 and trace statistics.

## 14. Dataset Split and Pretrained-Retriever Usage Notes

### 14.1 What Is Split by the Paper

The paper describes a data construction process that starts from raw GitHub repositories:

1. randomly select 10,000 large Python and Java repositories created before March 2023;
2. exclude repositories included in CrossCodeEval and RepoEval to reduce evaluation leakage;
3. analyze imports to build dependency clusters of files;
4. discard single-file clusters;
5. topologically sort files in each cluster;
6. choose files other than the first file as completion targets;
7. randomly mask a target span inside the selected file;
8. use other files in the cluster as cross-file candidate context;
9. split candidate files with the Split-Aggregate natural-candidate strategy.

The cloned repository does not include the full raw GitHub collection and dependency-analysis preprocessing scripts. Instead, the released code expects the preprocessed HuggingFace dataset `nov3630/Data4RLCoder` under `RLCoder/data`.

### 14.2 What Is Split by the Code

The code hardcodes two kinds of data loading:

Evaluation benchmark data:

```text
data/cceval/python/test.parquet
data/cceval/java/test.parquet
data/repoeval/line_level/test_0.parquet
data/repoeval/line_level/test_1.parquet
data/repoeval/api_level/test_0.parquet
data/repoeval/api_level/test_1.parquet
```

These are loaded by `datasets.load_test_dataset()`. For CrossCodeEval, it reads a single `test.parquet`; for RepoEval line/API, it concatenates `test_0.parquet` and `test_1.parquet`. The benchmark split is therefore already encoded in the released parquet files, not created at runtime.

Training/validation data:

```text
data/github_repos/python/train.parquet
data/github_repos/java/train.parquet
```

These are loaded by `datasets.load_train_and_valid_dataset()`. The code groups rows into repository/file clusters according to the `first` column. After grouping, it takes:

```text
all_data[:2000]      -> training clusters
all_data[2000:2200]  -> validation/GitHubEval clusters
```

This happens independently for Python and Java, then both language lists are shuffled.

Runtime sample construction:

`datasets.construct_dataset()` converts a cluster into completion examples by:

1. choosing one target file from `example[1:]`;
2. using every other file in the cluster as `related_files`;
3. selecting a random target position between 20% and 80% of the selected file;
4. sampling a target span length between 32 and 64 whitespace tokens;
5. requiring `left_context` to exceed 80 tokens and target code to exceed 8 tokens.

Training mode calls this on `training_raw_data` every epoch:

```text
construct_dataset(training_raw_data, args.data_per_epoch)
```

Evaluation mode also builds an extra internal `github_eval` from the validation clusters:

```text
construct_dataset(eval_raw_data, 1000)
```

In addition, it evaluates on CrossCodeEval Python/Java and RepoEval line/API from their released test parquet files.

### 14.3 If We Do Not Retrain RLRetriever, What Does the Code Still Do?

When using pretrained `nov3630/RLRetriever` with `--eval`, the source code still performs the full inference and evaluation pipeline:

1. load preprocessed benchmark examples;
2. split each example's related files into natural candidates;
3. build a BM25 index for each task;
4. recall candidate code blocks with BM25;
5. add the stop candidate when enabled;
6. rerank candidates with pretrained RLRetriever;
7. optionally do RepoCoder-style draft generation with `--enable_repocoder`;
8. construct prompts from retained cross-file context plus in-file context;
9. run the generator model;
10. write predictions and compute EM/ES/ID metrics.

So the code remains meaningful as:

- a faithful reproduction and evaluation harness for the pretrained RLCoder method;
- a baseline runner for `NoRetrieval`, `BM25`, `UniXcoder`, `RLRetriever`, and `RepoCoder + RLCoder`;
- the best integration point for our UCM modules because candidate construction, retrieval, prompt construction, generation, and metrics are already wired together.

What it does not do in pretrained-only mode:

- it does not update retriever parameters;
- it does not reproduce the raw GitHub repository crawling and dependency-analysis preprocessing;
- it does not create a new RL-trained model checkpoint.

Research implication:

Using pretrained RLRetriever alone is a baseline, not a new contribution. Our contribution should be framed as improving the context modeling around a frozen feedback-trained retriever: expanding the candidate pool, adding lightweight structural context, and selectively packing context into the prompt. Optional later work can retrain/fine-tune RLRetriever on the augmented candidate distribution, but that is not required for the first feasible thesis path.
