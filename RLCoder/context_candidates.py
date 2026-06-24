from collections import Counter

from datasets import CodeBlock


SOURCE_PRIORITY = {
    "base": 0,
    "draft": 1,
    "identifier": 2,
    "import_api": 3,
    "path": 4,
}


def recall_multi_path_candidates(args, examples, bm25_index, query_bundles):
    per_task_hits = [[] for _ in examples]
    source_counts = [Counter() for _ in examples]

    for source in _ordered_sources(query_bundles):
        batch_indices = []
        task_ids = []
        queries = []

        for idx, (example, bundle) in enumerate(zip(examples, query_bundles)):
            query = _query_for_source(bundle, source)
            if not query:
                continue
            batch_indices.append(idx)
            task_ids.append(example.task_id)
            queries.append(query)

        if not queries:
            continue

        source_results = bm25_index.query(
            task_ids,
            queries,
            topk=getattr(args, "ucm_topk_per_path", 20),
        )
        for idx, candidates in zip(batch_indices, source_results):
            source_counts[idx][source] += len(candidates)
            for rank, candidate in enumerate(candidates):
                per_task_hits[idx].append((source, rank, candidate))

    merged_candidates = []
    trace_rows = []
    for example, hits, counts in zip(examples, per_task_hits, source_counts):
        candidates, trace = merge_candidates(
            hits,
            candidate_pool_size=getattr(args, "ucm_candidate_pool_size", 100),
            task_id=example.task_id,
            source_counts=counts,
        )
        merged_candidates.append(candidates)
        trace_rows.append(trace)

    return merged_candidates, trace_rows


def merge_candidates(hits, candidate_pool_size, task_id=None, source_counts=None):
    grouped = {}

    for source, rank, candidate in hits:
        key = candidate_key(candidate)
        if key not in grouped:
            grouped[key] = {
                "candidate": candidate,
                "sources": set(),
                "best_rank_by_source": {},
            }
        grouped[key]["sources"].add(source)
        best_rank_by_source = grouped[key]["best_rank_by_source"]
        best_rank_by_source[source] = min(rank, best_rank_by_source.get(source, rank))

    ranked = sorted(grouped.values(), key=_candidate_sort_key)
    capped = ranked[:candidate_pool_size]
    candidates = [_copy_with_multi_type(item["candidate"], item["sources"]) for item in capped]

    trace = {
        "task_id": task_id,
        "source_hits": dict(source_counts or Counter()),
        "raw_hits": len(hits),
        "merged_candidates": len(ranked),
        "candidate_pool_size": len(candidates),
    }
    return candidates, trace


def candidate_key(candidate):
    return (candidate.file_path, candidate.description, candidate.code_content)


def log_retrieval_trace(dataset_name, trace_rows, retriever_inputs=None):
    prefix = f"[UCM trace][{dataset_name}]" if dataset_name else "[UCM trace]"
    final_counts = [len(x) for x in retriever_inputs] if retriever_inputs is not None else None
    for idx, trace in enumerate(trace_rows):
        source_hits = ", ".join(
            f"{source}:{count}" for source, count in sorted(trace["source_hits"].items())
        )
        retriever_input = (
            final_counts[idx]
            if final_counts is not None
            else trace["candidate_pool_size"]
        )
        print(
            f"{prefix} task={trace['task_id']} "
            f"sources=({source_hits}) raw={trace['raw_hits']} "
            f"merged={trace['merged_candidates']} retriever_input={retriever_input}"
        )


def _ordered_sources(query_bundles):
    sources = []
    seen = set()
    for bundle in query_bundles:
        for view in bundle:
            if view.source not in seen:
                seen.add(view.source)
                sources.append(view.source)
    return sorted(sources, key=lambda source: SOURCE_PRIORITY.get(source, 99))


def _query_for_source(bundle, source):
    for view in bundle:
        if view.source == source:
            return view.query
    return ""


def _candidate_sort_key(item):
    sources = item["sources"]
    best_source_priority = min(SOURCE_PRIORITY.get(source, 99) for source in sources)
    best_rank = min(item["best_rank_by_source"].values())
    base_or_draft_hit = 0 if sources.intersection({"base", "draft"}) else 1
    return (-len(sources), base_or_draft_hit, best_source_priority, best_rank)


def _copy_with_multi_type(candidate, sources):
    best_source = min(sources, key=lambda source: SOURCE_PRIORITY.get(source, 99))
    return CodeBlock(
        candidate.file_path,
        candidate.description,
        candidate.code_content,
        candidate.language,
        f"multi_{best_source}",
    )