import copy
import os
import re
from collections import Counter, defaultdict


IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+)|import\s+(.+)|package\s+([A-Za-z0-9_\.]+)|import\s+([A-Za-z0-9_\.]+)\s*;)")

KEYWORDS = {
    "False", "None", "True", "abstract", "and", "as", "assert", "async",
    "await", "boolean", "break", "byte", "case", "catch", "char", "class",
    "const", "continue", "def", "default", "del", "do", "double", "elif",
    "else", "enum", "except", "extends", "final", "finally", "float", "for",
    "from", "global", "goto", "if", "implements", "import", "in",
    "instanceof", "int", "interface", "is", "lambda", "long", "native",
    "new", "nonlocal", "not", "null", "or", "package", "pass", "private",
    "protected", "public", "raise", "return", "self", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "true", "void", "volatile", "while", "with", "yield",
}


class ContextGraphIndex:
    def __init__(self, code_blocks_by_task):
        self.code_blocks_by_task = code_blocks_by_task
        self.position_by_task = {}
        self.file_indices_by_task = {}
        self.identifier_index_by_task = {}
        self.path_index_by_task = {}
        self.identifier_df_by_task = {}
        self._build_positions()

    @classmethod
    def from_task_bm25(cls, task_bm25):
        return cls(getattr(task_bm25, "code_blocks", {}))

    def expand(self, example, seed_blocks, args):
        task_id = example.task_id
        max_seed = max(0, getattr(args, "ucm_graph_max_seed", 20))
        max_neighbors = max(0, getattr(args, "ucm_graph_max_neighbors_per_seed", 2))
        max_total = max(0, getattr(args, "ucm_graph_max_expanded", 40))
        enable_identifier = getattr(args, "ucm_graph_enable_identifier_edges", False)
        enable_import = getattr(args, "ucm_graph_enable_import_edges", False)
        query_identifiers = set(_extract_identifiers(getattr(example, "left_context", "")))
        query_import_tokens = _extract_import_tokens(getattr(example, "left_context", ""))

        seen = {block_key(block) for block in seed_blocks}
        proposals = {}
        proposed_counts = Counter()
        source_counts = Counter()

        for seed_rank, seed in enumerate(seed_blocks[:max_seed]):
            if _is_stop_block(seed):
                continue

            neighbors = []
            neighbors.extend(
                self._same_file_neighbors(
                    task_id,
                    seed,
                    max_neighbors,
                    query_identifiers,
                    args,
                )
            )
            if enable_identifier:
                neighbors.extend(
                    self._identifier_neighbors(
                        task_id,
                        seed,
                        max_neighbors,
                        query_identifiers,
                        args,
                    )
                )
            if enable_import:
                neighbors.extend(
                    self._import_neighbors(
                        task_id,
                        seed,
                        max_neighbors,
                        query_import_tokens,
                        args,
                    )
                )

            seed_weight = _seed_rank_weight(seed_rank, args)
            for neighbor, source, edge_score, edge_distance in neighbors:
                key = block_key(neighbor)
                if key in seen:
                    continue
                proposed_counts[source] += 1
                score = seed_weight * edge_score
                prev = proposals.get(key)
                if prev is None or score > prev["score"]:
                    proposals[key] = {
                        "block": neighbor,
                        "source": source,
                        "score": score,
                        "seed_rank": seed_rank,
                        "edge_distance": edge_distance,
                    }

        ranked = sorted(
            proposals.values(),
            key=lambda item: (
                -item["score"],
                item["seed_rank"],
                _graph_source_priority(item["source"]),
                block_key(item["block"]),
            ),
        )

        selected = []
        scores = []
        for item in ranked[:max_total]:
            selected.append(
                _copy_graph_block(
                    item["block"],
                    item["source"],
                    item["seed_rank"],
                    item["score"],
                    item["edge_distance"],
                )
            )
            source_counts[item["source"]] += 1
            scores.append(item["score"])

        trace = {
            "graph_expanded": len(selected),
            "graph_sources": dict(source_counts),
            "graph_proposed_sources": dict(proposed_counts),
            "graph_proposed_candidates": len(proposals),
            "graph_seed_count": min(max_seed, len(seed_blocks)),
            "graph_query_identifier_count": len(query_identifiers),
            "graph_query_import_token_count": len(query_import_tokens),
            "graph_score_min": round(min(scores), 6) if scores else None,
            "graph_score_max": round(max(scores), 6) if scores else None,
            "graph_score_avg": round(sum(scores) / len(scores), 6) if scores else None,
        }
        return selected, trace

    def _build_positions(self):
        for task_id, code_blocks in self.code_blocks_by_task.items():
            positions = {}
            file_indices = defaultdict(list)
            for idx, block in enumerate(code_blocks):
                positions[block_key(block)] = idx
                file_indices[getattr(block, "file_path", "")].append(idx)
            self.position_by_task[task_id] = positions
            self.file_indices_by_task[task_id] = dict(file_indices)

    def _same_file_neighbors(self, task_id, seed, max_neighbors, query_identifiers, args):
        if max_neighbors <= 0:
            return []

        code_blocks = self.code_blocks_by_task.get(task_id, [])
        seed_idx = self.position_by_task.get(task_id, {}).get(block_key(seed))
        if seed_idx is None:
            return []

        file_path = getattr(seed, "file_path", "")
        file_indices = self.file_indices_by_task.get(task_id, {}).get(file_path, [])
        try:
            file_pos = file_indices.index(seed_idx)
        except ValueError:
            return []

        neighbors = []
        radius = 1
        while len(neighbors) < max_neighbors and (file_pos - radius >= 0 or file_pos + radius < len(file_indices)):
            if file_pos - radius >= 0:
                block = code_blocks[file_indices[file_pos - radius]]
                neighbors.append((
                    block,
                    "graph_same_file",
                    _same_file_score(block, radius, query_identifiers, args),
                    radius,
                ))
                if len(neighbors) >= max_neighbors:
                    break
            if file_pos + radius < len(file_indices):
                block = code_blocks[file_indices[file_pos + radius]]
                neighbors.append((
                    block,
                    "graph_same_file",
                    _same_file_score(block, radius, query_identifiers, args),
                    radius,
                ))
            radius += 1

        return neighbors[:max_neighbors]

    def _identifier_neighbors(self, task_id, seed, max_neighbors, query_identifiers, args):
        if max_neighbors <= 0:
            return []

        self._ensure_identifier_index(task_id)
        identifier_index = self.identifier_index_by_task.get(task_id, {})
        identifier_df = self.identifier_df_by_task.get(task_id, {})
        max_df = getattr(args, "ucm_graph_identifier_max_df", 20)

        seed_identifiers = [
            token for token in _extract_identifiers(seed.code_content)
            if 1 < identifier_df.get(token, 0) <= max_df
        ]
        focused_identifiers = [
            token for token in seed_identifiers
            if token in query_identifiers
        ]
        if focused_identifiers:
            seed_identifiers = focused_identifiers
        if not seed_identifiers:
            return []

        seed_key = block_key(seed)
        scores = Counter()
        for token in seed_identifiers[:32]:
            for idx in identifier_index.get(token, []):
                block = self.code_blocks_by_task[task_id][idx]
                if block_key(block) != seed_key:
                    scores[idx] += 1
                    if token in query_identifiers:
                        scores[idx] += getattr(args, "ucm_graph_query_overlap_bonus", 2)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            (
                self.code_blocks_by_task[task_id][idx],
                "graph_identifier",
                _identifier_score(score, args),
                1,
            )
            for idx, score in ranked[:max_neighbors]
        ]

    def _import_neighbors(self, task_id, seed, max_neighbors, query_import_tokens, args):
        if max_neighbors <= 0:
            return []

        self._ensure_path_index(task_id)
        path_index = self.path_index_by_task.get(task_id, {})
        import_tokens = set(query_import_tokens)
        import_tokens.update(_extract_import_tokens(getattr(seed, "code_content", "")))
        if not import_tokens:
            return []

        seed_key = block_key(seed)
        scores = Counter()
        for token in import_tokens:
            for idx in path_index.get(token, []):
                block = self.code_blocks_by_task[task_id][idx]
                if block_key(block) != seed_key:
                    scores[idx] += 1

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            (
                self.code_blocks_by_task[task_id][idx],
                "graph_import",
                _import_score(score, args),
                1,
            )
            for idx, score in ranked[:max_neighbors]
        ]

    def _ensure_identifier_index(self, task_id):
        if task_id in self.identifier_index_by_task:
            return

        identifier_index = defaultdict(list)
        identifier_df = Counter()
        for idx, block in enumerate(self.code_blocks_by_task.get(task_id, [])):
            identifiers = set(_extract_identifiers(getattr(block, "code_content", "")))
            for token in identifiers:
                identifier_index[token].append(idx)
                identifier_df[token] += 1

        self.identifier_index_by_task[task_id] = dict(identifier_index)
        self.identifier_df_by_task[task_id] = identifier_df

    def _ensure_path_index(self, task_id):
        if task_id in self.path_index_by_task:
            return

        path_index = defaultdict(list)
        for idx, block in enumerate(self.code_blocks_by_task.get(task_id, [])):
            for token in _path_tokens(getattr(block, "file_path", "")):
                path_index[token].append(idx)

        self.path_index_by_task[task_id] = dict(path_index)


def expand_context_graph_candidates(args, examples, task_bm25, candidate_codeblocks):
    graph_index = get_or_build_context_graph_index(task_bm25)
    expanded_batches = []
    trace_rows = []

    for example, candidates in zip(examples, candidate_codeblocks):
        expanded, trace = graph_index.expand(example, candidates, args)
        merged = _append_deduped(candidates, expanded)
        expanded_batches.append(merged)
        trace["candidate_pool_size_after_graph"] = len(merged)
        trace_rows.append(trace)

    return expanded_batches, trace_rows


def add_graph_trace_results(trace_rows, graph_trace_rows):
    if not trace_rows or not graph_trace_rows:
        return trace_rows

    for trace, graph_trace in zip(trace_rows, graph_trace_rows):
        trace.update(graph_trace)
    return trace_rows


def get_or_build_context_graph_index(task_bm25):
    graph_index = getattr(task_bm25, "_ucm_context_graph_index", None)
    if graph_index is None:
        graph_index = ContextGraphIndex.from_task_bm25(task_bm25)
        setattr(task_bm25, "_ucm_context_graph_index", graph_index)
    return graph_index


def block_key(block):
    return (
        getattr(block, "file_path", ""),
        getattr(block, "description", ""),
        getattr(block, "code_content", ""),
    )


def _append_deduped(candidates, expanded):
    merged = list(candidates)
    seen = {block_key(block) for block in merged}
    for block in expanded:
        key = block_key(block)
        if key in seen:
            continue
        seen.add(key)
        merged.append(block)
    return merged


def _copy_graph_block(block, source, seed_rank, score=None, edge_distance=None):
    copied = copy.copy(block)
    copied._type = source
    copied._ucm_sources = (source,)
    copied._ucm_graph_seed_rank = seed_rank
    copied._ucm_graph_score = score
    copied._ucm_graph_edge_distance = edge_distance
    return copied


def _seed_rank_weight(seed_rank, args):
    decay = max(0.0, getattr(args, "ucm_graph_seed_rank_decay", 0.05))
    return 1.0 / (1.0 + decay * seed_rank)


def _same_file_score(block, distance, query_identifiers, args):
    base = getattr(args, "ucm_graph_same_file_weight", 1.0)
    distance_decay = getattr(args, "ucm_graph_distance_decay", 0.75)
    overlap = _query_identifier_overlap(block, query_identifiers)
    return base * (distance_decay ** max(0, distance - 1)) + overlap


def _identifier_score(overlap_count, args):
    base = getattr(args, "ucm_graph_identifier_weight", 1.2)
    return base + max(0, overlap_count)


def _import_score(overlap_count, args):
    base = getattr(args, "ucm_graph_import_weight", 1.4)
    return base + max(0, overlap_count)


def _query_identifier_overlap(block, query_identifiers):
    if not query_identifiers:
        return 0.0
    block_identifiers = set(_extract_identifiers(getattr(block, "code_content", "")))
    overlap = len(block_identifiers & query_identifiers)
    return min(2.0, 0.25 * overlap)


def _graph_source_priority(source):
    return {
        "graph_import": 0,
        "graph_identifier": 1,
        "graph_same_file": 2,
    }.get(source, 99)


def _extract_identifiers(text):
    tokens = []
    seen = set()
    for token in IDENTIFIER_RE.findall(text or ""):
        if len(token) <= 2 or token in KEYWORDS:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _extract_import_tokens(text):
    tokens = set()
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or ("import" not in stripped and not stripped.startswith("package ")):
            continue
        match = IMPORT_RE.match(stripped)
        if not match:
            continue
        for group in match.groups():
            if not group:
                continue
            for part in re.split(r"[,;\s]+", group):
                part = part.strip()
                if not part or part in {"as", "*"}:
                    continue
                tokens.update(_module_tokens(part))
    return tokens


def _path_tokens(file_path):
    normalized = (file_path or "").replace("\\", "/")
    stem, _ = os.path.splitext(normalized)
    parts = [part for part in re.split(r"[/._\-\s]+", stem) if part]
    tokens = set(parts)
    if parts:
        tokens.add(".".join(parts))
        tokens.add("/".join(parts))
        tokens.add(parts[-1])
    return {token.lower() for token in tokens if len(token) > 1}


def _module_tokens(module_name):
    cleaned = module_name.strip().strip("()")
    cleaned = cleaned.replace("/", ".")
    parts = [part for part in cleaned.split(".") if part]
    tokens = set(parts)
    if parts:
        for end in range(1, len(parts) + 1):
            tokens.add(".".join(parts[:end]))
        tokens.add(parts[-1])
    return {token.lower() for token in tokens if len(token) > 1}


def _is_stop_block(block):
    return getattr(block, "file_path", None) == ""
