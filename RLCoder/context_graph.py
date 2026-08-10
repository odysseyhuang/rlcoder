import copy
import os
import re
from collections import Counter, defaultdict

from typed_dependency_graph import TypedDependencyIndex


IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

TYPED_RELATIONS_BY_MODE = {
    "all": frozenset(
        {"graph_typed_type", "graph_typed_call", "graph_typed_def_use"}
    ),
    "type_only": frozenset({"graph_typed_type"}),
    "call_only": frozenset({"graph_typed_call"}),
    "def_use_only": frozenset({"graph_typed_def_use"}),
}
IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+)|import\s+(.+)|package\s+([A-Za-z0-9_\.]+)|import\s+([A-Za-z0-9_\.]+)\s*;)")
QUALIFIED_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*\(")
SIMPLE_CALL_RE = re.compile(r"(?<![\.\w])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
NEW_CALL_RE = re.compile(r"\bnew\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

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

WEAK_IDENTIFIERS = {
    "arg", "args", "cfg", "config", "configs", "data", "default", "item",
    "items", "key", "kwargs", "obj", "option", "options", "param", "params",
    "result", "results", "self", "test", "tests", "tmp", "value", "values",
}


class ContextGraphIndex:
    def __init__(self, code_blocks_by_task):
        self.code_blocks_by_task = code_blocks_by_task
        self.position_by_task = {}
        self.file_indices_by_task = {}
        self.identifier_index_by_task = {}
        self.path_index_by_task = {}
        self.api_call_index_by_task = {}
        self.identifier_df_by_task = {}
        self.api_call_df_by_task = {}
        self.typed_dependency_index = TypedDependencyIndex(code_blocks_by_task)
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
        enable_api_call = getattr(args, "ucm_graph_enable_api_call_edges", False)
        enable_same_file = not getattr(args, "ucm_graph_disable_same_file_edges", False)
        enable_typed_dependency = getattr(
            args, "ucm_graph_enable_typed_dependency_edges", False
        )
        query_identifiers = set(_extract_identifiers(getattr(example, "left_context", "")))
        query_import_tokens = _extract_import_tokens(getattr(example, "left_context", ""))
        query_api_tokens = _extract_api_call_tokens(getattr(example, "left_context", ""))
        same_file_direction = getattr(args, "ucm_graph_same_file_direction", "both")
        identifier_query_only = getattr(args, "ucm_graph_identifier_query_only", False)
        api_call_query_only = getattr(args, "ucm_graph_api_call_query_only", False)

        seen = {block_key(block) for block in seed_blocks}
        proposals = {}
        proposed_counts = Counter()
        source_counts = Counter()

        def add_proposals(neighbors, seed_rank, seed_weight):
            for edge in neighbors:
                neighbor, source, edge_score, edge_distance = edge[:4]
                metadata = edge[4] if len(edge) > 4 else {}
                key = block_key(neighbor)
                is_typed = source.startswith("graph_typed_")
                if key in seen and not is_typed:
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
                        "metadata": metadata,
                        "existing_candidate": key in seen,
                    }

        if enable_typed_dependency:
            query_lines = max(
                1, getattr(args, "ucm_graph_typed_query_context_lines", 80)
            )
            query_text = "\n".join(
                getattr(example, "left_context", "").splitlines()[-query_lines:]
            )
            query_neighbors = self._typed_dependency_neighbors(
                example,
                query_text,
                max(0, getattr(args, "ucm_graph_typed_query_max", 8)),
                args,
                origin="query",
            )
            add_proposals(query_neighbors, -1, 1.0)

        for seed_rank, seed in enumerate(seed_blocks[:max_seed]):
            if _is_stop_block(seed):
                continue

            neighbors = []
            if enable_same_file:
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
            if enable_api_call:
                neighbors.extend(
                    self._api_call_neighbors(
                        task_id,
                        seed,
                        max_neighbors,
                        query_api_tokens,
                        args,
                    )
                )
            if enable_typed_dependency:
                neighbors.extend(
                    self._typed_dependency_neighbors(
                        example,
                        getattr(seed, "code_content", ""),
                        max_neighbors,
                        args,
                        origin="seed",
                        seed=seed,
                    )
                )

            seed_weight = _seed_rank_weight(seed_rank, args)
            add_proposals(neighbors, seed_rank, seed_weight)

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
        relation_counts = Counter()
        origin_counts = Counter()
        matched_symbols = Counter()
        annotated_existing = 0
        for item in ranked[:max_total]:
            metadata = item.get("metadata", {})
            selected.append(
                _copy_graph_block(
                    item["block"],
                    item["source"],
                    item["seed_rank"],
                    item["score"],
                    item["edge_distance"],
                    metadata,
                )
            )
            source_counts[item["source"]] += 1
            scores.append(item["score"])
            if item.get("existing_candidate"):
                annotated_existing += 1
            relation = metadata.get("relation")
            if relation:
                relation_counts[relation] += 1
            origin = metadata.get("origin")
            if origin:
                origin_counts[origin] += 1
            matched_symbols.update(metadata.get("matched_symbols", ()))

        trace = {
            "graph_expanded": len(selected),
            "graph_sources": dict(source_counts),
            "graph_proposed_sources": dict(proposed_counts),
            "graph_proposed_candidates": len(proposals),
            "graph_new_candidates": len(selected) - annotated_existing,
            "graph_annotated_existing": annotated_existing,
            "graph_seed_count": min(max_seed, len(seed_blocks)),
            "graph_query_identifier_count": len(query_identifiers),
            "graph_query_import_token_count": len(query_import_tokens),
            "graph_query_api_call_count": len(query_api_tokens),
            "graph_same_file_direction": same_file_direction,
            "graph_identifier_query_only": identifier_query_only,
            "graph_api_call_query_only": api_call_query_only,
            "graph_typed_dependency_enabled": enable_typed_dependency,
            "graph_typed_relation_mode": getattr(
                args, "ucm_graph_typed_relation_mode", "all"
            ),
            "graph_typed_relations": dict(relation_counts),
            "graph_typed_origins": dict(origin_counts),
            "graph_typed_matched_symbols": dict(matched_symbols),
            "graph_score_min": round(min(scores), 6) if scores else None,
            "graph_score_max": round(max(scores), 6) if scores else None,
            "graph_score_avg": round(sum(scores) / len(scores), 6) if scores else None,
        }
        return selected, trace

    def _typed_dependency_neighbors(
        self,
        example,
        source_text,
        max_results,
        args,
        origin,
        seed=None,
    ):
        relation_mode = getattr(args, "ucm_graph_typed_relation_mode", "all")
        allowed_relations = TYPED_RELATIONS_BY_MODE.get(relation_mode)
        if allowed_relations is None:
            raise ValueError(f"Unknown typed dependency relation mode: {relation_mode}")
        weights = {
            "graph_typed_call": getattr(args, "ucm_graph_typed_call_weight", 1.8),
            "graph_typed_type": getattr(args, "ucm_graph_typed_type_weight", 2.0),
            "graph_typed_def_use": getattr(
                args, "ucm_graph_typed_def_use_weight", 1.4
            ),
        }
        exclude_file_path = ""
        if not getattr(args, "ucm_graph_typed_allow_target_file", False):
            exclude_file_path = getattr(example, "file_path", "")
        matches = self.typed_dependency_index.neighbors(
            example.task_id,
            source_text,
            getattr(example, "language", ""),
            max_results=max_results,
            max_df=max(1, getattr(args, "ucm_graph_typed_max_df", 12)),
            weights=weights,
            query_bonus=getattr(args, "ucm_graph_typed_query_bonus", 1.0),
            origin=origin,
            exclude_block_key=block_key(seed) if seed is not None else None,
            exclude_file_path=exclude_file_path,
            allowed_relations=allowed_relations,
        )
        return [
            (
                match["block"],
                match["relation"],
                match["score"],
                1,
                {
                    "relation": match["relation"],
                    "matched_symbols": match["symbols"],
                    "origin": match["origin"],
                },
            )
            for match in matches
        ]

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
        direction = getattr(args, "ucm_graph_same_file_direction", "both")
        if direction not in {"both", "prev", "next"}:
            direction = "both"

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
            if direction in {"both", "prev"} and file_pos - radius >= 0:
                block = code_blocks[file_indices[file_pos - radius]]
                neighbors.append((
                    block,
                    "graph_same_file",
                    _same_file_score(block, radius, query_identifiers, args),
                    radius,
                ))
                if len(neighbors) >= max_neighbors:
                    break
            if direction in {"both", "next"} and file_pos + radius < len(file_indices):
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
            if token not in WEAK_IDENTIFIERS and 1 < identifier_df.get(token, 0) <= max_df
        ]
        focused_identifiers = [
            token for token in seed_identifiers
            if token in query_identifiers
        ]
        if getattr(args, "ucm_graph_identifier_query_only", False):
            seed_identifiers = focused_identifiers
        elif focused_identifiers:
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
                "graph_import_path",
                _import_score(score, args),
                1,
            )
            for idx, score in ranked[:max_neighbors]
        ]

    def _api_call_neighbors(self, task_id, seed, max_neighbors, query_api_tokens, args):
        if max_neighbors <= 0:
            return []

        self._ensure_api_call_index(task_id)
        api_call_index = self.api_call_index_by_task.get(task_id, {})
        api_call_df = self.api_call_df_by_task.get(task_id, {})
        max_df = getattr(args, "ucm_graph_api_call_max_df", 20)

        seed_tokens = [
            token for token in _extract_api_call_tokens(getattr(seed, "code_content", ""))
            if 1 < api_call_df.get(token, 0) <= max_df
        ]
        focused_tokens = [
            token for token in seed_tokens
            if token in query_api_tokens
        ]
        if getattr(args, "ucm_graph_api_call_query_only", False):
            seed_tokens = focused_tokens
        elif focused_tokens:
            seed_tokens = focused_tokens
        if not seed_tokens:
            return []

        seed_key = block_key(seed)
        scores = Counter()
        for token in seed_tokens[:32]:
            df = max(1, api_call_df.get(token, 1))
            low_df_bonus = 1.0 / df
            for idx in api_call_index.get(token, []):
                block = self.code_blocks_by_task[task_id][idx]
                if block_key(block) == seed_key:
                    continue
                scores[idx] += 1.0 + low_df_bonus
                if token in query_api_tokens:
                    scores[idx] += getattr(args, "ucm_graph_query_api_bonus", 2)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            (
                self.code_blocks_by_task[task_id][idx],
                "graph_api_call",
                _api_call_score(score, args),
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

    def _ensure_api_call_index(self, task_id):
        if task_id in self.api_call_index_by_task:
            return

        api_call_index = defaultdict(list)
        api_call_df = Counter()
        for idx, block in enumerate(self.code_blocks_by_task.get(task_id, [])):
            api_tokens = set(_extract_api_call_tokens(getattr(block, "code_content", "")))
            for token in api_tokens:
                api_call_index[token].append(idx)
                api_call_df[token] += 1

        self.api_call_index_by_task[task_id] = dict(api_call_index)
        self.api_call_df_by_task[task_id] = api_call_df


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
    by_key = {block_key(block): block for block in merged}
    for block in expanded:
        key = block_key(block)
        if key in by_key:
            if getattr(block, "_type", "").startswith("graph_typed_"):
                _merge_typed_graph_metadata(by_key[key], block)
            continue
        by_key[key] = block
        merged.append(block)
    return merged


def _copy_graph_block(
    block,
    source,
    seed_rank,
    score=None,
    edge_distance=None,
    metadata=None,
):
    copied = copy.copy(block)
    copied._type = source
    copied._ucm_sources = (source,)
    copied._ucm_graph_seed_rank = seed_rank
    copied._ucm_graph_score = score
    copied._ucm_graph_edge_distance = edge_distance
    metadata = metadata or {}
    copied._ucm_graph_relation = metadata.get("relation")
    copied._ucm_graph_matched_symbols = tuple(
        metadata.get("matched_symbols", ())
    )
    copied._ucm_graph_origin = metadata.get("origin")
    return copied


def _merge_typed_graph_metadata(target, source):
    existing_sources = tuple(getattr(target, "_ucm_sources", ()))
    typed_source = getattr(source, "_type", "")
    if typed_source and typed_source not in existing_sources:
        target._ucm_sources = existing_sources + (typed_source,)

    existing_score = getattr(target, "_ucm_graph_score", None)
    source_score = getattr(source, "_ucm_graph_score", None)
    if existing_score is not None and source_score is not None and existing_score >= source_score:
        return

    for attr in (
        "_ucm_graph_seed_rank",
        "_ucm_graph_score",
        "_ucm_graph_edge_distance",
        "_ucm_graph_relation",
        "_ucm_graph_matched_symbols",
        "_ucm_graph_origin",
    ):
        setattr(target, attr, getattr(source, attr, None))


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


def _api_call_score(overlap_count, args):
    base = getattr(args, "ucm_graph_api_call_weight", 1.6)
    return base + max(0, overlap_count)


def _query_identifier_overlap(block, query_identifiers):
    if not query_identifiers:
        return 0.0
    block_identifiers = set(_extract_identifiers(getattr(block, "code_content", "")))
    overlap = len(block_identifiers & query_identifiers)
    return min(2.0, 0.25 * overlap)


def _graph_source_priority(source):
    return {
        "graph_typed_type": 0,
        "graph_typed_call": 1,
        "graph_typed_def_use": 2,
        "graph_api_call": 3,
        "graph_import_path": 4,
        "graph_identifier": 5,
        "graph_same_file": 6,
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
                if not part or part in {"as", "*"} or part in KEYWORDS:
                    continue
                tokens.update(_module_tokens(part))
    return tokens


def _extract_api_call_tokens(text):
    tokens = []
    seen = set()

    for match in QUALIFIED_CALL_RE.finditer(text or ""):
        qualified = match.group(1)
        parts = qualified.split(".")
        if not _valid_api_token(parts[-1]):
            continue
        if not all(_valid_api_token(part, allow_weak=True) for part in parts):
            continue
        _append_token(tokens, seen, qualified)
        _append_token(tokens, seen, parts[-1])

    for regex in (NEW_CALL_RE, SIMPLE_CALL_RE):
        for match in regex.finditer(text or ""):
            token = match.group(1)
            if _valid_api_token(token):
                _append_token(tokens, seen, token)

    return tokens


def _append_token(tokens, seen, token):
    if token not in seen:
        seen.add(token)
        tokens.append(token)


def _valid_api_token(token, allow_weak=False):
    if not token or len(token) <= 2:
        return False
    if token in KEYWORDS:
        return False
    if not allow_weak and token in WEAK_IDENTIFIERS:
        return False
    return bool(IDENTIFIER_RE.fullmatch(token))


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
