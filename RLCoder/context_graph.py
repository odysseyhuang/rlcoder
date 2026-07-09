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

        selected = []
        seen = {block_key(block) for block in seed_blocks}
        source_counts = Counter()

        for seed_rank, seed in enumerate(seed_blocks[:max_seed]):
            if len(selected) >= max_total:
                break
            if _is_stop_block(seed):
                continue

            neighbors = []
            neighbors.extend(self._same_file_neighbors(task_id, seed, max_neighbors))
            if enable_identifier:
                neighbors.extend(self._identifier_neighbors(task_id, seed, max_neighbors, args))
            if enable_import:
                neighbors.extend(self._import_neighbors(task_id, example, seed, max_neighbors))

            for neighbor, source in neighbors:
                if len(selected) >= max_total:
                    break
                key = block_key(neighbor)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(_copy_graph_block(neighbor, source, seed_rank))
                source_counts[source] += 1

        trace = {
            "graph_expanded": len(selected),
            "graph_sources": dict(source_counts),
            "graph_seed_count": min(max_seed, len(seed_blocks)),
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

    def _same_file_neighbors(self, task_id, seed, max_neighbors):
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
                neighbors.append((code_blocks[file_indices[file_pos - radius]], "graph_same_file"))
                if len(neighbors) >= max_neighbors:
                    break
            if file_pos + radius < len(file_indices):
                neighbors.append((code_blocks[file_indices[file_pos + radius]], "graph_same_file"))
            radius += 1

        return neighbors[:max_neighbors]

    def _identifier_neighbors(self, task_id, seed, max_neighbors, args):
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
        if not seed_identifiers:
            return []

        seed_key = block_key(seed)
        scores = Counter()
        for token in seed_identifiers[:32]:
            for idx in identifier_index.get(token, []):
                block = self.code_blocks_by_task[task_id][idx]
                if block_key(block) != seed_key:
                    scores[idx] += 1

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            (self.code_blocks_by_task[task_id][idx], "graph_identifier")
            for idx, _ in ranked[:max_neighbors]
        ]

    def _import_neighbors(self, task_id, example, seed, max_neighbors):
        if max_neighbors <= 0:
            return []

        self._ensure_path_index(task_id)
        path_index = self.path_index_by_task.get(task_id, {})
        import_tokens = _extract_import_tokens(getattr(example, "left_context", ""))
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
            (self.code_blocks_by_task[task_id][idx], "graph_import")
            for idx, _ in ranked[:max_neighbors]
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


def _copy_graph_block(block, source, seed_rank):
    copied = copy.copy(block)
    copied._type = source
    copied._ucm_sources = (source,)
    copied._ucm_graph_seed_rank = seed_rank
    return copied


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
