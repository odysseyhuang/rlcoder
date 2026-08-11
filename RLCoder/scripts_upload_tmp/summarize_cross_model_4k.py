#!/usr/bin/env python3
"""Summarize the X4K cross-model generalization matrix."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


DATASETS = ("cceval_python", "cceval_java", "repoeval_line", "repoeval_api")
CORE_MODELS = ("deepseekcoder_7b", "codellama_7b", "starcoderbase_7b")
PAIR_ORDER = (("M1", "M0"), ("M2", "M1"), ("M3", "M2"), ("M2", "M0"))
PERCENT_METRICS = {"em", "id_em", "id_f1"}


def load_detail(path: Path, metric: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["task_id"])] = float(row.get(metric, 0.0))
    return rows


def discover_runs(result_root: Path) -> dict[str, dict[str, Path]]:
    runs: dict[str, dict[str, Path]] = defaultdict(dict)
    pattern = re.compile(r"^X4K_(?P<model>.+)_(?P<group>M[0-3])_")
    for path in result_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        runs[match.group("model")][match.group("group")] = path
    return runs


def dataset_score(run_dir: Path, dataset: str, metric: str) -> float | None:
    detail_path = run_dir / dataset / "detailed_results.json"
    if not detail_path.exists():
        return None
    rows = load_detail(detail_path, metric)
    if not rows:
        return None
    return sum(rows.values()) / len(rows) * (100.0 if metric in PERCENT_METRICS else 1.0)


def paired_dataset_delta(
    run_a: Path, run_b: Path, dataset: str, metric: str
) -> tuple[list[float], int, int, int]:
    a_path = run_a / dataset / "detailed_results.json"
    b_path = run_b / dataset / "detailed_results.json"
    if not a_path.exists() or not b_path.exists():
        return [], 0, 0, 0
    a_rows = load_detail(a_path, metric)
    b_rows = load_detail(b_path, metric)
    task_ids = sorted(set(a_rows) & set(b_rows))
    diffs = [a_rows[task_id] - b_rows[task_id] for task_id in task_ids]
    if metric in PERCENT_METRICS:
        diffs = [diff * 100.0 for diff in diffs]
    wins = sum(1 for diff in diffs if diff > 0)
    losses = sum(1 for diff in diffs if diff < 0)
    ties = sum(1 for diff in diffs if diff == 0)
    return diffs, wins, losses, ties


def bootstrap_macro_ci(
    dataset_diffs: dict[str, list[float]], rounds: int, seed: int
) -> tuple[float, float] | None:
    usable = [diffs for diffs in dataset_diffs.values() if diffs]
    if not usable:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(rounds):
        dataset_means = []
        for diffs in usable:
            total = 0.0
            for _ in diffs:
                total += diffs[rng.randrange(len(diffs))]
            dataset_means.append(total / len(diffs))
        samples.append(sum(dataset_means) / len(dataset_means))
    samples.sort()
    lo = samples[int(0.025 * (rounds - 1))]
    hi = samples[int(0.975 * (rounds - 1))]
    return lo, hi


def sign_test_p_value(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    if n <= 200:
        prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        return min(1.0, 2.0 * prob)
    mean = n / 2.0
    std = math.sqrt(n / 4.0)
    z = (k + 0.5 - mean) / std
    return min(1.0, 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0)))


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root", default="result_infer", help="Directory containing X4K result runs"
    )
    parser.add_argument("--metric", default="em", choices=["em", "es", "id_em", "id_f1"])
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    runs = discover_runs(result_root)
    if not runs:
        raise SystemExit(f"No X4K result directories found under {result_root}")

    print(f"# X4K Cross-Model Summary ({args.metric})")
    print()
    print("## Scores")
    print("| model | group | cceval_python | cceval_java | repoeval_line | repoeval_api | macro |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for model in sorted(runs):
        for group in ("M0", "M1", "M2", "M3"):
            run_dir = runs[model].get(group)
            if not run_dir:
                continue
            scores = [dataset_score(run_dir, dataset, args.metric) for dataset in DATASETS]
            macro_values = [score for score in scores if score is not None]
            macro = sum(macro_values) / len(macro_values) if macro_values else None
            print(
                f"| {model} | {group} | "
                + " | ".join(fmt(score) for score in scores)
                + f" | {fmt(macro)} |"
            )

    print()
    print("## Paired Deltas")
    print("| model | delta | cceval_python | cceval_java | repoeval_line | repoeval_api | macro | wins | losses | ties | p_sign | macro_ci95 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    pair_model_results: dict[tuple[str, str], tuple[float, float, float]] = {}
    for model in sorted(runs):
        for high, low in PAIR_ORDER:
            if high not in runs[model] or low not in runs[model]:
                continue
            dataset_diffs: dict[str, list[float]] = {}
            dataset_means = []
            wins = losses = ties = 0
            for dataset in DATASETS:
                diffs, ds_wins, ds_losses, ds_ties = paired_dataset_delta(
                    runs[model][high], runs[model][low], dataset, args.metric
                )
                dataset_diffs[dataset] = diffs
                if diffs:
                    dataset_means.append(sum(diffs) / len(diffs))
                wins += ds_wins
                losses += ds_losses
                ties += ds_ties
            macro = sum(dataset_means) / len(dataset_means) if dataset_means else None
            ci = bootstrap_macro_ci(dataset_diffs, args.bootstrap_rounds, args.seed)
            p_value = sign_test_p_value(wins, losses)
            if macro is not None and ci is not None:
                pair_model_results[(model, f"{high}-{low}")] = (macro, ci[0], ci[1])
            ds_cells = [
                fmt(sum(dataset_diffs[d]) / len(dataset_diffs[d]) if dataset_diffs[d] else None)
                for d in DATASETS
            ]
            ci_text = "NA" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            print(
                f"| {model} | {high}-{low} | "
                + " | ".join(ds_cells)
                + f" | {fmt(macro)} | {wins} | {losses} | {ties} | {fmt(p_value)} | {ci_text} |"
            )

    print()
    print("## Generalization Check")
    print("| delta | positive_core_models | significant_negative_core_models | pass_rule |")
    print("|---|---:|---:|---:|")
    for pair_name in ("M1-M0", "M2-M1", "M3-M2", "M2-M0"):
        present = [
            pair_model_results[(model, pair_name)]
            for model in CORE_MODELS
            if (model, pair_name) in pair_model_results
        ]
        if not present:
            continue
        positive = sum(1 for macro, _, _ in present if macro > 0)
        significant_negative = sum(1 for _, _, hi in present if hi < 0)
        required = math.ceil(len(present) * 2 / 3)
        pass_rule = positive >= required and significant_negative == 0
        print(f"| {pair_name} | {positive}/{len(present)} | {significant_negative} | {pass_rule} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
