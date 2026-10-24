"""
nb14 Section 3 — Compare LexRank vs Abstractive semantic scores.
Lightweight — no model inference, just CSV/JSON reads.

Run after both section1 and section2 complete.

Usage:
    uv run python scripts/nb14_section3_compare.py
"""
import sys, json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from utils import config


def main():
    label_files = sorted(config.RESULTS_DIR.glob("cluster_labels_*.npy"))
    all_methods = [f.stem.replace("cluster_labels_", "") for f in label_files]

    lex_csv = pd.read_csv(config.RESULTS_DIR / "semantic_evaluation.csv")

    rows = []
    for method in all_methods:
        mapping_path = config.RESULTS_DIR / f"cluster_category_mapping_abstractive_{method}.json"
        metrics_path = config.RESULTS_DIR / f"metrics_{method}.json"

        if not mapping_path.exists() or not metrics_path.exists():
            print(f"  skipping {method} (missing mapping or metrics file)")
            continue

        metrics = json.load(open(metrics_path))
        with open(mapping_path, encoding="utf-8") as f:
            raw = json.load(f)
        mapping = {int(k): v for k, v in raw.items()}

        clusters_df  = pd.read_csv(config.RESULTS_DIR / f"clusters_{method}.csv")
        majority_map = dict(zip(clusters_df["cluster"], clusters_df["majority_true_label"]))

        abs_sims = [v["similarity"] for v in mapping.values()]
        abs_mean = float(np.mean(abs_sims)) if abs_sims else 0.0
        abs_acc  = sum(
            1 for cid, v in mapping.items()
            if v["mapped_category"] == majority_map.get(cid, "")
        ) / len(mapping) if mapping else 0.0

        lex_row  = lex_csv[lex_csv["method"] == method]
        lex_mean = float(lex_row["mean_semantic_score"].values[0]) if len(lex_row) else 0.0
        lex_macc = float(lex_row["mapping_accuracy"].values[0])    if len(lex_row) else 0.0

        rows.append({
            "method":            method,
            "ACC_hungarian":     round(metrics["ACC (Hungarian)"], 4),
            "lexrank_sem_score": round(lex_mean, 4),
            "lexrank_map_acc":   round(lex_macc, 4),
            "abs_sem_score":     round(abs_mean, 4),
            "abs_map_acc":       round(abs_acc,  4),
            "sem_score_delta":   round(abs_mean - lex_mean, 4),
        })

    if not rows:
        print("No complete results found.")
        return

    compare_df = pd.DataFrame(rows).sort_values("abs_sem_score", ascending=False)
    out_csv    = config.RESULTS_DIR / "semantic_evaluation_abstractive.csv"
    compare_df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}\n")

    print(f"{'Method':<35} {'ACC':>7} {'Lex Sim':>9} {'Abs Sim':>9} {'Delta':>7} {'Abs MapACC':>11}")
    print("-" * 85)
    for _, row in compare_df.iterrows():
        delta_str = f"+{row['sem_score_delta']:.4f}" if row['sem_score_delta'] >= 0 else f"{row['sem_score_delta']:.4f}"
        print(
            f"{row['method']:<35} {row['ACC_hungarian']:>7.4f} "
            f"{row['lexrank_sem_score']:>9.4f} {row['abs_sem_score']:>9.4f} "
            f"{delta_str:>7} {row['abs_map_acc']:>11.4f}"
        )

    print(f"\nBest abstractive semantic score : {compare_df.iloc[0]['method']} ({compare_df.iloc[0]['abs_sem_score']:.4f})")
    improved = (compare_df["sem_score_delta"] > 0).sum()
    print(f"Methods improved by abstractive : {improved}/{len(compare_df)}")
    print(f"Mean delta                      : {compare_df['sem_score_delta'].mean():+.4f}")


if __name__ == "__main__":
    main()
