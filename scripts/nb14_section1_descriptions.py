"""
nb14 Section 1 — Generate abstractive descriptions using FLAN-T5-large.
Processes one method at a time to keep RAM low. Safe to stop and resume
(each method caches immediately on completion).

Usage:
    uv run python scripts/nb14_section1_descriptions.py
    uv run python scripts/nb14_section1_descriptions.py --methods mpnet_kmeans mpnet_agglomerative
"""
import sys, gc, json, random, argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm.auto import tqdm

from utils import config

SAMPLE_SIZE = 100
BATCH_SIZE  = 15
RANDOM_SEED = 42


def t5_summarize(text: str, model, tokenizer, max_new_tokens: int = 128) -> str:
    prompt = (
        "Summarize the main topic of the following news articles "
        "in 1-2 general sentences, without mentioning specific names or events:\n\n"
        + text
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def abstractive_description(cluster_summaries, model, tokenizer):
    rng    = random.Random(RANDOM_SEED)
    sample = rng.sample(cluster_summaries, min(SAMPLE_SIZE, len(cluster_summaries)))

    mini_summaries = []
    for i in range(0, len(sample), BATCH_SIZE):
        batch_text = "\n- ".join(sample[i : i + BATCH_SIZE])
        mini_summaries.append(t5_summarize(batch_text, model, tokenizer, max_new_tokens=80))

    if len(mini_summaries) == 1:
        return mini_summaries[0]

    final_text = (
        "Summarize the main topic of the following news articles "
        "in 2-3 general sentences, without mentioning specific names or events:\n\n"
        + "\n".join(mini_summaries)
    )
    inputs = tokenizer(final_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=150, num_beams=4, early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", help="Specific methods to run (default: all pending)")
    args = parser.parse_args()

    label_files = sorted(config.RESULTS_DIR.glob("cluster_labels_*.npy"))
    all_methods = [f.stem.replace("cluster_labels_", "") for f in label_files]

    if args.methods:
        methods = [m for m in args.methods if m in all_methods]
    else:
        # Only methods that don't have a cache yet
        methods = [
            m for m in all_methods
            if not (config.RESULTS_DIR / f"cluster_descriptions_abstractive_{m}.json").exists()
        ]

    if not methods:
        print("All methods already cached. Nothing to do.")
        return

    print(f"Methods to process ({len(methods)}): {methods}\n")

    print("Loading train_clean.parquet ...")
    train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
    summaries   = train_clean["summary"].tolist()

    print("Loading google/flan-t5-large (loaded once, stays in RAM for all methods) ...")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model     = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    model.eval()
    print("Model ready.\n")

    for i, method in enumerate(methods, 1):
        out_path = config.RESULTS_DIR / f"cluster_descriptions_abstractive_{method}.json"

        if out_path.exists():
            print(f"[{i}/{len(methods)}] {method}: already cached, skipping.")
            continue

        print(f"[{i}/{len(methods)}] {method} ...")
        cluster_labels  = np.load(config.RESULTS_DIR / f"cluster_labels_{method}.npy")
        unique_clusters = sorted(set(cluster_labels))
        descs = {}

        for cid in tqdm(unique_clusters, desc=f"  clusters", leave=False):
            idx          = np.where(cluster_labels == cid)[0]
            cluster_sums = [summaries[j] for j in idx]
            descs[cid]   = abstractive_description(cluster_sums, model, tokenizer)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in descs.items()}, f, indent=2, ensure_ascii=False)
        print(f"  saved {len(descs)} descriptions → {out_path.name}")

        # free per-method data; model stays loaded
        del cluster_labels, descs
        gc.collect()

    print("\nSection 1 complete.")


if __name__ == "__main__":
    main()
