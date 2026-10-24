"""
nb14 Section 2 — Semantic category mapping using abstractive descriptions.
No T5 model loaded here — only sentence-transformers (all-mpnet-base-v2).

Run after nb14_section1_descriptions.py completes.

Usage:
    uv run python scripts/nb14_section2_mapping.py
"""
import sys, json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import config

CANONICAL_DESCRIPTIONS = {
    "ARTS & CULTURE":  "Articles about visual arts, performing arts, music, literature, cultural events, and creative expression.",
    "BUSINESS":        "Articles about corporate finance, markets, economics, trade, and business strategy.",
    "COMEDY":          "Articles about stand-up comedy, satirical shows, late-night television hosts, and humorous commentary on current events.",
    "CRIME":           "Articles about criminal investigations, arrests, court cases, and law enforcement activities.",
    "EDUCATION":       "Articles about schools, universities, student loans, teachers, and educational policy.",
    "ENTERTAINMENT":   "Articles about movies, television shows, celebrity news, awards ceremonies, and pop culture.",
    "ENVIRONMENT":     "Articles about climate change, environmental policy, wildlife, natural disasters, and conservation.",
    "HEALTH":          "Articles about medical research, public health, disease outbreaks, vaccines, and healthcare policy.",
    "MEDIA":           "Articles about journalism, news organisations, television networks, media industry controversies, and press freedom.",
    "NEWS":            "Breaking news and general current events not classified under a more specific category.",
    "POLITICS":        "Articles about elections, government policy, politicians, legislation, and political parties.",
    "RELIGION":        "Articles about religious institutions, faith, spirituality, religious leaders, and interfaith issues.",
    "SCIENCE":         "Articles about scientific discoveries, space exploration, NASA, astronomy, physics, and technology research.",
    "SPORTS":          "Articles about professional and amateur sports competitions, athletes, teams, tournaments, and sporting events.",
    "TECH":            "Articles about consumer technology, gadgets, software, social media platforms, and the tech industry.",
    "WOMEN":           "Articles about women's rights, gender equality, sexual harassment, feminism, and women in public life.",
}

assert list(CANONICAL_DESCRIPTIONS.keys()) == config.CLASS_NAMES


def main():
    label_files = sorted(config.RESULTS_DIR.glob("cluster_labels_*.npy"))
    all_methods = [f.stem.replace("cluster_labels_", "") for f in label_files]

    # Only methods with abstractive descriptions ready
    methods = [
        m for m in all_methods
        if (config.RESULTS_DIR / f"cluster_descriptions_abstractive_{m}.json").exists()
    ]

    if not methods:
        print("No abstractive description files found. Run nb14_section1_descriptions.py first.")
        return

    print(f"Methods to map ({len(methods)}): {methods}\n")

    print("Loading all-mpnet-base-v2 ...")
    embed_model    = SentenceTransformer("all-mpnet-base-v2")
    category_names = list(CANONICAL_DESCRIPTIONS.keys())
    category_texts = list(CANONICAL_DESCRIPTIONS.values())
    category_embs  = embed_model.encode(
        category_texts, batch_size=16, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    print(f"Category embeddings: {category_embs.shape}\n")

    for i, method in enumerate(methods, 1):
        desc_path = config.RESULTS_DIR / f"cluster_descriptions_abstractive_{method}.json"
        out_path  = config.RESULTS_DIR / f"cluster_category_mapping_abstractive_{method}.json"

        with open(desc_path, encoding="utf-8") as f:
            raw = json.load(f)
        descs       = {int(k): v for k, v in raw.items()}
        cluster_ids = sorted(descs.keys())
        desc_texts  = [descs[c] for c in cluster_ids]

        cluster_embs = embed_model.encode(
            desc_texts, batch_size=16, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)

        sim_matrix = cluster_embs @ category_embs.T
        mapping = {}
        for j, cid in enumerate(cluster_ids):
            best_idx = int(np.argmax(sim_matrix[j]))
            mapping[cid] = {
                "mapped_category": category_names[best_idx],
                "similarity":      float(sim_matrix[j, best_idx]),
            }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in mapping.items()}, f, indent=2, ensure_ascii=False)
        print(f"[{i}/{len(methods)}] {method}: mappings saved → {out_path.name}")

    print("\nSection 2 complete.")


if __name__ == "__main__":
    main()
