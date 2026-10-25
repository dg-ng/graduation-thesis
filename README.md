# Automatic Label Generation Methods for Text and Image Data Based on Semi-Supervised and Unsupervised Learning

Graduation thesis of **Nguyễn Lợi Thanh Dũng**.

| | |
|---|---|
| University | University of Economics Ho Chi Minh City (UEH), College of Technology and Design |
| Faculty | Business Information Technology |
| Major | Data Science |
| Class / Cohort | DS0001 / K49 |
| Supervisor | Dr. Đặng Ngọc Hoàng Thành |
| Academic year | 09/2023 – 12/2026 |
| Submitted | Ho Chi Minh City, July 31, 2026 |

## Overview

Assign category labels to unlabeled data **without full human annotation**, and compare how far each automatic approach gets. The same framework is applied to two datasets:

- **Text**: news and opinion articles, 16 classes (~4,781 rows after cleaning)
- **Image**: animal photos, 10 classes (5,500 images)

A Gradio demo takes a paragraph or an image and suggests tags from an unsupervised and a semi-supervised pipeline.

## Approaches

- **Unsupervised clustering**: KMeans, Agglomerative and DEC on embeddings, with clusters matched to class names.
- **Semi-supervised**: weak supervision, label propagation and pseudo-labeling from a 5% labeled seed.
- **Full-supervised baseline**: trained on all training labels, as a reference.

The text track clusters embeddings of generated summaries. The image track uses CLIP embeddings.

## Headline results

| Track | Best automatic method | Accuracy |
|---|---|---|
| Text (16 classes) | Pseudo-labeling (DistilBERT) | 0.887 |
| Image (10 classes) | Unsupervised DEC | 0.994 |

Text classes overlap heavily (e.g. POLITICS, NEWS, CRIME, MEDIA), so text scores are much lower than image scores. Full tables are in `results/comparison_table.csv` and `results/image_comparison.csv`.

## Documentation

| Document | Covers |
|---|---|
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Text track: dataset, pipeline, methods, results |
| [PROJECT_SUMMARY_IMAGE.md](PROJECT_SUMMARY_IMAGE.md) | Image track: dataset, pipeline, methods, results |
| [docs/pipeline_architecture.md](docs/pipeline_architecture.md) | Pipeline architecture |
| [docs/semi_supervised_methods.md](docs/semi_supervised_methods.md) | Semi-supervised methods in detail |
| [autolabel_project_spec.md](autolabel_project_spec.md) | Original project spec |
| [docs/PROJECT_SUMMARY_AGNEWS_ARCHIVE.md](docs/PROJECT_SUMMARY_AGNEWS_ARCHIVE.md) | Archived 4-class AG News version (retired) |

## Repository layout

```
app/        Gradio demo and inference code
notebooks/  Text pipeline (00–15) and image pipeline (nb_img_*)
utils/      Shared code (config, embeddings, DEC, metrics, ...)
scripts/    Model export and summary generation
tests/      pytest suite
results/    Metrics, cluster mappings, labeled outputs
data/       Datasets, manifests, cached embeddings
```

## Quick start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/export_models.py   # save models for the demo
uv run python app/app.py                 # launch the demo
uv run pytest                            # run tests
```

Raw data (`data/raw/`, `data/image-data/`, `data/master_data.csv`) is not version-controlled. Place it in `data/` and run the notebooks in order; see the project summaries for the notebook sequence.
