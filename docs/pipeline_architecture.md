# Pipeline Architecture Overview

```mermaid
flowchart TD
    subgraph INPUT["Input"]
        A[("News article\ndata")]
        B[("Image\ndata")]
    end

    subgraph SEED["Shared across all experiments"]
        S["5% seed · Test set · Evaluation metrics"]
    end

    subgraph TEXT["Text Pipeline"]
        A --> T1["Summarization\nBART\n3–4 sentences"]
        T1 --> T2["Feature embedding\nSentence-BERT"]
        A --> T3["Raw text\n(no summarization)"]
    end

    subgraph IMAGE["Image Pipeline"]
        B --> I1["Feature embedding\nCLIP ViT-B/32"]
    end

    subgraph CACHE["Storage"]
        T2 --> C[("embeddings_cache/")]
        I1 --> C
    end

    subgraph LABEL["Labeling Layer"]
        direction TB
        L1["Unsupervised\nKMeans · Agglomerative · DEC\n× MiniLM / RoBERTa"]
        L2["Semi-supervised\nDistilBERT · ELECTRA-small\nraw text input"]
        L3["Supervised\nbaseline"]
    end

    C --> L1
    T3 --> L2
    C --> L3

    subgraph EVAL["Evaluation Layer"]
        E1["Comparison table\nmetrics"]
        E2["Cluster description\nin natural language\n(CLIP text–image alignment)"]
    end

    L1 --> E1
    L2 --> E1
    L3 --> E1
    I1 --> E2

    E1 --> OUT["Gradio UI"]
    E2 --> OUT

    SEED -.-> TEXT
    SEED -.-> IMAGE
    SEED -.-> EVAL
```

## Design Notes

| Decision | Rationale |
|---|---|
| Summarize with BART before embedding | Original articles (400–800 words) exceed the optimal context window of sentence embeddings; BART distills core information and reduces noise |
| Semi-supervised models receive raw text | Preserves full contextual signal without summarization filtering |
| CLIP chosen for image embedding | Its embedding space is jointly aligned with text — a prerequisite for automatic cluster description in natural language |
| `embeddings_cache/` | Vectors are computed once and reused across all experiments |
