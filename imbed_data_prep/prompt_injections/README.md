# Prompt Injections

Data preparation for prompt injection detection research.

## Data source

[deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections)
is a text classification dataset for detecting prompt injection attacks on
LLMs.  It contains 662 prompts:

- **263 prompt injections** -- adversarial inputs attempting to manipulate
  the LLM
- **399 legitimate requests** -- benign queries and keyword searches

The dataset includes adversarial examples created by stacking legitimate and
injection prompts, plus translations into other languages (primarily German)
to prevent language-switching bypasses.

A fine-tuned DeBERTa model trained on this dataset achieves 99.1% accuracy.

## What it does

1. **Load from Hugging Face** -- download the dataset via the `datasets`
   library.
2. **Compute embeddings** -- batch-compute embeddings with chunked saves.
3. **Project to 2D** -- UMAP projection of the embedding space.

## Output

A DataFrame with columns:

| Column | Description |
|---|---|
| `text` | Prompt text |
| `label` | Binary label (0 = legitimate, 1 = injection) |
| embedding columns | High-dimensional embedding vector |
| `umap_x`, `umap_y` | 2D UMAP coordinates |

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
| `prompt_injection_w_umap_embeddings.tsv` | Pre-computed embeddings and UMAP coordinates |
