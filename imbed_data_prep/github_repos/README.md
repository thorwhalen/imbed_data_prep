# GitHub Repos

Data preparation for GitHub repository metadata.

## Data source

The dataset is a parquet file of GitHub repository metadata originally
sourced from Kaggle and mirrored on Dropbox.  It contains information about
public GitHub repositories including names, descriptions, star counts, and
other metadata.

- Kaggle datasets:
  [GitHub Repository Metadata](https://www.kaggle.com/datasets/pelmers/github-repository-metadata-with-5-stars),
  [GitHub Repository Metadata (isaacoresanya)](https://www.kaggle.com/datasets/isaacoresanya/github-repository-metadata)
- Download URL used by the module:
  `https://www.dropbox.com/s/kokiypcm2ylx4an/github-repos.parquet?dl=1`

## What it does

1. **Download and deduplicate** -- fetch the parquet file, remove duplicate
   repos by `nameWithOwner`.
2. **Extract text segments** -- pull out the text column used for embedding.
3. **Compute embeddings** -- generate embedding vectors for each repo's text.
4. **Project to 2D** -- reduce to planar coordinates (ncvis by default,
   UMAP as fallback).
5. **Cluster** -- run K-means to assign cluster labels.

## Output

A DataFrame indexed by `nameWithOwner` with columns:

| Column | Description |
|---|---|
| `nameWithOwner` | Repository identifier (e.g. `owner/repo`) |
| text column | Description or README text used for embedding |
| embedding columns | High-dimensional embedding vector |
| `x`, `y` | 2D projected coordinates |
| cluster columns | Cluster assignments |

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
| `github_repos.ipynb` | Notebook exploring the GitHub repos data |
