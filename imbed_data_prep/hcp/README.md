# HCP

Data preparation for Human Connectome Project publication and citation data.

## Data source

The [Human Connectome Project](https://www.humanconnectome.org/) (HCP) is a
large-scale NIH-funded effort to map the structural and functional
connectivity of the human brain.  The HCP-Young Adult study collected
high-resolution MRI data from 1,200 healthy adults aged 22--35.

This module does **not** work with the brain imaging data directly.  Instead
it works with the *publications* associated with HCP -- academic papers that
cite each other -- using their titles and embeddings to build citation graphs
and aggregated embedding representations.

The publication and citation data is loaded from a local file-based store
(pickle / parquet files) previously prepared from sources such as Semantic
Scholar or OpenAlex.

## What it does

1. **Load embeddings** -- read pre-computed HCP paper embeddings.
2. **Build citation graphs** -- construct citing-to-cited networks, filtering
   to papers that have embeddings.
3. **Aggregate embeddings** -- for each citing paper, compute mean embeddings
   of its cited papers using different strategies (cosine-weighted,
   normalized, simple mean).
4. **Project to 2D** -- UMAP reduction of the aggregated embeddings.
5. **Combine metadata** -- merge titles, citation counts, and 2D coordinates.

## Output

Multiple artifacts:

| Artifact | Description |
|---|---|
| Citation graph dict | `{citing_id: [cited_ids]}` |
| Aggregated embeddings | Mean embeddings of cited papers per citing paper |
| Planar coordinates | 2D UMAP projection of aggregated embeddings |
| Merged DataFrame | Titles, citation counts, and coordinates combined |

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
| `hcp_analysis.ipynb` | Notebook exploring HCP citation networks |
