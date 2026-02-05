# Embeddings of Aggregations

Experiments with aggregated embeddings over citation graphs.

## Data source

This module works with citation graph data and academic paper metadata.
It takes a set of nodes (papers) with embeddings and citation links, then
explores how aggregating the titles of cited papers and embedding those
aggregated strings compares to the original embeddings.

The data is expected to come from an external citation graph (e.g. Semantic
Scholar, OpenAlex, or a custom corpus) loaded as DataFrames with paper IDs,
titles, and citation edges.

## What it does

1. **Sample nodes** from a citation graph.
2. **Permute citations** -- for each citing paper, generate multiple random
   orderings of its cited papers.
3. **Aggregate titles** -- concatenate cited-paper titles in each permutation
   order into a single string.
4. **Embed aggregated strings** -- compute embeddings of these concatenated
   title strings.
5. **Compare** -- measure how the aggregated embeddings relate to the
   original paper embeddings, exploring whether citation context captures
   similar semantic information.

## Output

A DataFrame with columns:

| Column | Description |
|---|---|
| `citing_id` | ID of the citing paper |
| `n_cited` | Number of papers cited |
| `permutation_index` | Index of this particular citation ordering |
| `aggregated_title` | Concatenated cited-paper titles |
| `embedding` | Embedding vector of the aggregated title string |

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
| `embeddings_and_order.ipynb` | Notebook exploring aggregation experiments |
