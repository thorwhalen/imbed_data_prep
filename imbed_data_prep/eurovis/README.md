# EuroVis

Data preparation for EuroVis (Eurographics Conference on Visualization)
publication data.

## Data source

[EuroVis](https://www.eg.org/wp/eurovis/) is the annual visualization
conference organized by the Eurographics Working Group on Data Visualization,
co-supported by Eurographics and IEEE.  It has been held in Europe annually
since 1999 (as a conference since 2012) and is a premier venue for research
in visualization, from theoretical foundations to applied techniques.

The raw data is a CSV file (`raw_data.csv`) containing publication metadata
with titles and abstracts from EuroVis proceedings.  A related public
resource is [VisPubData](https://www.vispubdata.org/), which catalogues
publications from IEEE VIS, EuroVis, and CHI.

## What it does

The pipeline processes EuroVis papers through a standard
embed-project-cluster workflow:

1. **Load raw data** -- read CSV with paper titles and abstracts.
2. **Build embeddable segments** -- combine title and abstract into a single
   text segment, filtering by token-count limits.
3. **Compute embeddings** -- generate OpenAI embeddings for each segment.
4. **Project to 2D** -- reduce to planar coordinates using the `imbed`
   library (UMAP / ncvis).
5. **Cluster** -- run K-means at multiple resolutions (e.g. 5, 8, 13, 21
   clusters).
6. **Merge** -- combine all artifacts into a single DataFrame.

## Output

A merged DataFrame indexed by DOI with columns:

| Column | Description |
|---|---|
| `title` | Paper title |
| `abstract` | Paper abstract |
| `segment` | Combined title + abstract text |
| `n_tokens` | Token count of the segment |
| `x`, `y` | 2D projected coordinates |
| `cluster_5`, `cluster_8`, ... | Cluster assignments at various K values |

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
| `eurovis.ipynb` | Notebook exploring and visualizing the EuroVis data |
