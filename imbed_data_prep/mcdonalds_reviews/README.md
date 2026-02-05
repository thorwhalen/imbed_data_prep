# McDonald's Reviews

Data preparation for McDonald's customer review data.

## Data source

The dataset is a CSV file of cleaned McDonald's restaurant reviews,
originally sourced from Kaggle.  Several McDonald's review datasets exist:

- [McDonald's Store Reviews](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews)
  -- over 33,000 customer reviews
- [McDonald's Review Sentiment](https://data.world/crowdflower/mcdonalds-review-sentiment)
  -- 1,525 labeled reviews from CrowdFlower

The expected input file is `McDonalds_Reviews_Cleaned.csv`.

## What it does

Standard embed-project-cluster pipeline plus mood modeling:

1. **Load raw data** -- read the cleaned CSV.
2. **Build embeddable segments** -- prepare review text for embedding.
3. **Compute embeddings** -- generate OpenAI embeddings.
4. **Project to 2D** -- reduce to planar coordinates.
5. **Cluster** -- run K-means at multiple resolutions.
6. **Merge** -- combine all artifacts into a single DataFrame.
7. **Mood modeling** (extended class) -- train semantic attribute classifiers
   on the embeddings to predict sentiment/mood dimensions.

## Output

A merged DataFrame with columns:

| Column | Description |
|---|---|
| review text column | Original review text |
| `segment` | Text segment used for embedding |
| `n_tokens` | Token count |
| embedding columns | High-dimensional embedding vector |
| `x`, `y` | 2D projected coordinates |
| cluster columns | Cluster assignments at various K values |

The extended `McdonaldsReviewsMoodModeling` class adds mood prediction
columns based on trained semantic attribute classifiers.

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
| `mcdonalds_reviews_dacc.ipynb` | Notebook exploring McDonald's reviews data |
