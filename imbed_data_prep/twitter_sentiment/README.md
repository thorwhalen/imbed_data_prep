# Twitter Sentiment

Data preparation for the Sentiment140 Twitter dataset.

## Data source

[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) is a
benchmark dataset of 1.6 million tweets labeled with sentiment polarity,
created by Stanford researchers in 2009.

- **800,000 positive tweets** (target = 4)
- **800,000 negative tweets** (target = 0)
- Neutral tweets (target = 2) excluded in most versions

Labels were assigned automatically using emoticons as noisy labels:
positive tweets contained `:)` and negative tweets contained `:(`.

The dataset is widely used for training and evaluating sentiment analysis
models in NLP research.

- **Kaggle:** `kazanova/sentiment140`
- **Hugging Face:** `stanfordnlp/sentiment140`

## What it does

1. **Load from Kaggle** -- fetch the dataset via the `haggle` package.
2. (Partially implemented) Compute embeddings and clustering.

## Output

A DataFrame with columns:

| Column | Description |
|---|---|
| `target` | Sentiment polarity: 0 (negative), 2 (neutral), 4 (positive) |
| `id` | Tweet ID |
| `date` | Tweet timestamp |
| `flag` | Query flag (usually "NO_QUERY") |
| `user` | Twitter username |
| `text` | Tweet text |

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
| `twitter_sentiment.ipynb` | Notebook exploring the Sentiment140 data |
