# Trump vs Zelenskyy

Data preparation and analysis of the Trump-Zelenskyy meeting transcript.

## Data source

The transcript is from the widely publicized February/March 2025 meeting
between US President Donald Trump and Ukrainian President Volodymyr
Zelenskyy.  The raw text is hosted on GitHub:

`https://raw.githubusercontent.com/thorwhalen/content/refs/heads/master/text/trump-zelensky-2025-03-01--with_speakers.txt`

The transcript contains 858 speaking turns from 15 speakers, dominated by:

| Speaker | Turns |
|---|---|
| Trump | 509 |
| Zelenskyy | 251 |
| Vance | 32 |
| Others | 66 |

## What it does

1. **Fetch transcript** -- download the raw text from GitHub.
2. **Parse speakers** -- extract `[speaker]: text` patterns via regex.
3. **Consolidate speakers** -- collapse minor speakers into "Other".
4. **Compute embeddings** -- OpenAI embeddings for each speaking turn.
5. **Dimensionality reduction**:
   - **t-SNE** -- unsupervised 2D projection
   - **LDA** -- speaker-discriminative projection (2D and 1D)
   - **PCA** -- linear projection
6. **Sentiment analysis**:
   - **VADER** -- rule-based sentiment (neg, neu, pos, compound)
   - **text2emotion** -- emotion detection (Happy, Angry, Surprise, Sad, Fear)

## Output

A DataFrame with columns:

| Column | Description |
|---|---|
| `speaker` | Speaker name (Trump, Zelenskyy, Vance, Other) |
| `text` | Speaking turn text |
| `turn` | Turn number in conversation |
| `tsne_x`, `tsne_y` | t-SNE 2D coordinates |
| `lda_x`, `lda_y` | LDA 2D coordinates |
| `single_speaker_lda` | LDA 1D projection |
| `pca_1`, `pca_2` | PCA 2D coordinates |
| `neg`, `neu`, `pos`, `compound` | VADER sentiment scores |
| `Happy`, `Angry`, `Surprise`, `Sad`, `Fear` | text2emotion scores |

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
| `trump_vs_zelenskyy.ipynb` | Main analysis notebook |
| `trump_vs_zelensky.md` | Data preparation walkthrough (markdown with code) |
| `trump_vs_zelenskyy_transcript.parquet` | Processed transcript DataFrame |
| `trump_vs_zelenskyy_embeddings.parquet` | Pre-computed embeddings |
