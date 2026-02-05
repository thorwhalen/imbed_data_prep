# WordNet Words

Data preparation for WordNet lexical data and word embeddings.

## Data source

[WordNet](https://wordnet.princeton.edu/) is a large lexical database of
English developed at Princeton University.  Words are grouped into
**synsets** (synonym sets) representing distinct concepts, linked by
semantic relations (hypernymy, hyponymy, meronymy, etc.).

Additional data sources:

- **English word frequency list:**
  `https://github.com/thorwhalen/content/.../english-word-frequency.csv.zip`
- **GlossTag (sense-tagged glosses):**
  `https://wordnetcode.princeton.edu/glosstag-files/WordNet-3.0-glosstag.zip`

## Architecture: Synset-Based Indexing

This module uses **synsets as the primary index** for metadata and link data,
avoiding the edge explosion that comes from replicating concept-to-concept
edges across all lemma (word) combinations.

### Key data structures

| Structure | Description |
|---|---|
| `wordnet_metadata` | Synset-indexed DataFrame with definitions, POS, lemmas list |
| `word_and_synset` | Bipartite mapping between words and synsets |
| `synset_definition_links` | Synset→synset edges extracted from definitions |
| `glosstag_edges` | Synset→synset links from sense-tagged glosses |
| `word_indexed_metadata` | Word-indexed view with one representative synset per word |
| `wordnet_collection_meta` | Relationship metadata (hypernyms, hyponyms, etc.) |

## What it does

1. **Extract synset metadata** -- definitions, POS tags, lemma lists, and
   relationship links from NLTK's WordNet interface.
2. **Parse GlossTag** -- extract synset-to-synset edges from the
   sense-tagged gloss corpus.
3. **Compute word embeddings** -- OpenAI embeddings for individual words.
4. **Project to 2D** -- UMAP reduction of word embeddings.
5. **Build word-indexed view** -- select one representative synset per word
   and merge with frequency data and 2D coordinates.

## Output

Multiple DataFrames and link tables:

**Synset-indexed:**

| Column | Description |
|---|---|
| `definition` | Synset definition text |
| `pos` | Part of speech (n, v, a, r, s) |
| `lemmas` | List of words in the synset |
| relationship columns | Lists of related synset IDs |

**Word-indexed:**

| Column | Description |
|---|---|
| `synset` | Representative synset for this word |
| `definition` | Synset definition |
| `frequency` | Word frequency from frequency list |
| `umap_x`, `umap_y` | 2D coordinates from embedding projection |

**Link tables:** Source→target edges for hypernyms, hyponyms, meronyms, etc.

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
| `wordnet_words.ipynb` | Main exploration notebook |
| `test_synset_refactor.ipynb` | Tests for synset-based refactoring |
