# Epstein Files

Data preparation for visualizing the Epstein document network with
[Cosmograph](https://cosmograph.app/).

## Data source

The raw data comes from the
[Epstein Document Network Explorer](https://github.com/maxandrews/Epstein-doc-explorer)
project by Max Andrews.
A live demo of the original visualization is at
<https://epstein-doc-explorer-1.onrender.com/>.

The source documents are publicly released files from the House Oversight
Committee, available at:
- <https://drive.google.com/drive/folders/1ldncvdqIf6miiskDp_EDuGSDAaI_fJx8>
- <https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K/tree/main>

The original project used Claude AI to analyze 25,232 documents and extract
structured relationships, entities, and events, storing everything in a SQLite
database (`document_analysis.db`, ~266 MB). We convert the tables from that
database into parquet files for faster downstream use.

## Tables

### `documents` (25,232 rows)

Metadata and full text for each analyzed document.

| Column | Description |
|---|---|
| `id` | Auto-increment primary key |
| `doc_id` | Unique document identifier (e.g. `HOUSE_OVERSIGHT_010477`) |
| `file_path` | Original file path within the source corpus |
| `one_sentence_summary` | AI-generated one-line summary |
| `paragraph_summary` | AI-generated detailed summary |
| `date_range_earliest` | Earliest date mentioned in the document |
| `date_range_latest` | Latest date mentioned in the document |
| `category` | Document category (e.g. `book_excerpt`, `legal_filing`) |
| `content_tags` | JSON array of content tags extracted by the AI |
| `analysis_timestamp` | When the AI analysis was performed |
| `input_tokens` | Claude API input token count |
| `output_tokens` | Claude API output token count |
| `cache_read_tokens` | Claude API cache-read token count |
| `cost_usd` | Estimated API cost for analyzing this document |
| `error` | Error message if analysis failed (NaN otherwise) |
| `created_at` | Row creation timestamp |
| `full_text` | Complete document text |

### `rdf_triples` (107,030 rows)

The core relationship data. Each row is an extracted RDF-style triple
(subject-predicate-object) linking an actor to a target through an action.

| Column | Description |
|---|---|
| `id` | Auto-increment primary key |
| `doc_id` | Foreign key to `documents.doc_id` |
| `timestamp` | When the event occurred (ISO format, may be null) |
| `actor` | Subject of the relationship (person, org, etc.) |
| `action` | Verb/predicate (e.g. `contacted`, `transferred funds to`) |
| `target` | Object of the relationship |
| `location` | Where the event occurred (may be null) |
| `actor_likely_type` | Type hint when the actor is unknown/redacted |
| `triple_tags` | JSON array of semantic tags (e.g. `["coercion","grooming"]`) |
| `explicit_topic` | What the document directly says about this relationship |
| `implicit_topic` | What can be inferred from context |
| `sequence_order` | Order of this triple within its source document |
| `top_cluster_ids` | JSON array of the top 3 tag-cluster IDs for this triple |
| `created_at` | Row creation timestamp |

### `relationships_edges` (107,030 rows)

A convenience view of `rdf_triples` with columns renamed for graph use
(`actor` -> `source`). Same row count; drops `id` and `created_at`.

### `entity_aliases` (28,934 rows)

Maps variant names to canonical names. Built by LLM-based deduplication
(e.g. "Jeff Epstein" -> "Jeffrey Epstein", "Bannon" -> "Steve Bannon").

| Column | Description |
|---|---|
| `original_name` | The name variant as it appeared in a document |
| `canonical_name` | The resolved canonical name |
| `reasoning` | LLM's explanation for why these names match |
| `created_at` | Row creation timestamp |
| `created_by` | Source of the alias (typically `llm_dedupe`) |
| `hop_distance_from_principal` | BFS distance from Jeffrey Epstein |

### `canonical_entities` (26,690 rows)

Deduplicated entity list with pre-computed graph distance from Jeffrey Epstein.

| Column | Description |
|---|---|
| `canonical_name` | The canonical entity name |
| `hop_distance_from_principal` | Minimum BFS hops from Jeffrey Epstein in the relationship graph |
| `created_at` | Row creation timestamp |

### `tag_embeddings` (29,689 rows)

Embeddings for the unique semantic tags, used to cluster tags into 30 groups.

| Column | Description |
|---|---|
| `tag` | The tag string (e.g. `email_communication`) |
| `embedding` | 32-dimensional embedding vector (stored as string of floats) |
| `model` | Embedding model used (`Qwen3-Embedding-0.6B-ONNX-fp16-32d`) |
| `created_at` | Row creation timestamp |

### `sqlite_sequence` (2 rows)

SQLite internal bookkeeping for auto-increment counters. Not used in analysis.

## Preparing data for visualization

Two functions in `__init__.py` handle the pipeline:

### Step 1: `acquire_epstein_data`

Downloads (or reads a local copy of) the SQLite database and exports every
table to parquet. Also creates the `relationships_edges` convenience table.

```python
from imbed_data_prep.epstein_files import acquire_epstein_data

acquire_epstein_data(out_dir="path/to/save")
```

### Step 2: `prepare_cosmograph_data`

Transforms the raw parquet tables into Cosmograph-ready DataFrames:

1. **Resolve entity aliases** -- maps every `source` and `target` name through
   the `entity_aliases` table so that variant spellings collapse to a single
   canonical name.
2. **Deduplicate edges** -- collapses the 107,030 raw triples into ~65,000
   unique directed `(source, target)` pairs, adding a `weight` column with
   the count of collapsed edges.
3. **Build nodes** -- collects every unique entity name from both sides of the
   edges, computes `degree` (total edge count per node), and joins in
   `hop_distance` from `canonical_entities`.

```python
from imbed_data_prep.epstein_files import prepare_cosmograph_data

cosmo_data = prepare_cosmograph_data(data_dir="path/to/parquets")
points = cosmo_data['points']   # canonical_name, degree, hop_distance
links  = cosmo_data['links']    # source, target, weight, action, ...
```

The resulting DataFrames can be passed directly to Cosmograph:

```python
from cosmograph import cosmo

cosmo(
    points=points,
    links=links,
    point_id_by="canonical_name",
    point_size_by="degree",
    point_color_by="hop_distance",
    point_color_strategy="interpolate",
    point_label_by="canonical_name",
    link_source_by="source",
    link_target_by="target",
    link_color_by="action",
    link_width_by="weight",
    link_arrows=True,
    show_dynamic_labels=True,
)
```

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code (`acquire_epstein_data`, `prepare_cosmograph_data`) |
| `epstein_files.ipynb` | Notebook for exploring and demoing the data prep |
| `epstein_files_tables_info.json` | Cached table metadata (columns, shapes) |
| `epstein_files_tables_info.pickle` | Cached table metadata (richer, with stats) |
