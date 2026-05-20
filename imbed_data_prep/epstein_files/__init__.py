"""
Epstein Files Data Preparation Module.

This module provides tools for acquiring, normalizing, and preparing graph data
from the Epstein document corpus for visualization and analysis.

Data Source
-----------
The data originates from the Jeffrey Epstein case legal documents released by
the House Oversight Committee. This module sources data from a SQLite database
in the GitHub project:

    https://github.com/maxandrews/Epstein-doc-explorer

The project's document_analysis.db contains AI-extracted entities, relationships,
and semantic tags from ~25,000 legal documents. The original document corpus was
sourced from:

    https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K/tree/main

At the time of writing, processed documents are also available at:

    https://drive.google.com/drive/folders/1ldncvdqIf6miiskDp_EDuGSDAaI_fJx8

Database Structure
------------------
The source SQLite database contains these main tables:

- **documents**: Document metadata, AI-generated summaries, categories, content
  tags, date ranges, and full text (~25K documents)
- **rdf_triples**: Extracted relationships as subject-action-object triples with
  context (location, timestamp), classifications (actor type, tags, topics),
  and cluster assignments (~107K triples)
- **entity_aliases**: Deduplication mappings from variant names to canonical
  forms (e.g., "Jeff Epstein" → "Jeffrey Epstein") with LLM-generated reasoning
- **canonical_entities**: Unique entity names with hop distance from Jeffrey
  Epstein (the "principal" entity)
- **tag_embeddings**: Vector embeddings for ~30K unique tags extracted from
  triples, using a 32-dimensional model

Key Functions
-------------
- acquire_epstein_data: Download and export SQLite data to parquet files
- prepare_entity_graph_data: Entity network (people/orgs as nodes)
- prepare_document_graph_data: Document network (docs as nodes)
- prepare_tag_graph_data: Tag/topic network (tags as nodes)

All prep functions support caching via the @cached_graph_prep decorator.

Example
-------
>>> from imbed_data_prep.epstein_files import acquire_epstein_data, prepare_entity_graph_data  # doctest: +SKIP
>>> data_dir = acquire_epstein_data()  # downloads and exports to parquet  # doctest: +SKIP
>>> graph = prepare_entity_graph_data(data_dir)  # doctest: +SKIP
>>> graph['points']  # entity nodes with degree and hop distance  # doctest: +SKIP
>>> graph['links']   # relationship edges between entities  # doctest: +SKIP
"""

from typing import Dict, Optional, Union, Literal
from pathlib import Path
from collections import Counter
from itertools import combinations
import functools

from tabled.coerce import coerce_json_columns


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def cached_graph_prep(cache_name: str):
    """
    Decorator to cache graph prep functions to a tables store.

    The decorated function gains `tables` and `force_recompute` parameters.
    If `tables` is provided and cache exists, returns cached data.
    Otherwise computes, caches (if tables provided), and returns.

    Parameters
    ----------
    cache_name : str
        Base name for cache files (e.g., 'entity_graph' creates
        'prepped/{cache_name}_points' and 'prepped/{cache_name}_links')

    Example
    -------
    >>> @cached_graph_prep('entity_graph')  # doctest: +SKIP
    ... def prepare_entity_graph_data(data_dir=None, ...):
    ...     ...
    ...     return {'points': points_df, 'links': links_df}
    ...
    >>> # First call computes and caches:
    >>> data = prepare_entity_graph_data(data_dir=path, tables=tables)  # doctest: +SKIP
    >>> # Second call loads from cache:
    >>> data = prepare_entity_graph_data(data_dir=path, tables=tables)  # doctest: +SKIP
    >>> # Force recompute:
    >>> data = prepare_entity_graph_data(data_dir=path, tables=tables, force_recompute=True)  # doctest: +SKIP
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, tables=None, force_recompute=False, **kwargs):
            points_key = f'prepped/{cache_name}_points'
            links_key = f'prepped/{cache_name}_links'

            # Check cache (if tables provided and not forcing recompute)
            if tables is not None and not force_recompute:
                try:
                    if points_key in tables and links_key in tables:
                        return {
                            'points': tables[points_key],
                            'links': tables[links_key],
                        }
                except Exception:
                    pass  # Cache miss or error, proceed to compute

            # Compute (pass through tables for input data loading if function uses it)
            result = func(*args, **kwargs)

            # Save to cache if tables provided
            if tables is not None:
                tables[points_key] = result['points']
                tables[links_key] = result['links']

            return result

        # Expose cache info on wrapper
        wrapper._cache_name = cache_name
        return wrapper

    return decorator

def _get_clog(verbose):
    """Return a logging function based on verbosity setting."""
    return (lambda *args: print(*args)) if verbose else (lambda *args: None)


# ---------------------------------------------------------------------------
# Data normalization configuration
# ---------------------------------------------------------------------------

# Known columns that contain JSON list strings in the Epstein data
EPSTEIN_JSON_LIST_COLUMNS = {
    'documents': ['content_tags'],
    'rdf_triples': ['triple_tags', 'top_cluster_ids'],
    'relationships_edges': ['triple_tags', 'top_cluster_ids'],
    'tag_embeddings': ['embedding'],
}


def normalize_epstein_tables(
    tables: dict,
    *,
    save_to_store=None,
    verbose: bool = True,
) -> dict:
    """
    Normalize Epstein data tables by coercing JSON string columns to lists.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Dictionary of table name -> DataFrame.
    save_to_store : MutableMapping or None
        If provided, save the normalized tables back to this store.
    verbose : bool
        If True, print which columns were transformed.

    Returns
    -------
    dict[str, pd.DataFrame]
        Normalized tables.
    """
    clog = _get_clog(verbose)
    clog("Normalizing tables (coercing JSON string columns)...")

    result = {}
    for name, df in tables.items():
        # Get columns to coerce for this table (if known)
        columns = EPSTEIN_JSON_LIST_COLUMNS.get(name)

        if columns:
            df = coerce_json_columns(df, columns=columns, verbose=verbose)

        result[name] = df

        if save_to_store is not None:
            save_to_store[name] = df

    clog("Done normalizing.")
    return result


def _load_tables(data_dir, table_names, verbose=True, *, normalize=True):
    """
    Load parquet tables from a directory.

    Parameters
    ----------
    data_dir : Path or str or None
        Directory containing parquet files. If None, uses default imbed saves.
    table_names : list of str
        Names of tables to load (without .parquet extension).
    verbose : bool
        Whether to print progress messages.
    normalize : bool
        If True, coerce known JSON string columns to actual lists.
        This is mainly for backward compatibility with older parquet files
        that weren't normalized during acquisition. Files created by
        acquire_epstein_data(normalize=True) are already normalized.

    Returns
    -------
    dict[str, pd.DataFrame]
        Loaded tables keyed by name.
    """
    import pandas as pd

    clog = _get_clog(verbose)

    if data_dir is None:
        from imbed.util import DFLT_SAVES_DIR, process_path
        data_dir = process_path(DFLT_SAVES_DIR, 'epstein_files')
    data_dir = Path(data_dir)

    clog(f"Reading parquet files from {data_dir}")

    tables = {}
    for name in table_names:
        df = pd.read_parquet(data_dir / f'{name}.parquet')

        # Normalize JSON string columns if requested
        if normalize:
            columns = EPSTEIN_JSON_LIST_COLUMNS.get(name)
            if columns:
                df = coerce_json_columns(df, columns=columns, verbose=False)

        tables[name] = df

    return tables


def _resolve_aliases(edges, alias_map, src_col='source', tgt_col='target'):
    """
    Apply alias resolution to source and target columns of an edges DataFrame.

    Returns a copy with resolved names.
    """
    edges = edges.copy()
    edges[src_col] = edges[src_col].map(lambda x: alias_map.get(x, x))
    edges[tgt_col] = edges[tgt_col].map(lambda x: alias_map.get(x, x))
    return edges


def _compute_degree_counts(edges, src_col='source', tgt_col='target'):
    """Compute degree counts for nodes from an edges DataFrame."""
    degree = Counter()
    for src, tgt in zip(edges[src_col], edges[tgt_col]):
        degree[src] += 1
        degree[tgt] += 1
    return degree


def _deduplicate_edges(edges, src_col, tgt_col, verbose=True):
    """
    Collapse multiple edges between same (source, target) into one with weight.
    """
    clog = _get_clog(verbose)
    clog("Deduplicating edges...")

    metadata_cols = [c for c in edges.columns if c not in (src_col, tgt_col)]
    deduped = (
        edges.groupby([src_col, tgt_col], sort=False)
        .agg(
            weight=(src_col, 'size'),
            **{col: (col, 'first') for col in metadata_cols},
        )
        .reset_index()
    )
    clog(f"  {len(deduped)} unique directed edges")
    return deduped


def acquire_epstein_data(
    out_dir=None,
    *,
    sqlite_db_file=None,
    export_relationships=True,
    normalize=True,
    return_dataframes=False,
    verbose=True,
) -> Union[Path, tuple[Dict[str, "pd.DataFrame"], Optional[Path]]]:
    """
    Download (if needed) the Epstein document SQLite DB, read it with DuckDB,
    and export all tables to Parquet files and/or return as DataFrames.

    This function handles the full data acquisition pipeline: downloading the
    SQLite database from GitHub (if not provided), exporting all tables to
    parquet format, optionally normalizing JSON string columns, and creating
    a convenient relationships_edges view.

    Parameters
    ----------
    out_dir : str or Path or None
        Output directory for parquet files. Defaults to:
        ~/Downloads/epstein_doc_explorer_parquet
        If None and return_dataframes=True, no files are saved.
    sqlite_db_file : str or Path or None
        Path to the SQLite database file. If None, the DB is downloaded
        to a temporary file from the official GitHub source.
    export_relationships : bool
        Whether to create the extra relationships_edges.parquet file.
        This is a convenient view of rdf_triples with renamed columns
        (actor→source) for graph processing.
    normalize : bool
        If True, coerce known JSON string columns (like content_tags,
        triple_tags, embedding) to actual Python lists before saving.
        This makes the parquet files ready to use without post-processing.
    return_dataframes : bool
        If True, return (dataframes_dict, out_dir_path).
        If False, return just out_dir_path (original behavior).
    verbose : bool
        Controls printing of progress messages.

    Returns
    -------
    Union[Path, tuple[Dict[str, pd.DataFrame], Optional[Path]]]
        If return_dataframes=False: Path to output directory
        If return_dataframes=True: (dataframes_dict, output_directory_or_none)

    Notes
    -----
    The source SQLite database is cached locally using the graze library,
    so subsequent calls won't re-download unless the cache is cleared.
    """
    from tabled.sqlite_tools import (
        export_sqlite_to_parquet,
        export_sqlite_to_dataframes_and_parquet,
        export_sqlite_query_to_parquet,
    )

    clog = _get_clog(verbose)

    DEFAULT_DB_URL = (
        "https://github.com/maxandrews/Epstein-doc-explorer/"
        "raw/refs/heads/main/document_analysis.db"
    )

    # Resolve output directory
    if out_dir is None and not return_dataframes:
        out_dir = Path.home() / "Downloads" / "epstein_doc_explorer_parquet"
    elif out_dir is not None:
        out_dir = Path(out_dir)

    # Handle SQLite DB acquisition
    if sqlite_db_file is None:
        # Note: Cached download using graze
        import graze  # pip install graze

        def print_and_return(url):
            clog(f"Downloading Epstein DB from {url}...")
            return url

        sqlite_db_file = graze.graze(
            DEFAULT_DB_URL,
            return_filepaths=True,
            key_ingress=print_and_return,
        )
    else:
        sqlite_db_file = str(sqlite_db_file)

    # Choose the appropriate export method based on requirements
    if return_dataframes:
        # Get both DataFrames and optionally save to Parquet
        dataframes, final_out_dir = export_sqlite_to_dataframes_and_parquet(
            sqlite_db_file, out_dir, verbose=verbose
        )

        # Normalize JSON string columns if requested
        if normalize:
            dataframes = normalize_epstein_tables(
                dataframes, save_to_store=None, verbose=verbose
            )
            # Re-save normalized tables to parquet if we have an output directory
            if final_out_dir is not None:
                for name, df in dataframes.items():
                    if name in EPSTEIN_JSON_LIST_COLUMNS:
                        parquet_path = final_out_dir / f"{name}.parquet"
                        clog(f"Re-saving normalized {name} -> {parquet_path}")
                        df.to_parquet(parquet_path, compression="zstd")

        # Add custom relationships edge list if requested
        if export_relationships:
            relationships_query = """
                SELECT 
                    actor AS source,
                    target AS target,
                    action AS action,
                    timestamp,
                    location,
                    doc_id,
                    sequence_order,
                    actor_likely_type,
                    triple_tags,
                    explicit_topic,
                    implicit_topic,
                    top_cluster_ids
                FROM {schema}.rdf_triples
            """

            # Import duckdb for this custom query
            import duckdb

            con = duckdb.connect()
            try:
                con.execute("LOAD sqlite_scanner;")
                con.execute(f"ATTACH '{sqlite_db_file}' AS src (TYPE sqlite);")

                clog("Creating relationships edge DataFrame...")
                relationships_df = con.execute(
                    relationships_query.format(schema="src")
                ).df()

                # Normalize relationships_edges if requested
                if normalize:
                    rel_cols = EPSTEIN_JSON_LIST_COLUMNS.get('relationships_edges')
                    if rel_cols:
                        relationships_df = coerce_json_columns(
                            relationships_df, columns=rel_cols, verbose=verbose
                        )

                dataframes["relationships_edges"] = relationships_df

                # Also save to parquet if we have an output directory
                if final_out_dir is not None:
                    relationships_path = final_out_dir / "relationships_edges.parquet"
                    clog(f"Saving relationships_edges -> {relationships_path}")
                    relationships_df.to_parquet(relationships_path, compression="zstd")

            finally:
                con.close()

        clog("Done.")
        return dataframes, final_out_dir

    else:
        # Original behavior: just save to Parquet
        final_out_dir = export_sqlite_to_parquet(
            sqlite_db_file, out_dir, verbose=verbose
        )

        # Normalize JSON string columns if requested
        if normalize:
            import pandas as pd

            clog("Normalizing JSON string columns...")
            for table_name, columns in EPSTEIN_JSON_LIST_COLUMNS.items():
                parquet_path = final_out_dir / f"{table_name}.parquet"
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                    df = coerce_json_columns(df, columns=columns, verbose=verbose)
                    clog(f"Re-saving normalized {table_name} -> {parquet_path}")
                    df.to_parquet(parquet_path, compression="zstd")

        # Add custom relationships edge list if requested
        if export_relationships:
            relationships_path = final_out_dir / "relationships_edges.parquet"
            relationships_query = """
                SELECT 
                    actor AS source,
                    target AS target,
                    action AS action,
                    timestamp,
                    location,
                    doc_id,
                    sequence_order,
                    actor_likely_type,
                    triple_tags,
                    explicit_topic,
                    implicit_topic,
                    top_cluster_ids
                FROM {schema}.rdf_triples
            """

            from tabled.sqlite_tools import export_sqlite_query_to_parquet

            export_sqlite_query_to_parquet(
                sqlite_db_file,
                relationships_path,
                query=relationships_query,
                verbose=verbose,
            )

            # Normalize relationships_edges if requested
            if normalize:
                import pandas as pd

                rel_cols = EPSTEIN_JSON_LIST_COLUMNS.get('relationships_edges')
                if rel_cols:
                    df = pd.read_parquet(relationships_path)
                    df = coerce_json_columns(df, columns=rel_cols, verbose=verbose)
                    clog(f"Re-saving normalized relationships_edges -> {relationships_path}")
                    df.to_parquet(relationships_path, compression="zstd")

        clog(f"Done. Output folder: {final_out_dir}")
        return final_out_dir


@cached_graph_prep('entity_graph')
def prepare_entity_graph_data(
    data_dir=None,
    *,
    source_tables=None,
    deduplicate_edges=True,
    include_extended_info=False,
    verbose=True,
):
    """
    Prepare entity-based graph data where nodes are people/entities and edges
    are relationships between them.

    This resolves entity aliases (so "Jeff Epstein" -> "Jeffrey Epstein"),
    builds a nodes DataFrame with degree counts and hop distance from
    Jeffrey Epstein, and returns cleaned edges.

    Parameters
    ----------
    data_dir : str or Path or None
        Directory containing the parquet files. If None, uses the default
        imbed saves location. Ignored if ``source_tables`` is provided.
    source_tables : dict[str, pd.DataFrame] or None
        Pre-loaded DataFrames keyed by table name. Expected keys:
        'relationships_edges' (or 'rdf_triples'), 'entity_aliases',
        'canonical_entities'. If None, tables are read from ``data_dir``.
    deduplicate_edges : bool
        If True, collapse multiple edges between the same (source, target)
        pair into one row and add a ``weight`` column with the count.
    include_extended_info : bool
        If True, include additional columns in points: in_degree, out_degree,
        doc_count (number of unique documents mentioning entity).
    verbose : bool
        Controls printing of progress messages.
    tables : MutableMapping or None
        (Added by @cached_graph_prep) Cache store for saving/loading results.
        If provided and cache exists, returns cached data instead of recomputing.
    force_recompute : bool
        (Added by @cached_graph_prep) If True, recompute even if cache exists.

    Returns
    -------
    dict with keys:
        'points' : pd.DataFrame
            Columns: canonical_name, degree, hop_distance
            (plus in_degree, out_degree, doc_count if include_extended_info)
        'links' : pd.DataFrame
            Columns: source, target, action, [weight], plus original metadata
    """
    import pandas as pd

    clog = _get_clog(verbose)

    # ---- Load tables --------------------------------------------------------
    if source_tables is None:
        loaded = _load_tables(
            data_dir,
            ['relationships_edges', 'entity_aliases', 'canonical_entities'],
            verbose=verbose,
        )
        edges = loaded['relationships_edges']
        aliases = loaded['entity_aliases']
        entities = loaded['canonical_entities']
    else:
        edges = source_tables.get('relationships_edges', source_tables.get('rdf_triples'))
        aliases = source_tables['entity_aliases']
        entities = source_tables['canonical_entities']

    # ---- Resolve aliases ----------------------------------------------------
    clog("Resolving entity aliases...")
    alias_map = dict(zip(aliases['original_name'], aliases['canonical_name']))

    src_col = 'source' if 'source' in edges.columns else 'actor'
    tgt_col = 'target'

    edges = _resolve_aliases(edges, alias_map, src_col, tgt_col)

    # ---- Deduplicate edges (optionally) ------------------------------------
    if deduplicate_edges:
        edges = _deduplicate_edges(edges, src_col, tgt_col, verbose=verbose)

    # ---- Build nodes -------------------------------------------------------
    clog("Building nodes...")

    hop_map = dict(
        zip(entities['canonical_name'], entities['hop_distance_from_principal'])
    )

    if include_extended_info:
        # Compute in/out degree and doc counts
        in_degree = Counter(edges[tgt_col])
        out_degree = Counter(edges[src_col])
        all_names = set(in_degree.keys()) | set(out_degree.keys())

        # Count unique documents per entity
        doc_counts = Counter()
        for _, row in edges.iterrows():
            doc_id = row.get('doc_id')
            if doc_id:
                doc_counts[(row[src_col], doc_id)] = 1
                doc_counts[(row[tgt_col], doc_id)] = 1
        entity_doc_count = Counter()
        for (entity, _), _ in doc_counts.items():
            entity_doc_count[entity] += 1

        points = pd.DataFrame([
            {
                'canonical_name': name,
                'degree': in_degree.get(name, 0) + out_degree.get(name, 0),
                'in_degree': in_degree.get(name, 0),
                'out_degree': out_degree.get(name, 0),
                'hop_distance': hop_map.get(name),
                'doc_count': entity_doc_count.get(name, 0),
            }
            for name in all_names
        ])
    else:
        degree = _compute_degree_counts(edges, src_col, tgt_col)
        points = pd.DataFrame([
            {
                'canonical_name': name,
                'degree': deg,
                'hop_distance': hop_map.get(name),
            }
            for name, deg in degree.items()
        ])

    points = points.sort_values('degree', ascending=False).reset_index(drop=True)

    n_with_hop = points['hop_distance'].notna().sum()
    clog(
        f"  {len(points)} nodes "
        f"({n_with_hop} with hop distance, "
        f"{len(points) - n_with_hop} without)"
    )

    # ---- Rename edge columns for consistency --------------------------------
    if src_col == 'actor':
        edges = edges.rename(columns={'actor': 'source'})

    clog("Done.")
    return {'points': points, 'links': edges}


# Backward compatibility alias
prepare_cosmograph_data = prepare_entity_graph_data


# ---------------------------------------------------------------------------
# Document-based graph functions
# ---------------------------------------------------------------------------

DocumentLinkStrategy = Literal['shared_entities', 'shared_tags', 'shared_topics']


@cached_graph_prep('document_graph')
def prepare_document_graph_data(
    data_dir=None,
    *,
    source_tables=None,
    link_by: DocumentLinkStrategy = 'shared_entities',
    min_shared: int = 1,
    verbose=True,
):
    """
    Prepare document-based graph data where nodes are documents and edges
    connect documents that share entities, tags, or topics.

    Parameters
    ----------
    data_dir : str or Path or None
        Directory containing the parquet files. If None, uses the default
        imbed saves location. Ignored if ``source_tables`` is provided.
    source_tables : dict[str, pd.DataFrame] or None
        Pre-loaded DataFrames keyed by table name. If None, tables are read
        from ``data_dir``.
    link_by : {'shared_entities', 'shared_tags', 'shared_topics'}
        Strategy for creating links between documents:
        - 'shared_entities': Link docs that mention the same entity
        - 'shared_tags': Link docs whose triples share tags (from triple_tags)
        - 'shared_topics': Link docs with same explicit_topic or implicit_topic
    min_shared : int
        Minimum number of shared items required to create a link.
    verbose : bool
        Controls printing of progress messages.
    tables : MutableMapping or None
        (Added by @cached_graph_prep) Cache store for saving/loading results.
        If provided and cache exists, returns cached data instead of recomputing.
    force_recompute : bool
        (Added by @cached_graph_prep) If True, recompute even if cache exists.

    Returns
    -------
    dict with keys:
        'points' : pd.DataFrame
            Columns: doc_id, entity_count, triple_count, plus document metadata
        'links' : pd.DataFrame
            Columns: source, target, weight, shared_items
    """
    import pandas as pd
    import json

    clog = _get_clog(verbose)

    # ---- Load tables --------------------------------------------------------
    if source_tables is None:
        loaded = _load_tables(
            data_dir,
            ['relationships_edges', 'documents', 'entity_aliases'],
            verbose=verbose,
        )
        edges = loaded['relationships_edges']
        documents = loaded['documents']
        aliases = loaded['entity_aliases']
    else:
        edges = source_tables.get('relationships_edges', source_tables.get('rdf_triples'))
        documents = source_tables['documents']
        aliases = source_tables.get('entity_aliases')

    # Resolve aliases if available
    if aliases is not None:
        clog("Resolving entity aliases...")
        alias_map = dict(zip(aliases['original_name'], aliases['canonical_name']))
        src_col = 'source' if 'source' in edges.columns else 'actor'
        edges = _resolve_aliases(edges, alias_map, src_col, 'target')
    else:
        src_col = 'source' if 'source' in edges.columns else 'actor'

    # ---- Build doc -> items mapping based on strategy -----------------------
    clog(f"Building document links by '{link_by}'...")

    doc_items = {}  # doc_id -> set of items

    if link_by == 'shared_entities':
        for _, row in edges.iterrows():
            doc_id = row['doc_id']
            if doc_id not in doc_items:
                doc_items[doc_id] = set()
            doc_items[doc_id].add(row[src_col])
            doc_items[doc_id].add(row['target'])

    elif link_by == 'shared_tags':
        for _, row in edges.iterrows():
            doc_id = row['doc_id']
            if doc_id not in doc_items:
                doc_items[doc_id] = set()
            tags_raw = row.get('triple_tags', '[]')
            if isinstance(tags_raw, str):
                try:
                    tags = json.loads(tags_raw)
                except (json.JSONDecodeError, TypeError):
                    tags = []
            else:
                tags = tags_raw if tags_raw else []
            doc_items[doc_id].update(tags)

    elif link_by == 'shared_topics':
        for _, row in edges.iterrows():
            doc_id = row['doc_id']
            if doc_id not in doc_items:
                doc_items[doc_id] = set()
            explicit = row.get('explicit_topic')
            implicit = row.get('implicit_topic')
            if explicit and pd.notna(explicit):
                doc_items[doc_id].add(f"explicit:{explicit}")
            if implicit and pd.notna(implicit):
                doc_items[doc_id].add(f"implicit:{implicit}")
    else:
        raise ValueError(f"Unknown link_by strategy: {link_by}")

    # ---- Build links between documents --------------------------------------
    clog("Computing document pairs...")

    # Invert: item -> set of doc_ids
    item_docs = {}
    for doc_id, items in doc_items.items():
        for item in items:
            if item not in item_docs:
                item_docs[item] = set()
            item_docs[item].add(doc_id)

    # Count shared items between doc pairs
    doc_pair_shared = Counter()
    for item, docs in item_docs.items():
        if len(docs) > 1:
            for d1, d2 in combinations(sorted(docs), 2):
                doc_pair_shared[(d1, d2)] += 1

    # Filter by min_shared and build links DataFrame
    links_data = []
    for (d1, d2), count in doc_pair_shared.items():
        if count >= min_shared:
            links_data.append({
                'source': d1,
                'target': d2,
                'weight': count,
            })

    links = pd.DataFrame(links_data)
    clog(f"  {len(links)} document pairs with >= {min_shared} shared {link_by}")

    # ---- Build points (documents) -------------------------------------------
    clog("Building document points...")

    # Compute per-doc stats
    doc_entity_count = Counter()
    doc_triple_count = Counter()
    for _, row in edges.iterrows():
        doc_id = row['doc_id']
        doc_triple_count[doc_id] += 1
        doc_entity_count[doc_id] += 2  # source and target

    # Merge with document metadata
    doc_ids_in_graph = set(doc_items.keys())
    points_data = []
    doc_info = documents.set_index('doc_id')

    for doc_id in doc_ids_in_graph:
        row = {'doc_id': doc_id}
        row['entity_count'] = len(doc_items.get(doc_id, set()))
        row['triple_count'] = doc_triple_count.get(doc_id, 0)

        # Add document metadata if available
        if doc_id in doc_info.index:
            doc_row = doc_info.loc[doc_id]
            row['one_sentence_summary'] = doc_row.get('one_sentence_summary', '')
            row['category'] = doc_row.get('category', '')
            row['date_range_earliest'] = doc_row.get('date_range_earliest')
            row['date_range_latest'] = doc_row.get('date_range_latest')

        points_data.append(row)

    points = pd.DataFrame(points_data)
    points = points.sort_values('triple_count', ascending=False).reset_index(drop=True)

    clog(f"  {len(points)} document nodes")
    clog("Done.")

    return {'points': points, 'links': links}


# ---------------------------------------------------------------------------
# Tag-based graph functions
# ---------------------------------------------------------------------------

TagLinkStrategy = Literal['co_occurrence', 'embedding_similarity']


@cached_graph_prep('tag_graph')
def prepare_tag_graph_data(
    data_dir=None,
    *,
    source_tables=None,
    link_by: TagLinkStrategy = 'co_occurrence',
    min_co_occurrence: int = 2,
    similarity_threshold: float = 0.7,
    verbose=True,
):
    """
    Prepare tag-based graph data where nodes are unique tags and edges
    connect tags that co-occur or are semantically similar.

    Parameters
    ----------
    data_dir : str or Path or None
        Directory containing the parquet files. If None, uses the default
        imbed saves location. Ignored if ``source_tables`` is provided.
    source_tables : dict[str, pd.DataFrame] or None
        Pre-loaded DataFrames keyed by table name. If None, tables are read
        from ``data_dir``.
    link_by : {'co_occurrence', 'embedding_similarity'}
        Strategy for creating links between tags:
        - 'co_occurrence': Link tags that appear together in the same triple
        - 'embedding_similarity': Link tags with similar embeddings
    min_co_occurrence : int
        For 'co_occurrence': minimum times tags must co-occur to create link.
    similarity_threshold : float
        For 'embedding_similarity': minimum cosine similarity to create link.
    verbose : bool
        Controls printing of progress messages.
    tables : MutableMapping or None
        (Added by @cached_graph_prep) Cache store for saving/loading results.
        If provided and cache exists, returns cached data instead of recomputing.
    force_recompute : bool
        (Added by @cached_graph_prep) If True, recompute even if cache exists.

    Returns
    -------
    dict with keys:
        'points' : pd.DataFrame
            Columns: tag, occurrence_count, doc_count
        'links' : pd.DataFrame
            Columns: source, target, weight (co-occurrence count or similarity)
    """
    import pandas as pd
    import json

    clog = _get_clog(verbose)

    # ---- Load tables --------------------------------------------------------
    table_names = ['relationships_edges']
    if link_by == 'embedding_similarity':
        table_names.append('tag_embeddings')

    if source_tables is None:
        loaded = _load_tables(data_dir, table_names, verbose=verbose)
        edges = loaded['relationships_edges']
        tag_embeddings = loaded.get('tag_embeddings')
    else:
        edges = source_tables.get('relationships_edges', source_tables.get('rdf_triples'))
        tag_embeddings = source_tables.get('tag_embeddings')

    # ---- Parse tags from triples --------------------------------------------
    clog("Parsing tags from triples...")

    tag_occurrences = Counter()  # tag -> total count
    tag_doc_count = Counter()  # tag -> unique doc count
    tag_docs = {}  # tag -> set of doc_ids
    triple_tags_list = []  # list of (doc_id, set of tags) for each triple

    for _, row in edges.iterrows():
        doc_id = row['doc_id']
        tags_raw = row.get('triple_tags', '[]')
        if isinstance(tags_raw, str):
            try:
                tags = json.loads(tags_raw)
            except (json.JSONDecodeError, TypeError):
                tags = []
        else:
            tags = tags_raw if tags_raw else []

        tags_set = set(tags)
        triple_tags_list.append((doc_id, tags_set))

        for tag in tags_set:
            tag_occurrences[tag] += 1
            if tag not in tag_docs:
                tag_docs[tag] = set()
            tag_docs[tag].add(doc_id)

    for tag, docs in tag_docs.items():
        tag_doc_count[tag] = len(docs)

    clog(f"  {len(tag_occurrences)} unique tags found")

    # ---- Build links based on strategy --------------------------------------
    if link_by == 'co_occurrence':
        clog("Computing tag co-occurrences...")

        co_occurrence = Counter()
        for doc_id, tags_set in triple_tags_list:
            if len(tags_set) > 1:
                for t1, t2 in combinations(sorted(tags_set), 2):
                    co_occurrence[(t1, t2)] += 1

        links_data = []
        for (t1, t2), count in co_occurrence.items():
            if count >= min_co_occurrence:
                links_data.append({
                    'source': t1,
                    'target': t2,
                    'weight': count,
                })

        links = pd.DataFrame(links_data)
        clog(f"  {len(links)} tag pairs with >= {min_co_occurrence} co-occurrences")

    elif link_by == 'embedding_similarity':
        clog("Computing tag embedding similarities...")

        if tag_embeddings is None:
            raise ValueError(
                "tag_embeddings table required for 'embedding_similarity' strategy"
            )

        import numpy as np

        # Build tag -> embedding mapping
        tag_to_emb = {}
        for _, row in tag_embeddings.iterrows():
            tag = row['tag']
            emb_raw = row['embedding']
            if isinstance(emb_raw, str):
                emb = np.array(json.loads(emb_raw))
            else:
                emb = np.array(emb_raw)
            tag_to_emb[tag] = emb / np.linalg.norm(emb)  # normalize

        # Only consider tags that appear in our data
        tags_with_emb = [t for t in tag_occurrences if t in tag_to_emb]
        clog(f"  {len(tags_with_emb)} tags have embeddings")

        links_data = []
        for i, t1 in enumerate(tags_with_emb):
            for t2 in tags_with_emb[i + 1:]:
                sim = np.dot(tag_to_emb[t1], tag_to_emb[t2])
                if sim >= similarity_threshold:
                    links_data.append({
                        'source': t1,
                        'target': t2,
                        'weight': float(sim),
                    })

        links = pd.DataFrame(links_data)
        clog(f"  {len(links)} tag pairs with similarity >= {similarity_threshold}")

    else:
        raise ValueError(f"Unknown link_by strategy: {link_by}")

    # ---- Build points (tags) ------------------------------------------------
    clog("Building tag points...")

    points_data = []
    for tag, count in tag_occurrences.items():
        points_data.append({
            'tag': tag,
            'occurrence_count': count,
            'doc_count': tag_doc_count.get(tag, 0),
        })

    points = pd.DataFrame(points_data)
    points = points.sort_values('occurrence_count', ascending=False).reset_index(
        drop=True
    )

    clog(f"  {len(points)} tag nodes")
    clog("Done.")

    return {'points': points, 'links': links}
