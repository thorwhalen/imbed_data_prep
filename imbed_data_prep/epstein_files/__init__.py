"""Preparing Epstein Files Data"""

from typing import Dict, Optional, Union
from pathlib import Path
from collections import Counter


def acquire_epstein_data(
    out_dir=None,
    *,
    sqlite_db_file=None,
    export_relationships=True,
    return_dataframes=False,
    verbose=True,
) -> Union[Path, tuple[Dict[str, "pd.DataFrame"], Optional[Path]]]:
    """
    Download (if needed) the Epstein document SQLite DB, read it with DuckDB,
    and export all tables to Parquet files and/or return as DataFrames.

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
    """
    from tabled.sqlite_tools import (
        export_sqlite_to_parquet,
        export_sqlite_to_dataframes,
        export_sqlite_to_dataframes_and_parquet,
        export_sqlite_query_to_parquet,
    )
    import tempfile
    import urllib.request
    import shutil

    # Verbose logging helper
    clog = (lambda *args: print(*args)) if verbose else (lambda *args: None)

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

        clog(f"Done. Output folder: {final_out_dir}")
        return final_out_dir


def prepare_cosmograph_data(
    data_dir=None,
    *,
    tables=None,
    deduplicate_edges=True,
    verbose=True,
):
    """
    Transform raw Epstein parquet tables into Cosmograph-ready points and links
    DataFrames.

    This resolves entity aliases (so "Jeff Epstein" -> "Jeffrey Epstein"),
    builds a nodes DataFrame with degree counts and hop distance from
    Jeffrey Epstein, and returns cleaned edges.

    Parameters
    ----------
    data_dir : str or Path or None
        Directory containing the parquet files. If None, uses the default
        imbed saves location. Ignored if ``tables`` is provided.
    tables : dict[str, pd.DataFrame] or None
        Pre-loaded DataFrames keyed by table name. Expected keys:
        'relationships_edges' (or 'rdf_triples'), 'entity_aliases',
        'canonical_entities'. If None, tables are read from ``data_dir``.
    deduplicate_edges : bool
        If True, collapse multiple edges between the same (source, target)
        pair into one row and add a ``weight`` column with the count.
    verbose : bool
        Controls printing of progress messages.

    Returns
    -------
    dict with keys:
        'points' : pd.DataFrame
            Columns: canonical_name, degree, hop_distance
        'links' : pd.DataFrame
            Columns: source, target, action, [weight], plus original metadata
    """
    import pandas as pd

    clog = (lambda *args: print(*args)) if verbose else (lambda *args: None)

    # ---- Load tables --------------------------------------------------------
    if tables is None:
        if data_dir is None:
            from imbed.util import DFLT_SAVES_DIR, process_path

            data_dir = process_path(DFLT_SAVES_DIR, 'epstein_files')
        data_dir = Path(data_dir)
        clog(f"Reading parquet files from {data_dir}")
        edges = pd.read_parquet(data_dir / 'relationships_edges.parquet')
        aliases = pd.read_parquet(data_dir / 'entity_aliases.parquet')
        entities = pd.read_parquet(data_dir / 'canonical_entities.parquet')
    else:
        edges = tables.get('relationships_edges', tables.get('rdf_triples'))
        aliases = tables['entity_aliases']
        entities = tables['canonical_entities']

    # ---- Resolve aliases ----------------------------------------------------
    clog("Resolving entity aliases...")
    alias_map = dict(zip(aliases['original_name'], aliases['canonical_name']))

    # Figure out which column names to use for source/target
    src_col = 'source' if 'source' in edges.columns else 'actor'
    tgt_col = 'target'

    edges = edges.copy()
    edges[src_col] = edges[src_col].map(lambda x: alias_map.get(x, x))
    edges[tgt_col] = edges[tgt_col].map(lambda x: alias_map.get(x, x))

    # ---- Deduplicate edges (optionally) ------------------------------------
    if deduplicate_edges:
        clog("Deduplicating edges...")
        # Keep first occurrence's metadata, add weight
        metadata_cols = [
            c for c in edges.columns if c not in (src_col, tgt_col)
        ]
        deduped = (
            edges.groupby([src_col, tgt_col], sort=False)
            .agg(
                weight=(src_col, 'size'),  # count of edges per pair
                **{col: (col, 'first') for col in metadata_cols},
            )
            .reset_index()
        )
        edges = deduped
        clog(f"  {len(edges)} unique directed edges")

    # ---- Build nodes -------------------------------------------------------
    clog("Building nodes...")
    degree = Counter()
    for src, tgt in zip(edges[src_col], edges[tgt_col]):
        degree[src] += 1
        degree[tgt] += 1

    hop_map = dict(
        zip(entities['canonical_name'], entities['hop_distance_from_principal'])
    )

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

    # ---- Rename edge columns for Cosmograph --------------------------------
    if src_col == 'actor':
        edges = edges.rename(columns={'actor': 'source'})

    clog("Done.")
    return {'points': points, 'links': edges}
