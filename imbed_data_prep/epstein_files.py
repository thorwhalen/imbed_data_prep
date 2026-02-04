"""Preparing Epstein Files Data"""


def acquire_epstein_data(
    out_dir=None,
    *,
    sqlite_db_file=None,
    export_relationships=True,
    verbose=True,
):
    """
    Download (if needed) the Epstein document SQLite DB, read it with DuckDB,
    and export all tables to Parquet files.

    Parameters
    ----------
    out_dir : str or Path or None
        Output directory for parquet files. Defaults to:
        ~/Downloads/epstein_doc_explorer_parquet
    sqlite_db_file : str or Path or None
        Path to the SQLite database file. If None, the DB is downloaded
        to a temporary file from the official GitHub source.
    export_relationships : bool
        Whether to create the extra relationships_edges.parquet file.
    verbose : bool
        Controls printing of progress messages.

    Returns
    -------
    Path
        The directory containing the exported parquet files.
    """
    # Imports kept inside for full self-containment
    from pathlib import Path
    import duckdb
    import tempfile
    import urllib.request
    import shutil

    DEFAULT_DB_URL = (
        "https://github.com/maxandrews/Epstein-doc-explorer/"
        "raw/refs/heads/main/document_analysis.db"
    )

    # Resolve output directory
    if out_dir is None:
        out_dir = Path.home() / "Downloads" / "epstein_doc_explorer_parquet"
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Handle SQLite DB acquisition
    if sqlite_db_file is None:
        if verbose:
            print("Downloading SQLite DB from GitHub to a temporary file...")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        with urllib.request.urlopen(DEFAULT_DB_URL) as response, open(
            tmp.name, "wb"
        ) as f:
            shutil.copyfileobj(response, f)
        sqlite_db_file = tmp.name
        if verbose:
            print(f"SQLite DB downloaded to: {sqlite_db_file}")
    else:
        sqlite_db_file = str(sqlite_db_file)

    # DuckDB connection
    if verbose:
        print("Connecting DuckDB and loading sqlite_scanner...")
    con = duckdb.connect()
    con.execute("INSTALL sqlite_scanner;")
    con.execute("LOAD sqlite_scanner;")

    # Attach SQLite DB
    if verbose:
        print("Attaching SQLite database...")
    con.execute(f"ATTACH '{sqlite_db_file}' AS src (TYPE sqlite);")

    # Discover tables
    tables = con.execute("SHOW TABLES FROM src;").fetchall()

    if verbose:
        print("Tables discovered:", [t[0] for t in tables])

    # Export tables to parquet
    for (table_name,) in tables:
        out_path = out_dir / f"{table_name}.parquet"
        if verbose:
            print(f"Exporting {table_name} -> {out_path}")
        con.execute(
            f"""
            COPY (SELECT * FROM src.{table_name})
            TO '{out_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )

    # Optional relationships parquet
    if export_relationships:
        relationships_path = out_dir / "relationships_edges.parquet"
        if verbose:
            print(f"Exporting rdf_triples edge list -> {relationships_path}")
        con.execute(
            f"""
            COPY (
                SELECT
                  actor   AS source,
                  target  AS target,
                  action  AS action,
                  timestamp,
                  location,
                  doc_id,
                  sequence_order,
                  actor_likely_type,
                  triple_tags,
                  explicit_topic,
                  implicit_topic,
                  top_cluster_ids
                FROM src.rdf_triples
            )
            TO '{relationships_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )

    con.close()

    if verbose:
        print("Done. Output folder:", out_dir)

    return out_dir
