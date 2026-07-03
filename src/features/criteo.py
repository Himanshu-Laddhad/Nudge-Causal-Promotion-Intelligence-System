"""Feature engineering for the Criteo Uplift v2 dataset via DuckDB."""

from __future__ import annotations

from pathlib import Path
import duckdb


RAW_PARQUET_URL = (
    "https://criteo-uplift-dataset.s3-us-west-2.amazonaws.com/"
    "criteo-uplift-v2.1.csv.gz"
)

FEATURE_COLS = [f"f{i}" for i in range(12)]


def load_raw(
    con: duckdb.DuckDBPyConnection,
    path: str | Path | None = None,
) -> str:
    """
    Register Criteo Uplift v2 in DuckDB as 'criteo_raw'.

    Accepts a local CSV.gz, parquet, or the remote URL.
    DuckDB handles gzip decompression natively for CSV.

    Returns
    -------
    str – table name 'criteo_raw'
    """
    src = str(path) if path else RAW_PARQUET_URL
    if str(src).endswith(".parquet") or str(src).endswith(".parquet.gz"):
        con.execute(f"""
            CREATE OR REPLACE TABLE criteo_raw AS
            SELECT * FROM read_parquet('{src}')
        """)
    else:
        con.execute(f"""
            CREATE OR REPLACE TABLE criteo_raw AS
            SELECT * FROM read_csv_auto('{src}', header=True, compression='gzip')
        """)
    return "criteo_raw"


def build_features(
    con: duckdb.DuckDBPyConnection,
    sample_n: int | None = None,
    sql_export_path: Path | None = None,
) -> str:
    """
    Build modelling features from 'criteo_raw'.

    Features are already zero-centred and unit-scaled by the dataset authors.
    NTILE decile buckets are added for non-parametric diagnostics.

    Returns
    -------
    str – view name 'criteo_features'
    """
    sample_clause = f"USING SAMPLE {sample_n}" if sample_n else ""
    sql = f"""
CREATE OR REPLACE VIEW criteo_features AS
SELECT
    treatment,
    visit,
    conversion,

    {', '.join(FEATURE_COLS)},

    -- Decile buckets for QQ-plot diagnostics (10 = top decile)
    {chr(10).join(
        f"    NTILE(10) OVER (ORDER BY f{i}) AS f{i}_decile,"
        for i in range(12)
    )}

    -- Composite engagement score: unweighted mean of all features
    (f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11) / 12.0 AS mean_feature

FROM criteo_raw
{sample_clause}
"""
    if sql_export_path:
        Path(sql_export_path).write_text(sql, encoding="utf-8")
    con.execute(sql)
    return "criteo_features"


def export_sample(
    con: duckdb.DuckDBPyConnection,
    out_path: str | Path,
    n: int = 1_000_000,
) -> None:
    """Export a stratified 1 M-row sample for rapid notebook iteration."""
    con.execute(f"""
        COPY (
            SELECT * FROM criteo_raw
            USING SAMPLE {n} ROWS (stratify BY treatment)
        )
        TO '{out_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
