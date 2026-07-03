"""Feature engineering for the Hillstrom Email Marketing RCT dataset."""

from __future__ import annotations

import duckdb
import pandas as pd
from pathlib import Path


RAW_URL = (
    "https://raw.githubusercontent.com/mshenfield/hillstrom-email-marketing/"
    "master/data/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

COLUMN_DTYPES = {
    "recency": "INTEGER",
    "history_segment": "VARCHAR",
    "history": "DOUBLE",
    "mens": "INTEGER",
    "womens": "INTEGER",
    "zip_code": "VARCHAR",
    "newbie": "INTEGER",
    "channel": "VARCHAR",
    "segment": "VARCHAR",
    "visit": "INTEGER",
    "conversion": "INTEGER",
    "spend": "DOUBLE",
}

# Map Hillstrom's three-level 'segment' to binary treatment
TREATMENT_MAP = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 1}


def load_raw(con: duckdb.DuckDBPyConnection, path: str | Path | None = None) -> str:
    """
    Read the Hillstrom CSV into DuckDB, register as 'hillstrom_raw'.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    path : local CSV path; if None, downloads from GitHub.

    Returns
    -------
    str – DuckDB table name 'hillstrom_raw'
    """
    src = str(path) if path else RAW_URL
    con.execute(f"""
        CREATE OR REPLACE TABLE hillstrom_raw AS
        SELECT * FROM read_csv_auto('{src}', header=True)
    """)
    return "hillstrom_raw"


def build_features(
    con: duckdb.DuckDBPyConnection,
    binary_treatment: bool = True,
    sql_export_path: Path | None = None,
) -> str:
    """
    Build the modelling feature table from 'hillstrom_raw'.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    binary_treatment : collapse treatment to 0/1.
    sql_export_path : if given, writes the SQL string to this .sql file.

    Returns
    -------
    str – DuckDB view name 'hillstrom_features'
    """
    sql = _feature_sql(binary_treatment=binary_treatment)
    if sql_export_path:
        Path(sql_export_path).write_text(sql, encoding="utf-8")
    con.execute(sql)
    return "hillstrom_features"


def _feature_sql(binary_treatment: bool = True) -> str:
    treatment_expr = (
        "CASE segment\n"
        "            WHEN 'No E-Mail'   THEN 0\n"
        "            WHEN 'Mens E-Mail' THEN 1\n"
        "            WHEN 'Womens E-Mail' THEN 1\n"
        "        END"
        if binary_treatment
        else
        "CASE segment\n"
        "            WHEN 'No E-Mail'   THEN 0\n"
        "            WHEN 'Mens E-Mail' THEN 1\n"
        "            WHEN 'Womens E-Mail' THEN 2\n"
        "        END"
    )

    return f"""
CREATE OR REPLACE VIEW hillstrom_features AS
SELECT
    ROW_NUMBER() OVER () AS customer_id,
    visit,
    conversion,
    spend,

    {treatment_expr} AS treatment,
    CASE segment
        WHEN 'No E-Mail'     THEN 0
        WHEN 'Mens E-Mail'   THEN 1
        WHEN 'Womens E-Mail' THEN 2
    END AS treatment_arm,

    recency,
    history,
    mens,
    womens,
    newbie,

    -- Captures non-linear recency effects seen in email response curves.
    CASE
        WHEN recency <=  3 THEN 0   -- active
        WHEN recency <=  6 THEN 1   -- warm
        WHEN recency <= 12 THEN 2   -- cooling
        ELSE                    3   -- lapsed
    END AS recency_bucket,

    -- Maps history_segment to an ordinal integer for monotonic-constraint compatibility.
    CASE history_segment
        WHEN '$0 - $100'     THEN 0
        WHEN '$100 - $200'   THEN 1
        WHEN '$200 - $350'   THEN 2
        WHEN '$350 - $500'   THEN 3
        WHEN '$500 - $750'   THEN 4
        WHEN '$750 - $1,000' THEN 5
        WHEN '$1,000 +'      THEN 6
        ELSE                     0
    END AS spend_tier,

    CASE WHEN zip_code = 'Urban'    THEN 1 ELSE 0 END AS zip_urban,
    CASE WHEN zip_code = 'Surburban' THEN 1 ELSE 0 END AS zip_suburban,
    CASE WHEN zip_code = 'Rural'    THEN 1 ELSE 0 END AS zip_rural,

    CASE WHEN channel = 'Phone'     THEN 1 ELSE 0 END AS ch_phone,
    CASE WHEN channel = 'Web'       THEN 1 ELSE 0 END AS ch_web,
    CASE WHEN channel = 'Multichannel' THEN 1 ELSE 0 END AS ch_multi,

    -- High-spend newbies are a known high-uplift segment (first-purchase discount amplification).
    newbie * spend_tier AS newbie_x_spend_tier,
    recency * spend_tier AS recency_x_spend_tier

FROM hillstrom_raw
"""


def export_features(
    con: duckdb.DuckDBPyConnection,
    out_path: str | Path,
    view_name: str = "hillstrom_features",
) -> None:
    """Write the feature view to a parquet file for downstream notebooks."""
    con.execute(f"""
        COPY (SELECT * FROM {view_name})
        TO '{out_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


def describe(con: duckdb.DuckDBPyConnection, view: str = "hillstrom_features") -> pd.DataFrame:
    """Return a pandas summary (only used for display, not feature engineering)."""
    return con.execute(f"SUMMARIZE {view}").df()
