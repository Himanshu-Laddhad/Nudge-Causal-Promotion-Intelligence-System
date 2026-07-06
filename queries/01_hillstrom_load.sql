-- 01_hillstrom_load.sql
-- Load the Hillstrom Email Marketing dataset (3-arm RCT, n=64,000) from CSV into DuckDB.

CREATE OR REPLACE TABLE hillstrom_raw AS
SELECT *
FROM read_csv_auto(
    'data/raw/hillstrom.csv',
    header = TRUE
);

-- RCT balance check: equal arm sizes confirm successful randomisation.
SELECT
    segment,
    COUNT(*)                                                    AS n,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)         AS pct,
    ROUND(AVG(conversion), 4)                                   AS conv_rate,
    ROUND(AVG(visit),      4)                                   AS visit_rate,
    ROUND(AVG(spend),      2)                                   AS avg_spend
FROM hillstrom_raw
GROUP BY segment
ORDER BY segment;
