-- =============================================================================
-- 01_hillstrom_load.sql
-- Load the Hillstrom Email Marketing dataset from a local CSV into DuckDB.
--
-- Causal context
-- --------------
-- This is a genuine 3-arm RCT (n=64,000):
--   segment = 'No E-Mail'     → control   (T=0)  ~1/3 of sample
--   segment = 'Mens E-Mail'   → mens email (T=1)  ~1/3 of sample
--   segment = 'Womens E-Mail' → womens email (T=2) ~1/3 of sample
--
-- Outcome window: 2-week follow-up after email send date.
-- Primary outcome: conversion (binary purchase indicator).
-- Secondary outcomes: visit (website visit), spend (total $ spent).
--
-- RCT balance verification below confirms roughly equal arm sizes and
-- similar baseline conversion rates — a prerequisite for valid uplift
-- estimation.
-- =============================================================================

CREATE OR REPLACE TABLE hillstrom_raw AS
SELECT *
FROM read_csv_auto(
    'data/raw/hillstrom.csv',
    header = TRUE
);

-- RCT balance check: equal arm sizes and similar conversion rates
-- confirm successful randomisation.  Large imbalances would indicate
-- a data loading error or non-random assignment.
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
