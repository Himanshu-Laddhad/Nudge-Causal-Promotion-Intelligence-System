-- =============================================================================
-- 03_criteo_load.sql
-- Load Criteo Uplift v2 dataset.
--
-- Dataset : 14M rows, binary treatment (ad exposure), binary outcomes:
--             visit      (any site visit after exposure)
--             conversion (purchase)
-- Features: f0–f11 (anonymised, real-valued, already normalised by Criteo)
-- Treatment: 1 = user was shown the ad, 0 = user was not shown the ad
-- Columns : f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion,visit,exposure
--
-- Source   : https://criteo-uplift-dataset.s3-us-west-2.amazonaws.com/criteo-uplift-v2.1.csv.gz
-- Reference: Diemert et al. (2021), "A Large Scale Benchmark for Uplift Modeling",
--            AdKDD & TargetAd Workshop, KDD 2021
--
-- Usage    : Run via DuckDB CLI or from the Phase 3 notebook.
--            Supports both uncompressed (.csv) and compressed (.csv.gz) files.
-- =============================================================================

-- 1. Load full dataset
--    Uncompressed (~3 GB):  adjust path to data/raw/criteo-uplift-v2.1.csv
--    Compressed   (~0.5 GB): data/raw/criteo-uplift-v2.1.csv.gz  (add compression='gzip')
CREATE OR REPLACE TABLE criteo_raw AS
SELECT *
FROM read_csv_auto(
    'data/raw/criteo-uplift-v2.1.csv',
    header = TRUE
);

-- 2. Sanity check: treatment balance and outcome rates
SELECT
    treatment,
    COUNT(*)                                              AS n,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)  AS pct,
    ROUND(AVG(visit),      4)                            AS visit_rate,
    ROUND(AVG(conversion), 4)                            AS conv_rate
FROM criteo_raw
GROUP BY treatment
ORDER BY treatment;

-- Expected output (approximate):
--   treatment |    n    | pct  | visit_rate | conv_rate
--   ---------+---------+------+------------+----------
--       0     | ~9.7M   | 69.1 |   0.0384   |  0.0017
--       1     | ~4.3M   | 30.9 |   0.0474   |  0.0029

-- 3. 10% sample for fast benchmarking (~1.4M rows)
--    Note: DuckDB 1.5 requires USING SAMPLE inside a subquery before GROUP BY / aggregation.
CREATE OR REPLACE TABLE criteo_sample AS
SELECT * FROM (
    SELECT * FROM criteo_raw
    USING SAMPLE 10 PERCENT
);

SELECT COUNT(*) AS n_sample FROM criteo_sample;

-- 4. Feature summary on sample
SELECT
    ROUND(MIN(f0),  3) AS f0_min,  ROUND(MAX(f0),  3) AS f0_max,  ROUND(AVG(f0),  3) AS f0_mean,
    ROUND(MIN(f1),  3) AS f1_min,  ROUND(MAX(f1),  3) AS f1_max,  ROUND(AVG(f1),  3) AS f1_mean,
    ROUND(MIN(f11), 3) AS f11_min, ROUND(MAX(f11), 3) AS f11_max, ROUND(AVG(f11), 3) AS f11_mean
FROM criteo_sample;

-- 5. Export sample to parquet for fast reloading
COPY criteo_sample TO 'outputs/criteo_sample.parquet' (FORMAT PARQUET);
