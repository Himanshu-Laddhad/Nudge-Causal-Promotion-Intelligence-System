-- 03_criteo_load.sql
-- Load Criteo Uplift v2 dataset (14M rows, binary treatment, binary outcomes).
-- Features: f0–f11 (anonymised, normalised). Treatment: ad exposure (0/1).
-- Source: https://criteo-uplift-dataset.s3-us-west-2.amazonaws.com/criteo-uplift-v2.1.csv.gz
-- Reference: Diemert et al. (2021), AdKDD & TargetAd Workshop, KDD 2021

-- 1. Load full dataset
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

-- 3. 10% sample for fast benchmarking (~1.4M rows)
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
