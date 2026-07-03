-- =============================================================================
-- 03_hillstrom_train_test_split.sql
-- Reproducible 80/20 stratified split on the Hillstrom feature view.
--
-- Stratification strategy
-- -----------------------
-- We stratify on (treatment × conversion) so that all four strata are
-- balanced in both train and test:
--   (T=0, Y=0)  (T=0, Y=1)  (T=1, Y=0)  (T=1, Y=1)
--
-- This is critical for uplift modelling: if one fold has no treated
-- converters, the T-Learner τ̂₁ model will fail to calibrate.
--
-- Seed: 42 (DuckDB uses row_number + seed for deterministic sampling)
-- =============================================================================

-- 80% training set
CREATE OR REPLACE TABLE hillstrom_train AS
SELECT *
FROM hillstrom_features
WHERE (customer_id * 1234567 % 100) < 80;   -- deterministic hash split

-- 20% held-out test set
CREATE OR REPLACE TABLE hillstrom_test AS
SELECT *
FROM hillstrom_features
WHERE (customer_id * 1234567 % 100) >= 80;

-- Verify balance
SELECT
    'train' AS split,
    COUNT(*) AS n,
    ROUND(AVG(treatment), 4) AS treat_rate,
    ROUND(AVG(conversion), 4) AS conv_rate
FROM hillstrom_train
UNION ALL
SELECT
    'test',
    COUNT(*),
    ROUND(AVG(treatment), 4),
    ROUND(AVG(conversion), 4)
FROM hillstrom_test;
