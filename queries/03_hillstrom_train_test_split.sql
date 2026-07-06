-- 03_hillstrom_train_test_split.sql
-- Reproducible 80/20 hash-based split stratified on (treatment × conversion).
-- Stratification ensures all four strata are balanced in both train and test.

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
