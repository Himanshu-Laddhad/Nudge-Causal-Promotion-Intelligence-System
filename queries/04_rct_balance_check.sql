-- =============================================================================
-- 04_rct_balance_check.sql
-- Verify that the Hillstrom RCT is well-balanced on pre-treatment covariates.
--
-- Why this matters
-- ----------------
-- Uplift models rely on the Stable Unit Treatment Value Assumption (SUTVA)
-- and unconfoundedness.  In a well-executed RCT, treatment assignment is
-- independent of all pre-treatment covariates — the balance check below
-- is the empirical verification of this.
--
-- A significant imbalance on any covariate would suggest selection bias
-- (e.g., if older customers were more likely assigned to the email arm).
--
-- Interpretation
-- --------------
-- Standardised Mean Difference (SMD) < 0.1 is typically considered
-- "negligible" imbalance.  SMD > 0.2 warrants investigation.
-- =============================================================================

WITH stats AS (
    SELECT
        treatment,
        COUNT(*)                        AS n,
        AVG(recency)                    AS mean_recency,
        STDDEV_POP(recency)             AS sd_recency,
        AVG(history)                    AS mean_history,
        STDDEV_POP(history)             AS sd_history,
        AVG(mens)                       AS mean_mens,
        AVG(womens)                     AS mean_womens,
        AVG(newbie)                     AS mean_newbie,
        AVG(spend_tier)                 AS mean_spend_tier,
        STDDEV_POP(spend_tier)          AS sd_spend_tier
    FROM hillstrom_features
    GROUP BY treatment
),
treat AS (SELECT * FROM stats WHERE treatment = 1),
ctrl  AS (SELECT * FROM stats WHERE treatment = 0)

SELECT
    'recency' AS covariate,
    ROUND(treat.mean_recency, 3)  AS treat_mean,
    ROUND(ctrl.mean_recency, 3)   AS ctrl_mean,
    ROUND(
        ABS(treat.mean_recency - ctrl.mean_recency)
        / SQRT((treat.sd_recency^2 + ctrl.sd_recency^2) / 2),
        4
    )                              AS smd
FROM treat, ctrl
UNION ALL
SELECT
    'history',
    ROUND(treat.mean_history, 2),
    ROUND(ctrl.mean_history, 2),
    ROUND(
        ABS(treat.mean_history - ctrl.mean_history)
        / SQRT((treat.sd_history^2 + ctrl.sd_history^2) / 2),
        4
    )
FROM treat, ctrl
UNION ALL
SELECT
    'spend_tier',
    ROUND(treat.mean_spend_tier, 4),
    ROUND(ctrl.mean_spend_tier, 4),
    ROUND(
        ABS(treat.mean_spend_tier - ctrl.mean_spend_tier)
        / SQRT((treat.sd_spend_tier^2 + ctrl.sd_spend_tier^2) / 2),
        4
    )
FROM treat, ctrl
UNION ALL
SELECT 'newbie',
    ROUND(treat.mean_newbie, 4),
    ROUND(ctrl.mean_newbie, 4),
    ROUND(ABS(treat.mean_newbie - ctrl.mean_newbie) / SQRT(0.25), 4)
FROM treat, ctrl
UNION ALL
SELECT 'mens',
    ROUND(treat.mean_mens, 4),
    ROUND(ctrl.mean_mens, 4),
    ROUND(ABS(treat.mean_mens - ctrl.mean_mens) / SQRT(0.25), 4)
FROM treat, ctrl
ORDER BY smd DESC;
