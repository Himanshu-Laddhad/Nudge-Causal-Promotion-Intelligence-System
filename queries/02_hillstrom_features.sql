-- =============================================================================
-- 02_hillstrom_features.sql
-- Feature engineering for the Hillstrom uplift modelling dataset.
--
-- Design decisions
-- ----------------
-- 1. Binary treatment: collapses the two email arms into T=1 (any email)
--    vs T=0 (no email).  This is the standard setup for single-binary CATE.
--    For multi-arm analysis, use treatment_arm (0/1/2) instead.
--
-- 2. Recency buckets: recency is months since last purchase.  We bucket
--    to capture non-linear diminishing returns (active < warm < cooling < lapsed).
--    Explicit buckets help linear models; tree models benefit from cleaner splits.
--
-- 3. Spend tier ordinal: history_segment is already a discrete quintile.
--    Mapping to integer 0-6 preserves monotonic ordering without dummy expansion
--    and is compatible with monotonic constraints in gradient boosting.
--
-- 4. Channel & zip one-hot: avoids implicit ordinal assumptions.
--    Note: Hillstrom's dataset has a typo — 'Surburban' (not 'Suburban').
--    We preserve the original spelling so the CASE matches correctly.
--
-- 5. Interaction features:
--    newbie_x_spend_tier — first-purchase amplification effect: new customers
--    at a high spend tier respond more strongly to promotions.
--    recency_x_spend_tier — lapsed high-value customers are a distinct segment.
-- =============================================================================

CREATE OR REPLACE VIEW hillstrom_features AS
SELECT
    -- ── Identifiers & outcomes ──────────────────────────────────────────────
    ROW_NUMBER() OVER ()                         AS customer_id,
    visit,
    conversion,
    spend,

    -- ── Treatment encoding ──────────────────────────────────────────────────
    -- Binary: any email (1) vs no email (0)
    CASE segment
        WHEN 'No E-Mail'     THEN 0
        WHEN 'Mens E-Mail'   THEN 1
        WHEN 'Womens E-Mail' THEN 1
    END                                          AS treatment,

    -- Three-arm for multi-arm Phase 3 analysis
    CASE segment
        WHEN 'No E-Mail'     THEN 0
        WHEN 'Mens E-Mail'   THEN 1
        WHEN 'Womens E-Mail' THEN 2
    END                                          AS treatment_arm,

    -- ── Raw numerics ────────────────────────────────────────────────────────
    recency,
    history,
    mens,
    womens,
    newbie,

    -- ── Recency bucket (0=active … 3=lapsed) ────────────────────────────────
    CASE
        WHEN recency <=  3 THEN 0
        WHEN recency <=  6 THEN 1
        WHEN recency <= 12 THEN 2
        ELSE                    3
    END                                          AS recency_bucket,

    -- ── Spend tier ordinal (0=lowest … 6=highest) ───────────────────────────
    CASE history_segment
        WHEN '$0 - $100'     THEN 0
        WHEN '$100 - $200'   THEN 1
        WHEN '$200 - $350'   THEN 2
        WHEN '$350 - $500'   THEN 3
        WHEN '$500 - $750'   THEN 4
        WHEN '$750 - $1,000' THEN 5
        WHEN '$1,000 +'      THEN 6
        ELSE                     0
    END                                          AS spend_tier,

    -- ── Zip code (one-hot integers) ─────────────────────────────────────────
    -- Note: original data spells 'Suburban' as 'Surburban' (typo preserved)
    CASE WHEN zip_code = 'Urban'      THEN 1 ELSE 0 END AS zip_urban,
    CASE WHEN zip_code = 'Surburban'  THEN 1 ELSE 0 END AS zip_suburban,
    CASE WHEN zip_code = 'Rural'      THEN 1 ELSE 0 END AS zip_rural,

    -- ── Channel (one-hot integers) ───────────────────────────────────────────
    CASE WHEN channel = 'Phone'        THEN 1 ELSE 0 END AS ch_phone,
    CASE WHEN channel = 'Web'          THEN 1 ELSE 0 END AS ch_web,
    CASE WHEN channel = 'Multichannel' THEN 1 ELSE 0 END AS ch_multi,

    -- ── Interaction features ─────────────────────────────────────────────────
    newbie * (CASE history_segment
                  WHEN '$0 - $100'     THEN 0
                  WHEN '$100 - $200'   THEN 1
                  WHEN '$200 - $350'   THEN 2
                  WHEN '$350 - $500'   THEN 3
                  WHEN '$500 - $750'   THEN 4
                  WHEN '$750 - $1,000' THEN 5
                  WHEN '$1,000 +'      THEN 6
                  ELSE 0
              END)                               AS newbie_x_spend_tier,

    recency * (CASE history_segment
                   WHEN '$0 - $100'     THEN 0
                   WHEN '$100 - $200'   THEN 1
                   WHEN '$200 - $350'   THEN 2
                   WHEN '$350 - $500'   THEN 3
                   WHEN '$500 - $750'   THEN 4
                   WHEN '$750 - $1,000' THEN 5
                   WHEN '$1,000 +'      THEN 6
                   ELSE 0
               END)                              AS recency_x_spend_tier

FROM hillstrom_raw;
