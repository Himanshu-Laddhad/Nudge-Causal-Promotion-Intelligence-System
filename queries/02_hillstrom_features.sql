-- 02_hillstrom_features.sql
-- Feature engineering for the Hillstrom uplift modelling dataset.
--
-- Design decisions:
--   1. Binary treatment: collapses two email arms to T=1 (any email) vs T=0 (no email).
--   2. Recency buckets: captures non-linear diminishing returns (active < warm < cooling < lapsed).
--   3. Spend tier ordinal: preserves monotonic ordering without dummy expansion.
--   4. Channel & zip one-hot: avoids implicit ordinal assumptions.
--      Note: original data spells 'Suburban' as 'Surburban' (typo preserved for CASE match).
--   5. Interaction features: newbie_x_spend_tier captures first-purchase amplification.

CREATE OR REPLACE VIEW hillstrom_features AS
SELECT
    ROW_NUMBER() OVER ()                         AS customer_id,
    visit,
    conversion,
    spend,

    -- Binary: any email (1) vs no email (0)
    CASE segment
        WHEN 'No E-Mail'     THEN 0
        WHEN 'Mens E-Mail'   THEN 1
        WHEN 'Womens E-Mail' THEN 1
    END                                          AS treatment,

    -- Three-arm for multi-arm analysis
    CASE segment
        WHEN 'No E-Mail'     THEN 0
        WHEN 'Mens E-Mail'   THEN 1
        WHEN 'Womens E-Mail' THEN 2
    END                                          AS treatment_arm,

    recency,
    history,
    mens,
    womens,
    newbie,

    CASE
        WHEN recency <=  3 THEN 0
        WHEN recency <=  6 THEN 1
        WHEN recency <= 12 THEN 2
        ELSE                    3
    END                                          AS recency_bucket,

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

    CASE WHEN zip_code = 'Urban'      THEN 1 ELSE 0 END AS zip_urban,
    CASE WHEN zip_code = 'Surburban'  THEN 1 ELSE 0 END AS zip_suburban,
    CASE WHEN zip_code = 'Rural'      THEN 1 ELSE 0 END AS zip_rural,

    CASE WHEN channel = 'Phone'        THEN 1 ELSE 0 END AS ch_phone,
    CASE WHEN channel = 'Web'          THEN 1 ELSE 0 END AS ch_web,
    CASE WHEN channel = 'Multichannel' THEN 1 ELSE 0 END AS ch_multi,

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
