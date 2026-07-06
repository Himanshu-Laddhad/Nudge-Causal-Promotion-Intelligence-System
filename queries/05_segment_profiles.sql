-- 05_segment_profiles.sql
-- Profile predicted CATE segments (persuadable / neutral / sleeping dog).
-- Run after a model has written CATE scores to outputs/phase1_naive_scores.parquet.
--
-- Thresholds: persuadable > +0.01, sleeping_dog < -0.01, neutral otherwise.

CREATE OR REPLACE VIEW cate_segments AS
SELECT
    s.customer_id,
    s.cate_score,
    CASE
        WHEN s.cate_score >  0.01 THEN 'persuadable'
        WHEN s.cate_score < -0.01 THEN 'sleeping_dog'
        ELSE                           'neutral'
    END                              AS segment,

    f.recency,
    f.history,
    f.newbie,
    f.spend_tier,
    f.recency_bucket,
    f.zip_urban,
    f.zip_suburban,
    f.zip_rural,
    f.ch_phone,
    f.ch_web,
    f.ch_multi,
    f.mens,
    f.womens,
    f.conversion,
    f.spend,
    f.treatment
FROM read_parquet('outputs/phase1_naive_scores.parquet') s
JOIN hillstrom_features f USING (customer_id);

-- Segment summary table (used in dashboard waterfall chart)
SELECT
    segment,
    COUNT(*)                           AS n,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct,
    ROUND(AVG(cate_score), 5)          AS avg_cate,
    ROUND(AVG(conversion), 4)          AS obs_conv_rate,
    ROUND(AVG(spend), 2)               AS avg_spend,
    ROUND(AVG(recency), 1)             AS avg_recency,
    ROUND(AVG(spend_tier), 2)          AS avg_spend_tier,
    ROUND(AVG(newbie), 3)              AS pct_newbie
FROM cate_segments
GROUP BY segment
ORDER BY avg_cate DESC;
