-- Strength-weighted (user, app) training matrix with negative sampling.
-- Snowflake dialect.
--
-- Collapses the event log to one (user, app) row with a graded relevance label,
-- matching collapse_pairs + _sample_unobserved in Python.
--
-- event_name grades: af_view=1 < af_click=2 < af_install=3 < af_open=4
-- < af_complete_registration=5 < af_level_achieved=6 < af_purchase=7

CREATE OR REPLACE TABLE user_app_training AS
WITH observed AS (
    SELECT
        user_id,
        app_id,
        MAX(CASE event_name
            WHEN 'af_purchase'              THEN 7
            WHEN 'af_level_achieved'        THEN 6
            WHEN 'af_complete_registration' THEN 5
            WHEN 'af_open'                  THEN 4
            WHEN 'af_install'               THEN 3
            WHEN 'af_click'                 THEN 2
            WHEN 'af_view'                  THEN 1
            ELSE 0 END)                     AS label,
        -- pointwise-positive flag: anything at or past the install barrier
        MAX(CASE WHEN event_name IN (
                     'af_install','af_open','af_complete_registration',
                     'af_level_achieved','af_purchase')
                 THEN 1 ELSE 0 END)         AS is_positive,
        -- funnel weight for the pointwise loss (strongest event in the pair)
        MAX(CASE event_name
            WHEN 'af_purchase'              THEN 6
            WHEN 'af_level_achieved'        THEN 5
            WHEN 'af_complete_registration' THEN 4
            WHEN 'af_open'                  THEN 3
            WHEN 'af_install'               THEN 2
            ELSE 1 END)                     AS loss_weight
    FROM interactions
    WHERE event_time < :horizon
    GROUP BY user_id, app_id
),
-- Unobserved negatives: same-platform, non-blocked apps the user never touched.
-- Excluding seen apps keeps a held-out future install out of the negative pool.
negatives AS (
    SELECT
        u.user_id,
        s.app_id,
        0 AS label,
        0 AS is_positive,
        1 AS loss_weight
    FROM (SELECT DISTINCT user_id, platform FROM users) u
    CROSS JOIN (SELECT app_id, platform FROM apps WHERE status <> 'blocked') s
    LEFT JOIN observed o ON o.user_id = u.user_id AND o.app_id = s.app_id
    WHERE o.app_id IS NULL
      AND u.platform = s.platform
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY u.user_id
        ORDER BY HASH(u.user_id || s.app_id)  -- deterministic pseudo-random
    ) <= 5                                -- n_neg_per_user
)
SELECT user_id, app_id, label, is_positive, loss_weight, 'observed' AS source
FROM observed
UNION ALL
SELECT user_id, app_id, label, is_positive, loss_weight, 'negative' AS source
FROM negatives;