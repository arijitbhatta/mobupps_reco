-- Per-user interaction-aggregate feature table (Snowflake dialect).
--
-- Mirrors FeatureBuilder.fit in Python. The horizon is the latest event time in
-- the training slice; in production this would be an explicit snapshot timestamp
-- rather than MAX(event_time).

CREATE OR REPLACE TABLE user_features AS
WITH train AS (
    SELECT *
    FROM interactions
    WHERE event_time < :horizon  -- point-in-time cutoff, no future leakage
),
user_agg AS (
    SELECT
        user_id,
        MAX(event_time)                         AS last_event_time,
        COUNT(*)                                AS user_n_events,
        COUNT(DISTINCT app_id)                  AS user_n_distinct_apps,
        SUM(CASE WHEN event_name IN (
                      'af_install','af_open','af_complete_registration',
                      'af_level_achieved','af_purchase')
                 THEN 1 ELSE 0 END)             AS user_n_installs
    FROM train
    GROUP BY user_id
),
category_mode AS (
    -- most frequent event category per user, ties broken by recency then name
    SELECT user_id, app_category AS user_pref_category
    FROM (
        SELECT
            i.user_id,
            a.category                                   AS app_category,
            COUNT(*)                                     AS n,
            MAX(i.event_time)                            AS last_seen,
            ROW_NUMBER() OVER (
                PARTITION BY i.user_id
                ORDER BY COUNT(*) DESC, MAX(i.event_time) DESC, a.category
            )                                            AS rn
        FROM train i
        JOIN apps a ON a.app_id = i.app_id
        GROUP BY i.user_id, a.category
    )
    WHERE rn = 1
)
SELECT
    ag.user_id,
    ag.user_n_events,
    ag.user_n_distinct_apps,
    ag.user_n_installs,
    COALESCE(ag.user_n_installs, 0) / NULLIF(ag.user_n_events, 0) AS user_install_rate,
    DATEDIFF('day', ag.last_event_time, :horizon)                 AS user_recency_days,
    cm.user_pref_category
FROM user_agg ag
LEFT JOIN category_mode cm USING (user_id);