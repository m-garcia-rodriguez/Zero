-- ZERO study -- median response time and error rate per multiplication fact, by classroom course age AND
-- data-collection WAVE inside the school year. Same population and filters as rt_error_by_fact_age.sql
-- (DECISIONS #1-#11). The wave comes from the test date (fluency_test_metrics.response_at), see DECISIONS #47:
--   T1  September-January   (ES_2025: Nov-Dec 2025;  ES_2024: Dec 2024)
--   T2  February-April      (ES_2025: Mar 2026;      ES_2024: practically empty)
--   T3  May-August          (ES_2025: May-Jun 2026;  ES_2024: May-Jun 2025)
-- n_help is returned so the Help-as-error choice (#8) can be undone at any time.
-- The academic year is a marker: notebooks substitute the -- {YEAR} line and pass sql_text=.
WITH responses AS (
    SELECT
        m.classroom_course_age              AS course_age,
        CASE WHEN month(m.response_at) IN (9, 10, 11, 12, 1) THEN 'T1'
             WHEN month(m.response_at) IN (2, 3, 4)          THEN 'T2'
             ELSE 'T3' END                  AS wave,
        replace(s.operation, '·', '×')      AS operation,
        s.statement_result,
        s.statement_seconds_spent           AS rt_seconds
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE s.academic_year_id  = 'ES_2025'   -- {YEAR}
      AND s.activity_codename = 'A998'
      AND s.operation RLIKE '^[0-9][×·][0-9]$'          -- multiplication only, both renderings
      AND m.classroom_course_age BETWEEN 8 AND 15
      AND (s.statement_result = 'Correct' OR coalesce(s.statement_seconds_spent, 1e9) >= 1.2)
)
SELECT
    course_age,
    wave,
    operation,
    cast(split(operation, '×')[0] AS int)                                    AS left_factor,
    cast(split(operation, '×')[1] AS int)                                    AS right_factor,
    count(*)                                                                 AS n_responses,
    sum(CASE WHEN statement_result <> 'Correct' THEN 1 ELSE 0 END)           AS n_errors,
    sum(CASE WHEN statement_result =  'Help'    THEN 1 ELSE 0 END)           AS n_help,
    avg(CASE WHEN statement_result <> 'Correct' THEN 1.0 ELSE 0.0 END)       AS error_rate,
    percentile(CASE WHEN statement_result = 'Correct' THEN rt_seconds END, 0.5) AS median_rt_correct
FROM responses
GROUP BY course_age, wave, operation
HAVING count(*) >= 30
ORDER BY course_age, wave, left_factor, right_factor
