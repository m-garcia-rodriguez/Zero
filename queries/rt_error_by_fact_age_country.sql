-- ZERO study -- median response time and error rate per multiplication fact,
-- broken down by classroom course age, for a GROUP OF COUNTRIES (all academic years pooled).
-- Source: A998 fluency test (statements joined to metrics for course age).
-- All academic years are pooled; the country group is a marker: notebooks substitute the
-- -- {COUNTRIES} line and pass sql_text=. Needed because only ES has volume within a single year.
WITH responses AS (
    SELECT
        m.classroom_course_age              AS course_age,
        replace(s.operation, '·', '×')      AS operation,
        s.statement_result,
        s.statement_seconds_spent           AS rt_seconds
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE split(s.academic_year_id, '_')[0] IN ('ES')   -- {COUNTRIES}
      AND s.activity_codename = 'A998'
      AND s.operation RLIKE '^[0-9][×·][0-9]$'          -- multiplication only, both renderings
      AND m.classroom_course_age BETWEEN 8 AND 15
      -- anticipations / mis-taps: non-correct answers faster than 1.2 s
      AND (s.statement_result = 'Correct' OR coalesce(s.statement_seconds_spent, 1e9) >= 1.2)
)
SELECT
    course_age,
    operation,
    cast(split(operation, '×')[0] AS int)                                    AS left_factor,
    cast(split(operation, '×')[1] AS int)                                    AS right_factor,
    count(*)                                                                 AS n_responses,
    sum(CASE WHEN statement_result <> 'Correct' THEN 1 ELSE 0 END)           AS n_errors,
    sum(CASE WHEN statement_result =  'Help'    THEN 1 ELSE 0 END)           AS n_help,
    avg(CASE WHEN statement_result <> 'Correct' THEN 1.0 ELSE 0.0 END)       AS error_rate,
    percentile(CASE WHEN statement_result = 'Correct' THEN rt_seconds END, 0.5) AS median_rt_correct
FROM responses
GROUP BY course_age, operation
HAVING count(*) >= 30
ORDER BY course_age, left_factor, right_factor
