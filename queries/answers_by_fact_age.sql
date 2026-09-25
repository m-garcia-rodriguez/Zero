-- ZERO study -- distribution of the answer given, per multiplication fact and course age.
-- One row per (course_age, operation, statement_result, answer_kind, answer).
-- Source: A998 fluency test. Same population and filters as queries/rt_error_by_fact_age.sql
-- (DECISIONS #1-#10): multiplication only, '·' normalised to '×', ages 8-15, 1.2 s fast-wrong rule.
-- The academic year is a marker: notebooks substitute the -- {YEAR} line and pass sql_text=.
WITH responses AS (
    SELECT
        m.classroom_course_age              AS course_age,
        replace(s.operation, '·', '×')      AS operation,
        s.statement_result,
        CASE WHEN s.user_answer IS NULL              THEN 'none'
             WHEN try_cast(s.user_answer AS int) IS NULL THEN 'malformed'
             ELSE 'int' END                 AS answer_kind,
        try_cast(s.user_answer AS int)      AS answer
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE s.academic_year_id  = 'ES_2025'   -- {YEAR}
      AND s.activity_codename = 'A998'
      AND s.operation RLIKE '^[0-9][×·][0-9]$'          -- multiplication only, both renderings
      AND m.classroom_course_age BETWEEN 8 AND 15
      -- anticipations / mis-taps: non-correct answers faster than 1.2 s
      AND (s.statement_result = 'Correct' OR coalesce(s.statement_seconds_spent, 1e9) >= 1.2)
)
SELECT
    course_age,
    operation,
    cast(split(operation, '×')[0] AS int) AS left_factor,
    cast(split(operation, '×')[1] AS int) AS right_factor,
    statement_result,
    answer_kind,
    answer,
    count(*)                              AS n_responses
FROM responses
GROUP BY course_age, operation, statement_result, answer_kind, answer
ORDER BY course_age, left_factor, right_factor, statement_result, answer
