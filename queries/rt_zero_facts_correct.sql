-- ZERO study -- response time of every CORRECT answer to a fact containing a 0,
-- one row per response, for the RT-distribution (violin) notebook.
-- Source: A998 fluency test. Same population and filters as queries/rt_error_by_fact_age.sql
-- (DECISIONS #1-#6): multiplication only, '·' normalised to '×', ages 8-15.
-- The 1.2 s fast-wrong rule (#7) does not bite here: it only ever drops non-correct responses.
-- Only the 0-facts are pulled (0×n, n×0 and 0×0); the >= 30 rule (#11) is applied in the
-- notebook against queries/rt_error_by_fact_age.sql, which counts ALL responses, not just correct ones.
-- The academic year is a marker: notebooks substitute the -- {YEAR} line and pass sql_text=.
SELECT
    m.classroom_course_age                                      AS course_age,
    replace(s.operation, '·', '×')                              AS operation,
    cast(split(replace(s.operation, '·', '×'), '×')[0] AS int)  AS left_factor,
    cast(split(replace(s.operation, '·', '×'), '×')[1] AS int)  AS right_factor,
    s.student_uuid,
    s.statement_seconds_spent                                   AS rt_seconds
FROM bi_gold_prod.dm_research.fluency_test_statements s
JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
WHERE s.academic_year_id  = 'ES_2025'   -- {YEAR}
  AND s.activity_codename = 'A998'
  AND s.operation RLIKE '^(0[×·][0-9]|[0-9][×·]0)$'   -- facts containing a 0, both renderings
  AND m.classroom_course_age BETWEEN 8 AND 15
  AND s.statement_result  = 'Correct'                 -- RT is measured on correct answers only (#9)
  AND s.statement_seconds_spent IS NOT NULL
