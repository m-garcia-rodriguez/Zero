-- ZERO study -- one row per KEPT RESPONSE (correct and incorrect) to a fact containing a 0.
-- Feeds the trial-level mixed models: RT on the correct trials, accuracy on all of them.
-- Same population and filters as queries/rt_error_by_fact_age.sql (DECISIONS #1-#8): multiplication
-- only, '·' normalised to '×', ages 8-15, 1.2 s fast-wrong rule, Help counts as an error.
-- Superset of queries/rt_zero_facts_correct.sql, which keeps the correct responses only.
-- The academic year is a marker: notebooks substitute the -- {YEAR} line and pass sql_text=.
SELECT
    m.classroom_course_age                                      AS course_age,
    replace(s.operation, '·', '×')                              AS operation,
    cast(split(replace(s.operation, '·', '×'), '×')[0] AS int)  AS left_factor,
    cast(split(replace(s.operation, '·', '×'), '×')[1] AS int)  AS right_factor,
    s.student_uuid,
    s.statement_result,
    s.statement_seconds_spent                                   AS rt_seconds
FROM bi_gold_prod.dm_research.fluency_test_statements s
JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
WHERE s.academic_year_id  = 'ES_2025'   -- {YEAR}
  AND s.activity_codename = 'A998'
  AND s.operation RLIKE '^(0[×·][0-9]|[0-9][×·]0)$'   -- facts containing a 0, both renderings
  AND m.classroom_course_age BETWEEN 8 AND 15
  AND (s.statement_result = 'Correct' OR coalesce(s.statement_seconds_spent, 1e9) >= 1.2)
