-- ZERO study -- response time of every CORRECT answer, one row per response, for EVERY directional
-- fact whose two operands differ (a != b). Squares (a×a) are excluded: they have no other order.
-- Feeds the order-asymmetry comparison: is a×b answered faster than b×a, for the 0-facts and for
-- every other pair. Superset of queries/rt_zero_facts_correct.sql -- same population and filters
-- (DECISIONS #1-#6, #9, #18), only the fact list is wider.
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
  AND s.operation RLIKE '^[0-9][×·][0-9]$'            -- multiplication only, both renderings
  AND m.classroom_course_age BETWEEN 8 AND 15
  AND s.statement_result  = 'Correct'                 -- RT is measured on correct answers only (#9)
  AND s.statement_seconds_spent IS NOT NULL
  AND substring(replace(s.operation, '·', '×'), 1, 1)
   <> substring(replace(s.operation, '·', '×'), 3, 1) -- a != b
