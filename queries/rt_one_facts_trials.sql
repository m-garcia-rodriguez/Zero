-- ZERO study -- one row per KEPT RESPONSE (correct and incorrect) to a 1-FACT: one operand is 1, the other
-- is 2-9 (1×n and n×1). The 1-table control for queries/rt_zero_facts_trials.sql, same population and
-- filters (DECISIONS #1-#8): multiplication only, '·' normalised to '×', ages 8-15, 1.2 s fast-wrong rule,
-- Help counts as an error. 1×1 has no second order and 0×1 / 1×0 belong to the 0-facts (#36), so neither
-- is pulled. Feeds analysis/zero_vs_one.ipynb (DECISIONS #42-#47).
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
  AND s.operation RLIKE '^(1[×·][2-9]|[2-9][×·]1)$'   -- 1-facts with a second operand 2-9, both renderings
  AND m.classroom_course_age BETWEEN 8 AND 15
  AND (s.statement_result = 'Correct' OR coalesce(s.statement_seconds_spent, 1e9) >= 1.2)
