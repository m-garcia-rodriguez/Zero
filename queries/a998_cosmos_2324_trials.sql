-- ZERO study -- every kept A998 multiplication response, in the follow-up years, of the students who
-- sat COSMOS in 2023-24 (any round). One row per response. Feeds analysis/cognitive_predictors.ipynb
-- (DECISIONS #50-#56).
-- Same population filters as the rest of the study (DECISIONS #3, #4, #8): multiplication only,
-- '·' normalised to '×', Help counts as an error. The 1.2 s fast-wrong rule (#7) is NOT applied here: it is
-- returned as the flag fast_wrong and applied in the notebook, so the analysis can be run with and without it
-- (DECISIONS #58). NOT restricted to Spain and not
-- to ages 8-15: the COSMOS 2023-24 cohort is mostly Mexican and is 8-9 years old in the follow-up years.
-- The follow-up years are a marker (-- {YEARS}); the default is 2024-25 and 2025-26 for every country.
SELECT
    s.academic_year_id,
    split(s.academic_year_id, '_')[0]                            AS a998_country,
    m.classroom_course_age                                       AS course_age,
    m.classroom_uuid                                             AS a998_classroom_uuid,
    s.metric_id,
    m.response_at,
    s.statement_idx,
    replace(s.operation, '·', '×')                               AS operation,
    cast(split(replace(s.operation, '·', '×'), '×')[0] AS int)   AS left_factor,
    cast(split(replace(s.operation, '·', '×'), '×')[1] AS int)   AS right_factor,
    s.student_uuid,
    s.statement_result,
    s.statement_seconds_spent                                    AS rt_seconds,
    try_cast(s.user_answer AS int)                               AS answer,
    (s.statement_result <> 'Correct'
     AND coalesce(s.statement_seconds_spent, 1e9) < 1.2)         AS fast_wrong
FROM bi_gold_prod.dm_research.fluency_test_statements s
JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
WHERE s.academic_year_id RLIKE '_(2024|2025)$'   -- {YEARS}
  AND s.activity_codename = 'A998'
  AND s.operation RLIKE '^[0-9][×·][0-9]$'
  AND s.student_uuid IN (
        SELECT student_uuid FROM bi_silver_prod.processed_cosmos_reports.test_round_students
        WHERE test_round_name LIKE '%23-24')
