-- ZERO study -- per-student controls measured OUTSIDE the zero problems, so they are independent of
-- the effect being tested: general multiplication proficiency and general response speed.
-- Baseline set: every kept response to a fact with NO zero operand. The _nr columns repeat the same
-- measures on the non-rule facts only (both operands 2-9), since n×1 = n is a rule like n×0 = 0 and
-- arguably does not belong in a "general proficiency" baseline either.
-- Filters as DECISIONS #1-#8. The academic year is a marker: -- {YEAR}.
WITH kept AS (
    SELECT
        s.student_uuid,
        m.classroom_course_age                                       AS course_age,
        cast(split(replace(s.operation, '·', '×'), '×')[0] AS int)   AS a,
        cast(split(replace(s.operation, '·', '×'), '×')[1] AS int)   AS b,
        s.statement_result,
        s.statement_seconds_spent                                    AS rt
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE s.academic_year_id  = 'ES_2025'   -- {YEAR}
      AND s.activity_codename = 'A998'
      AND s.operation RLIKE '^[0-9][×·][0-9]$'
      AND m.classroom_course_age BETWEEN 8 AND 15
      AND (s.statement_result = 'Correct' OR coalesce(s.statement_seconds_spent, 1e9) >= 1.2)
)
SELECT
    student_uuid,
    max(course_age)                                                          AS course_age,
    count(*)                                                                 AS n_base,
    avg(CASE WHEN statement_result = 'Correct' THEN 1.0 ELSE 0.0 END)        AS acc_base,
    avg(CASE WHEN statement_result = 'Correct' THEN ln(rt) END)              AS mean_lnrt_base,
    percentile(CASE WHEN statement_result = 'Correct' THEN rt END, 0.5)      AS median_rt_base,
    sum(CASE WHEN a > 1 AND b > 1 THEN 1 ELSE 0 END)                         AS n_nr,
    avg(CASE WHEN a > 1 AND b > 1 THEN
             CASE WHEN statement_result = 'Correct' THEN 1.0 ELSE 0.0 END END)   AS acc_nr,
    avg(CASE WHEN a > 1 AND b > 1 AND statement_result = 'Correct' THEN ln(rt) END) AS mean_lnrt_nr
FROM kept
WHERE a <> 0 AND b <> 0                      -- the controls never see a zero problem
GROUP BY student_uuid
HAVING count(*) >= 20                        -- a control needs a minimum of evidence
