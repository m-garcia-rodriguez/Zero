-- ZERO study -- A998 multiplication, aggregated to one row per student x academic year x course age x wave, for the
-- students of ONE COSMOS round year. Used for the large-cohort contrast (COSMOS 2024-25 -> A998 2025-26, where the
-- 7-year-olds are at age 8 and A998 has 1xn but no 0-facts). DECISIONS #3, #4, #8, #47, #61.
-- Fact types as #42 / the notebook: zero (one operand 0, 0x0 out), one (one operand 1, no 0, 1x1 out), retrieval (2-9).
-- Counts with the 1.2 s fast-wrong rule (#7) applied (n_*, k_*) and without it (n_*_all, k_*_all).
-- Markers: -- {COSMOS_YEAR} (default '24-25') and -- {YEARS} (default 2025, i.e. 2025-26).
WITH cos AS (
    SELECT DISTINCT student_uuid
    FROM bi_silver_prod.processed_cosmos_reports.test_round_students
    WHERE test_round_name RLIKE '^Cosmos (PRE|POST) [67]y 24-25$'   -- {COSMOS_YEAR}
),
t AS (
    SELECT
        s.student_uuid,
        s.academic_year_id,
        split(s.academic_year_id, '_')[0]                                   AS a998_country,
        m.classroom_course_age                                              AS course_age,
        CASE WHEN month(m.response_at) IN (9, 10, 11, 12, 1) THEN 'T1'
             WHEN month(m.response_at) IN (2, 3, 4)          THEN 'T2' ELSE 'T3' END AS wave,
        m.classroom_uuid                                                    AS a998_classroom_uuid,
        cast(split(replace(s.operation, '·', '×'), '×')[0] AS int)          AS a,
        cast(split(replace(s.operation, '·', '×'), '×')[1] AS int)          AS b,
        CASE WHEN s.statement_result = 'Correct' THEN 1 ELSE 0 END          AS correct,
        CASE WHEN s.statement_result <> 'Correct'
              AND coalesce(s.statement_seconds_spent, 1e9) < 1.2 THEN 1 ELSE 0 END AS fw
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE s.academic_year_id RLIKE '_(2025)$'   -- {YEARS}
      AND s.activity_codename = 'A998'
      AND s.operation RLIKE '^[0-9][×·][0-9]$'
      AND s.student_uuid IN (SELECT student_uuid FROM cos)
),
typed AS (
    SELECT *,
        CASE WHEN a = 0 AND b = 0 THEN '0x0' WHEN a = 0 OR b = 0 THEN 'zero'
             WHEN a = 1 AND b = 1 THEN '1x1' WHEN a = 1 OR b = 1 THEN 'one' ELSE 'retrieval' END AS type
    FROM t
)
SELECT
    student_uuid, academic_year_id, a998_country, course_age, wave,
    max(a998_classroom_uuid)                                                AS a998_classroom_uuid,
    sum(CASE WHEN type = 'zero'      AND fw = 0 THEN 1 ELSE 0 END)          AS n_zero,
    sum(CASE WHEN type = 'zero'      AND fw = 0 THEN correct ELSE 0 END)    AS k_zero,
    sum(CASE WHEN type = 'one'       AND fw = 0 THEN 1 ELSE 0 END)          AS n_one,
    sum(CASE WHEN type = 'one'       AND fw = 0 THEN correct ELSE 0 END)    AS k_one,
    sum(CASE WHEN type = 'retrieval' AND fw = 0 THEN 1 ELSE 0 END)          AS n_retrieval,
    sum(CASE WHEN type = 'retrieval' AND fw = 0 THEN correct ELSE 0 END)    AS k_retrieval,
    sum(CASE WHEN type = 'one'       THEN 1 ELSE 0 END)                     AS n_one_all,
    sum(CASE WHEN type = 'one'       THEN correct ELSE 0 END)               AS k_one_all,
    sum(CASE WHEN type = 'retrieval' THEN 1 ELSE 0 END)                     AS n_retrieval_all,
    sum(CASE WHEN type = 'retrieval' THEN correct ELSE 0 END)               AS k_retrieval_all
FROM typed
GROUP BY student_uuid, academic_year_id, a998_country, course_age, wave
