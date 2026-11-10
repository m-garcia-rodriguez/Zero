-- ZERO study -- one row per student x course age: the A998 outcomes and controls for the correlates
-- notebook (zero_rule_correlates). Fact types as #36/#42: 0-facts (0×0 excluded, 0×1/1×0 included),
-- 1-facts (1×1, 0×1, 1×0 excluded), retrieval (both operands 2-9). Filters #1-#8 (1.2 s rule, Help = error).
-- Controls are measured on the retrieval facts only, so they never see a rule fact.
-- SEs are clustered by SCHOOL: A998 ES_2024 carries no classroom_uuid (100% NULL), so #35's classroom
-- clustering is not available in the discovery year; school is used in both years for consistency.
-- The academic year is a marker: notebooks substitute the -- {YEAR} line and pass sql_text=.
WITH kept AS (
    SELECT
        s.student_uuid,
        m.organization_uuid                                          AS school_uuid,
        m.classroom_course_age                                       AS course_age,
        cast(split(replace(s.operation, '·', '×'), '×')[0] AS int)   AS a,
        cast(split(replace(s.operation, '·', '×'), '×')[1] AS int)   AS b,
        CASE WHEN s.statement_result = 'Correct' THEN 1 ELSE 0 END   AS ok,
        s.statement_seconds_spent                                    AS rt
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE s.academic_year_id  = 'ES_2024'   -- {YEAR}
      AND s.activity_codename = 'A998'
      AND s.operation RLIKE '^[0-9][×·][0-9]$'
      AND m.classroom_course_age BETWEEN 9 AND 15
      AND (s.statement_result = 'Correct' OR coalesce(s.statement_seconds_spent, 1e9) >= 1.2)
), typed AS (
    SELECT *,
        CASE WHEN a = 0 AND b = 0 THEN 'none'
             WHEN a = 0 OR  b = 0 THEN 'zero'
             WHEN a = 1 AND b = 1 THEN 'none'
             WHEN a = 1 OR  b = 1 THEN 'one'
             ELSE 'retr' END                                         AS kind
    FROM kept
)
SELECT
    student_uuid,
    course_age,
    max(school_uuid)                                                 AS school_uuid,
    sum(CASE WHEN kind = 'zero' THEN 1 ELSE 0 END)                   AS n_zero,
    sum(CASE WHEN kind = 'zero' THEN ok ELSE 0 END)                  AS k_zero,
    sum(CASE WHEN kind = 'one'  THEN 1 ELSE 0 END)                   AS n_one,
    sum(CASE WHEN kind = 'one'  THEN ok ELSE 0 END)                  AS k_one,
    sum(CASE WHEN kind = 'retr' THEN 1 ELSE 0 END)                   AS n_retr,
    sum(CASE WHEN kind = 'retr' THEN ok ELSE 0 END)                  AS k_retr,
    avg(CASE WHEN kind = 'retr' AND ok = 1 THEN ln(rt) END)          AS lnrt_retr
FROM typed
GROUP BY student_uuid, course_age
HAVING sum(CASE WHEN kind = 'zero' THEN 1 ELSE 0 END) >= 1
