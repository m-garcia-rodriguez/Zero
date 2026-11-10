-- ZERO study -- the OTHER multiplications of every student who met a 0-fact: do students who answer "n" to a
-- 0-fact also make the additive error elsewhere? Feeds H1 of analysis/n_error_hypotheses.ipynb (DECISIONS #63).
--
-- Population and filters as DECISIONS #1-#8: A998, ES_2025, '·' -> '×', multiplication only, 1.2 s rule applied,
-- Help counts as an error. Ages 9-15 and only (student, course age) units with at least one 0-fact statement (the
-- same units as queries/n_error_zero_trials.sql). 0-facts are NOT returned here: they define the groups, so the
-- outcome is measured on facts that never enter the group definition.
--
-- Two kinds of rows (column `level`):
--   'fact'   one row per student x age x operation WITH AT LEAST ONE ERROR -- the cells where an additive answer can
--            occur. n_err_ans = errors with a numeric answer (Help without answer cannot be classified, #16).
--   'total'  one row per student x age x ptype with every kept response -- denominators and proficiency.
-- Additive answer: answer = a+b on a WRONG response (2×2 never counts: a+b = a×b). n_add_rev also counts an answer
-- whose digits reversed equal a+b (the stored-reversed artefact of #46), as a sensitivity only.
-- Operand copy (DECISIONS #68): a WRONG answer equal to one of the operands. On a 1-fact n×1 / 1×n (n >= 2) that is
-- the answer 1 -- the mirror of "= n" on the 0-facts (#44); on retrieval it is an operand intrusion (3×4 -> 4).
-- The academic year is a marker: -- {YEAR}.
WITH stmt AS (
    SELECT
        s.student_uuid,
        m.classroom_course_age                                  AS course_age,
        replace(s.operation, '·', '×')                          AS operation,
        s.statement_result,
        s.user_answer,
        try_cast(s.user_answer AS int)                          AS answer,
        s.statement_seconds_spent                               AS rt_seconds
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE s.academic_year_id  = 'ES_2025'   -- {YEAR}
      AND s.activity_codename = 'A998'
      AND m.classroom_course_age BETWEEN 9 AND 15
      AND s.operation RLIKE '^[0-9][×·][0-9]$'
),
units AS (
    SELECT DISTINCT student_uuid, course_age FROM stmt WHERE operation RLIKE '^(0×[0-9]|[0-9]×0)$'
),
kept AS (
    SELECT st.*,
        cast(split(operation, '×')[0] AS int)                   AS a,
        cast(split(operation, '×')[1] AS int)                   AS b,
        CASE WHEN operation RLIKE '^(1×[0-9]|[0-9]×1)$' THEN 'one' ELSE 'retrieval' END AS ptype
    FROM stmt st JOIN units USING (student_uuid, course_age)
    WHERE operation NOT RLIKE '0'                                               -- no 0-facts
      AND (statement_result = 'Correct' OR coalesce(rt_seconds, 1e9) >= 1.2)    -- 1.2 s rule (#7)
),
flagged AS (
    SELECT *,
        CASE WHEN statement_result <> 'Correct' THEN 1 ELSE 0 END                               AS err,
        CASE WHEN statement_result <> 'Correct' AND answer IS NOT NULL THEN 1 ELSE 0 END        AS err_ans,
        CASE WHEN statement_result <> 'Correct' AND a + b <> a * b AND answer = a + b
             THEN 1 ELSE 0 END                                                                  AS add_exact,
        CASE WHEN statement_result <> 'Correct' AND a + b <> a * b
              AND (answer = a + b OR try_cast(reverse(user_answer) AS int) = a + b)
             THEN 1 ELSE 0 END                                                                  AS add_rev,
        CASE WHEN statement_result <> 'Correct' AND (answer = a OR answer = b)
             THEN 1 ELSE 0 END                                                                  AS copy_op
    FROM kept
)
SELECT 'fact' AS level, student_uuid, course_age, ptype, operation,
       min(a) AS left_factor, min(b) AS right_factor,
       count(*) AS n_resp, sum(err) AS n_err, sum(err_ans) AS n_err_ans,
       sum(add_exact) AS n_add, sum(add_rev) AS n_add_rev, sum(copy_op) AS n_copy
FROM flagged
GROUP BY student_uuid, course_age, ptype, operation
HAVING sum(err) > 0
UNION ALL
SELECT 'total' AS level, student_uuid, course_age, ptype, 'ALL' AS operation,
       NULL AS left_factor, NULL AS right_factor,
       count(*) AS n_resp, sum(err) AS n_err, sum(err_ans) AS n_err_ans,
       sum(add_exact) AS n_add, sum(add_rev) AS n_add_rev, sum(copy_op) AS n_copy
FROM flagged
GROUP BY student_uuid, course_age, ptype
