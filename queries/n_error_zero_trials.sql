-- ZERO study -- one row per response to a fact containing a 0, with what came JUST BEFORE it in the same test.
-- Feeds analysis/n_error_hypotheses.ipynb: the x1-interference check (H2) and the RT of the "= n" answer (H3).
--
-- Population as DECISIONS #1-#6: A998, ES_2025, '·' normalised to '×', classroom course age 9-15 (age 8 has no
-- 0-facts, #12). The 1.2 s rule (#7) is NOT applied here: every response is returned with a fast_wrong flag and the
-- notebook applies it, because (a) a statement dropped by #7 was still SEEN, so it must count as context for the
-- next trial, and (b) the RT comparison needs the same 1.2 s floor on correct and wrong answers (DECISIONS #64).
--
-- CONTEXT (DECISIONS #63). The lags run over ALL statements of the same test (metric_id), ordered by statement_idx,
-- divisions included -- a division between a 1-fact and a 0-fact is a real gap. A lag that falls before the start
-- of the test is NULL ('none'). prev1_* is the statement immediately before, prev2_* two before, prev3_* three.
-- Answer column as #14: user_answer parsed with try_cast; NULL when no answer or non-numeric.
-- The academic year is a marker: -- {YEAR}.
WITH all_stmt AS (
    SELECT
        s.student_uuid,
        m.classroom_course_age                                  AS course_age,
        s.metric_id,
        s.statement_idx,
        replace(s.operation, '·', '×')                          AS operation,
        s.statement_result,
        try_cast(s.user_answer AS int)                          AS answer,
        s.statement_seconds_spent                               AS rt_seconds
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE s.academic_year_id  = 'ES_2025'   -- {YEAR}
      AND s.activity_codename = 'A998'
      AND m.classroom_course_age BETWEEN 9 AND 15
),
typed AS (
    SELECT *,
        CASE WHEN operation NOT RLIKE '^[0-9]×[0-9]$'                      THEN 'division'
             WHEN operation = '0×0'                                        THEN 'zerozero'
             WHEN operation RLIKE '^(0×[0-9]|[0-9]×0)$'                    THEN 'zero'
             WHEN operation RLIKE '^(1×[0-9]|[0-9]×1)$'                    THEN 'one'
             ELSE 'retrieval' END                                          AS ptype
    FROM all_stmt
),
lagged AS (
    SELECT *,
        count(*) OVER (PARTITION BY metric_id)                                         AS n_in_test,
        row_number() OVER (PARTITION BY metric_id ORDER BY statement_idx)              AS pos_in_test,
        coalesce(lag(ptype, 1) OVER w, 'none')                                         AS prev1_ptype,
        coalesce(lag(ptype, 2) OVER w, 'none')                                         AS prev2_ptype,
        coalesce(lag(ptype, 3) OVER w, 'none')                                         AS prev3_ptype,
        lag(operation, 1) OVER w                                                       AS prev1_operation,
        lag(statement_result, 1) OVER w                                                AS prev1_result,
        lag(statement_result, 2) OVER w                                                AS prev2_result,
        lag(statement_result, 3) OVER w                                                AS prev3_result,
        lag(answer, 1) OVER w                                                          AS prev1_answer
    FROM typed
    WINDOW w AS (PARTITION BY metric_id ORDER BY statement_idx)
)
SELECT
    student_uuid, course_age, metric_id, statement_idx, pos_in_test, n_in_test,
    operation,
    cast(split(operation, '×')[0] AS int)                                    AS left_factor,
    cast(split(operation, '×')[1] AS int)                                    AS right_factor,
    ptype, statement_result, answer, rt_seconds,
    CASE WHEN statement_result <> 'Correct' AND coalesce(rt_seconds, 1e9) < 1.2
         THEN 1 ELSE 0 END                                                   AS fast_wrong,
    prev1_ptype, prev2_ptype, prev3_ptype, prev1_operation,
    prev1_result, prev2_result, prev3_result, prev1_answer
FROM lagged
WHERE ptype IN ('zero', 'zerozero')
