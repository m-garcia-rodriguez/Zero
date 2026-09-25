-- ZERO study -- one row per KEPT multiplication response, carrying its POSITION in the student's own
-- sequence of statements. Feeds the within-student sequential analysis: does answering one ×0 problem
-- correctly predict the next ×0 problem (analysis/sequential_zero_consistency.ipynb).
--
-- Population and filters as DECISIONS #1-#8: A998 only, '·' normalised to '×', single-digit
-- multiplication, classroom course age 8-15, 1.2 s fast-wrong rule, Help counts as an error.
--
-- SEQUENCING (DECISIONS #34). A998 stores one timestamp per METRIC (one test administration), not per
-- statement, so elapsed time inside a test is not recorded. Two distance measures are emitted instead:
--   pos_all      position among ALL of the student's statements -- divisions and trials dropped by the
--                1.2 s rule included, because they still consume test time. Differences give
--                trials_between.
--   cum_seconds  seconds of answering time accumulated inside the metric up to and including this
--                statement (sum of statement_seconds_spent). Differences give the elapsed time between
--                two trials of the SAME test; across tests the gap comes from response_at.
--   metric_seq   which test of that student this is (1 = first), so same-test and across-test pairs
--                can be separated.
--
-- PROBLEM TYPE (DECISIONS #24, #36): zero (exactly one operand 0), one (no zero, at least one 1),
-- retrieval (both operands 2-9), zerozero (0×0, reported apart -- DECISIONS #15). Both rule tables are
-- kept because n×1 = n is a rule exactly as n×0 = 0, so "ordinary multiplication" means retrieval only.
--
-- CAP (DECISIONS #37): only the student's first N kept trials of a given type AT A GIVEN AGE are
-- returned. The ×0 count per student is 4 at the median and 14 at the 99th percentile, while retrieval
-- runs to 122; without a cap the problem-type comparison would be a comparison of sequence lengths.
-- Markers substituted by the notebook: -- {YEAR}, -- {CAP}.
WITH all_stmt AS (
    SELECT
        s.student_uuid,
        s.metric_id,
        s.statement_idx,
        s.response_at,
        m.classroom_course_age                          AS course_age,
        s.activity_pack,
        replace(s.operation, '·', '×')                  AS operation,
        s.statement_result,
        s.statement_seconds_spent                       AS rt_seconds
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE s.academic_year_id  = 'ES_2025'   -- {YEAR}
      AND s.activity_codename = 'A998'
      AND m.classroom_course_age BETWEEN 8 AND 15
      -- no operation filter here: divisions occupy positions in the sequence too
),
positioned AS (
    SELECT *,
        row_number() OVER (PARTITION BY student_uuid
                           ORDER BY response_at, metric_id, statement_idx)          AS pos_all,
        dense_rank() OVER (PARTITION BY student_uuid
                           ORDER BY response_at, metric_id)                         AS metric_seq,
        sum(coalesce(rt_seconds, 0)) OVER (PARTITION BY student_uuid, metric_id
                                           ORDER BY statement_idx
                                           ROWS BETWEEN UNBOUNDED PRECEDING
                                                    AND CURRENT ROW)                AS cum_seconds
    FROM all_stmt
),
kept AS (
    SELECT
        student_uuid, course_age, activity_pack, metric_id, metric_seq, statement_idx,
        pos_all, cum_seconds, response_at, statement_result, rt_seconds,
        cast(split(operation, '×')[0] AS int)   AS left_factor,
        cast(split(operation, '×')[1] AS int)   AS right_factor,
        CASE WHEN operation  = '0×0'                                        THEN 'zerozero'
             WHEN operation RLIKE '^(0×[0-9]|[0-9]×0)$'                     THEN 'zero'
             WHEN operation RLIKE '^(1×[0-9]|[0-9]×1)$'                     THEN 'one'
             ELSE 'retrieval' END               AS ptype
    FROM positioned
    WHERE operation RLIKE '^[0-9]×[0-9]$'                    -- multiplication only (#3)
      AND (statement_result = 'Correct'
           OR coalesce(rt_seconds, 1e9) >= 1.2)              -- 1.2 s fast-wrong rule (#7)
),
ranked AS (
    SELECT *,
        row_number() OVER (PARTITION BY student_uuid, course_age, ptype
                           ORDER BY pos_all)                                        AS trial_rank
    FROM kept
)
SELECT
    student_uuid, course_age, ptype, trial_rank,
    left_factor, right_factor, statement_result, rt_seconds,
    metric_seq, statement_idx, pos_all, cum_seconds, response_at, activity_pack
FROM ranked
WHERE trial_rank <= 14   -- {CAP}
