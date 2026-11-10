-- ZERO study -- errors per STUDENT x directional fact, for every fact whose two operands differ
-- (a != b). One row per student x course age x operation (e.g. 3×8 and 8×3 are separate rows).
-- Feeds panel B of rt_direction_by_pair: the within-student error-rate difference
-- large-first − small-first, with a student-level CI like the RT panel (A).
-- Same population and filters as rt_error_by_fact_age.sql: 1.2 s fast-wrong rule (#7),
-- Help counts as an error (#8). The >= 30 cell rule (#11) is applied in the notebook by joining
-- to the rt_error_by_fact_age cells on (course_age, operation), as load_pairs() does for RT.
-- The academic year is a marker: notebooks substitute the -- {YEAR} line and pass sql_text=.
WITH responses AS (
    SELECT
        s.student_uuid,
        m.classroom_course_age              AS course_age,
        replace(s.operation, '·', '×')      AS operation,
        s.statement_result
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE s.academic_year_id  = 'ES_2025'   -- {YEAR}
      AND s.activity_codename = 'A998'
      AND s.operation RLIKE '^[0-9][×·][0-9]$'          -- multiplication only, both renderings
      AND m.classroom_course_age BETWEEN 8 AND 15
      AND substring(replace(s.operation, '·', '×'), 1, 1)
       <> substring(replace(s.operation, '·', '×'), 3, 1) -- a != b
      -- anticipations / mis-taps: non-correct answers faster than 1.2 s (#7)
      AND (s.statement_result = 'Correct' OR coalesce(s.statement_seconds_spent, 1e9) >= 1.2)
)
SELECT
    student_uuid,
    course_age,
    operation,
    cast(split(operation, '×')[0] AS int)                                    AS left_factor,
    cast(split(operation, '×')[1] AS int)                                    AS right_factor,
    count(*)                                                                 AS n_responses,
    sum(CASE WHEN statement_result <> 'Correct' THEN 1 ELSE 0 END)           AS n_errors
FROM responses
GROUP BY student_uuid, course_age, operation
