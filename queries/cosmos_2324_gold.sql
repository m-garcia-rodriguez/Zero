-- ZERO study -- COSMOS 2023-24 rebuilt in the GOLD format (one row per student, one column per
-- cognitive factor x PRE/POST), because bi_gold_prod.dwh_assessment.f_cosmos_student_results only
-- covers 2025-26 and processed_cosmos_reports.student_cognitive_factor_raw_score starts in 2024-25.
-- Source: bi_silver_prod.processed_cosmos_reports.student_statement_score, the only table with every
-- COSMOS round (one row per student x statement). DECISIONS #50-#53.
--
-- RAW SCORE (#51). The pipeline's 2024-25 raw scores were reproduced exactly from the statement table:
--   working_memory                                  100 * score (a single span statement, score 0-1)
--   magnitude_comparison, numerical_line,
--   processing_speed, reasoning                     correct - error_coefficient * incorrect
--   arithmetic_fluency, execution_speed,
--   number_knowledge                                (correct - error_coefficient * incorrect) / seconds
-- 2023-24 has NO response times for the three speeded tasks (0 in PRE, NULL in POST, also 0 in the
-- bronze payload), so here every factor uses the NET score correct - ec * incorrect. These tasks are
-- time-limited (activity time IQR 27-29 s, 37-38 s, 106-114 s), and in 2024-25 the net score ranks
-- students like the pipeline rate does: Spearman .96-.98 in every round.
--
-- NORM (#52). Percentile = mid-rank of the 2023-24 net score inside the 2024-25 distribution of the
-- SAME task, phase (PRE/POST) and course age (same test E13D532F..., same activity per factor; ~26-31k
-- students per round). 2023-24 itself (~400-800 per round, mostly Mexico) is too small to be its own
-- norm. score_group follows the gold thresholds (lower 30, higher 85 percentile).
WITH stmt AS (
    SELECT
        regexp_extract(test_round_name, '(23-24|24-25)', 1)   AS yr,
        regexp_extract(test_round_name, '(PRE|POST)', 1)       AS phase,
        classroom_course_age                                   AS course_age,
        test_student_uuid, student_uuid, classroom_uuid, organization_uuid,
        CASE cognitive_factor WHEN 'numerical_line' THEN 'numeric_line' ELSE cognitive_factor END AS factor,
        score, error_coefficient, answered_at
    FROM bi_silver_prod.processed_cosmos_reports.student_statement_score
    WHERE test_round_name RLIKE '^Cosmos (PRE|POST) [67]y (23-24|24-25)$'
),
raw AS (
    SELECT
        yr, phase, course_age, test_student_uuid, student_uuid, factor,
        max(classroom_uuid)                                         AS classroom_uuid,
        max(organization_uuid)                                      AS organization_uuid,
        min(answered_at)                                            AS answered_at,
        count(*)                                                    AS n_items,
        sum(CASE WHEN score = 1 THEN 1 ELSE 0 END)                  AS n_correct,
        sum(CASE WHEN score = 0 THEN 1 ELSE 0 END)                  AS n_incorrect,
        CASE WHEN factor = 'working_memory' THEN 100 * max(score)
             ELSE sum(CASE WHEN score = 1 THEN 1 ELSE 0 END)
                  - max(error_coefficient) * sum(CASE WHEN score = 0 THEN 1 ELSE 0 END)
        END                                                         AS raw_score
    FROM stmt
    GROUP BY yr, phase, course_age, test_student_uuid, student_uuid, factor
),
normed AS (
    SELECT
        *,
        sum(CASE WHEN yr = '24-25' THEN 1 ELSE 0 END) OVER (
            PARTITION BY phase, course_age, factor ORDER BY raw_score
            RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)                      AS n_le,
        sum(CASE WHEN yr = '24-25' THEN 1 ELSE 0 END) OVER (
            PARTITION BY phase, course_age, factor, raw_score)                      AS n_eq,
        sum(CASE WHEN yr = '24-25' THEN 1 ELSE 0 END) OVER (
            PARTITION BY phase, course_age, factor)                                 AS n_norm
    FROM raw
),
long AS (
    SELECT
        student_uuid,
        concat(factor, '_', lower(phase))                                           AS k,
        raw_score,
        round(100 * (n_le - 0.5 * n_eq) / n_norm, 2)                                AS percentile,
        CASE WHEN 100 * (n_le - 0.5 * n_eq) / n_norm < 30 THEN 'low'
             WHEN 100 * (n_le - 0.5 * n_eq) / n_norm >= 85 THEN 'high'
             ELSE 'medium' END                                                      AS score_group,
        n_items
    FROM normed
    WHERE yr = '23-24'
),
wide AS (
    SELECT * FROM long
    PIVOT (
        first(raw_score) AS raw_score, first(percentile) AS percentile,
        first(score_group) AS score_group, first(n_items) AS n_items
        FOR k IN (
            'working_memory_pre', 'working_memory_post', 'reasoning_pre', 'reasoning_post',
            'processing_speed_pre', 'processing_speed_post', 'execution_speed_pre', 'execution_speed_post',
            'number_knowledge_pre', 'number_knowledge_post', 'magnitude_comparison_pre', 'magnitude_comparison_post',
            'numeric_line_pre', 'numeric_line_post', 'arithmetic_fluency_pre', 'arithmetic_fluency_post'
        )
    )
),
students AS (
    SELECT
        r.student_uuid,
        max(r.course_age)                                                    AS cosmos_course_age,
        count(DISTINCT r.course_age)                                         AS n_course_ages,
        max_by(r.classroom_uuid, r.phase)                                    AS cosmos_classroom_uuid,   -- PRE sorts after POST: PRE wins
        max_by(r.organization_uuid, r.phase)                                 AS cosmos_organization_uuid,
        max(c.country)                                                       AS country,
        min(CASE WHEN r.phase = 'PRE'  THEN r.answered_at END)               AS pre_answered_at,
        min(CASE WHEN r.phase = 'POST' THEN r.answered_at END)               AS post_answered_at
    FROM raw r
    LEFT JOIN (SELECT DISTINCT classroom_uuid, country
               FROM bi_silver_prod.processed_cosmos_reports.test_round_classrooms
               WHERE test_round_name LIKE '%23-24') c USING (classroom_uuid)
    WHERE r.yr = '23-24'
    GROUP BY r.student_uuid
)
SELECT s.*, w.* EXCEPT (student_uuid)
FROM students s
JOIN wide w USING (student_uuid)
