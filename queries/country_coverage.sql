-- ZERO study -- how much usable A998 multiplication data each country has.
-- Same filters as rt_error_by_fact_age.sql (multiplication only, ages 8-15, 1.2 s rule),
-- so the counts are directly comparable to what the scatter figures are built on.
WITH responses AS (
    SELECT
        split(s.academic_year_id, '_')[0]   AS country,
        s.academic_year_id,
        m.classroom_course_age              AS course_age,
        replace(s.operation, '·', '×')      AS operation
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE s.activity_codename = 'A998'
      AND s.operation RLIKE '^[0-9][×·][0-9]$'
      AND m.classroom_course_age BETWEEN 8 AND 15
      AND (s.statement_result = 'Correct' OR coalesce(s.statement_seconds_spent, 1e9) >= 1.2)
)
SELECT
    country,
    count(*)                                                              AS n_responses,
    sum(CASE WHEN operation LIKE '0%' OR operation LIKE '%0' THEN 1 ELSE 0 END) AS n_zero_fact_responses,
    count(DISTINCT academic_year_id)                                      AS n_years,
    count(DISTINCT course_age)                                            AS n_ages,
    min(course_age)                                                       AS min_age,
    max(course_age)                                                       AS max_age
FROM responses
GROUP BY country
ORDER BY n_responses DESC
