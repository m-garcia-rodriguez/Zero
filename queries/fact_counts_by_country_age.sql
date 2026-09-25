-- ZERO study -- raw response counts per country x course age x fact (all academic years).
-- No minimum applied: this is the input for deciding how small a country group can be
-- while still clearing the >=30 rule (decision #11) on most facts.
SELECT
    split(s.academic_year_id, '_')[0]   AS country,
    m.classroom_course_age              AS course_age,
    replace(s.operation, '·', '×')      AS operation,
    count(*)                            AS n_responses
FROM bi_gold_prod.dm_research.fluency_test_statements s
JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
WHERE s.activity_codename = 'A998'
  AND s.operation RLIKE '^[0-9][×·][0-9]$'
  AND m.classroom_course_age BETWEEN 8 AND 15
  AND (s.statement_result = 'Correct' OR coalesce(s.statement_seconds_spent, 1e9) >= 1.2)
GROUP BY 1, 2, 3
