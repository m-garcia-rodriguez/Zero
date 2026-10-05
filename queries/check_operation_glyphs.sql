-- ZERO study -- data check (DECISIONS #3, #45): every operation string pattern in A998, digits replaced by 'd'.
-- Answers the question of whether multiplications written with other signs (x, X, *, ·) are lost by the
-- '^[0-9][×·][0-9]$' filter. All academic years, all countries, no other filter.
SELECT
    split(academic_year_id, '_')[0]          AS country,
    regexp_replace(operation, '[0-9]', 'd')  AS pattern,
    count(*)                                 AS n_statements,
    min(operation)                           AS example_min,
    max(operation)                           AS example_max
FROM bi_gold_prod.dm_research.fluency_test_statements
WHERE activity_codename = 'A998'
GROUP BY 1, 2
ORDER BY n_statements DESC
