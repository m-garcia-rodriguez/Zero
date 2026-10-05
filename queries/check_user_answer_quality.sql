-- ZERO study -- data check (DECISIONS #45, #46): how the recorded answer relates to the product, per result
-- and per problem group, after the population filters of #1-#7 (multiplication, ages 8-15, 1.2 s rule).
-- Flags: answer = product; answer whose digits REVERSED equal the product (e.g. 7×4 stored as '82');
-- a digit repeated 3+ times (e.g. '4444', '555'); more than 2 significant digits; above 81; Help with an answer.
-- The academic year is a marker: -- {YEAR}.
WITH r AS (
    SELECT
        s.statement_result,
        s.user_answer                                                        AS ua,
        replace(s.operation, '·', '×')                                       AS op,
        cast(substring(s.operation, 1, 1) AS int)                            AS a,
        cast(substring(s.operation, 3, 1) AS int)                            AS b,
        s.statement_seconds_spent                                            AS rt
    FROM bi_gold_prod.dm_research.fluency_test_statements s
    JOIN bi_gold_prod.dm_research.fluency_test_metrics m USING (metric_id)
    WHERE s.academic_year_id  = 'ES_2025'   -- {YEAR}
      AND s.activity_codename = 'A998'
      AND s.operation RLIKE '^[0-9][×·][0-9]$'
      AND m.classroom_course_age BETWEEN 8 AND 15
      AND (s.statement_result = 'Correct' OR coalesce(s.statement_seconds_spent, 1e9) >= 1.2)
)
SELECT
    statement_result,
    CASE WHEN a = 0 OR b = 0 THEN '0-facts' WHEN a = 1 OR b = 1 THEN '1-facts' ELSE 'Retrieval' END AS grp,
    count(*)                                                                                 AS n,
    sum(CASE WHEN ua IS NULL THEN 1 ELSE 0 END)                                              AS no_answer,
    sum(CASE WHEN try_cast(ua AS int) = a * b THEN 1 ELSE 0 END)                             AS eq_product,
    sum(CASE WHEN try_cast(ua AS int) <> a * b AND try_cast(reverse(ua) AS int) = a * b THEN 1 ELSE 0 END)
                                                                                             AS reversed_eq_product,
    sum(CASE WHEN ua RLIKE '^([0-9])\\1{2,}$' THEN 1 ELSE 0 END)                             AS repeated_digit_3plus,
    sum(CASE WHEN length(regexp_replace(ua, '^0+', '')) >= 3 THEN 1 ELSE 0 END)              AS three_plus_digits,
    sum(CASE WHEN try_cast(ua AS int) > 81 THEN 1 ELSE 0 END)                                AS above_81,
    sum(CASE WHEN ua RLIKE '^0+[0-9]' THEN 1 ELSE 0 END)                                     AS leading_zero,
    sum(CASE WHEN ua IS NOT NULL AND try_cast(ua AS int) IS NULL THEN 1 ELSE 0 END)          AS non_numeric,
    percentile(rt, 0.5)                                                                      AS median_rt
FROM r
GROUP BY 1, 2
ORDER BY 1, 2
