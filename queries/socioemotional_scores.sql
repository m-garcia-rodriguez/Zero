-- ZERO study -- socio-emotional test (MAES: math anxiety, growth mindset, attitudes, goals, stereotypes),
-- one row per student x academic year x subarea, ALL years (ages 4-10), so a student can be linked from an
-- earlier year. Warm-up subareas are training items and are dropped. Exploratory: ~1k A998 students overlap.
SELECT student_uuid, academic_year_id, student_course_age, area, subarea,
       avg(score) AS score, max(num_assessable_statements) AS n_items
FROM bi_gold_prod.dm_research.socioemotional_test_results
WHERE area <> 'Warm-up'
GROUP BY 1, 2, 3, 4, 5
