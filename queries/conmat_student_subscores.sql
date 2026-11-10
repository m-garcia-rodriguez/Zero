-- ZERO study -- ConMAT (Innovamat curricular test) per student x test x tag: items and items correct.
-- Tags are the BUBBLE_TREE ones: content_block (5 blocks, primary and secondary codes merged in the notebook)
-- and cognitive_domain (6 processes). One row per tag, plus tag = 'all' for the whole test.
-- The item "Identify the properties of multiplication" (reports_conmat_contents_0007) may contain n×0 and is
-- kept OUT of every tag and of 'all'; it is returned on its own as tag 'mult_properties' (validity check only).
-- Last answer per student x statement. Completed tests only.
-- The academic year is a marker: notebooks substitute the -- {YEAR} line and pass sql_text=.
WITH r AS (
    SELECT user_uuid AS student_uuid, test_uuid, statement_uuid, test_round_name,
           CASE WHEN is_correct THEN 1 ELSE 0 END AS ok,
           row_number() OVER (PARTITION BY user_uuid, test_uuid, statement_uuid ORDER BY answered_at DESC) AS rn
    FROM bi_silver_prod.lean_processed_conmat_reports.student_statement_results
    WHERE academic_year_id = 'ES_2024'   -- {YEAR}
      AND test_completed = 1
), last AS (SELECT * FROM r WHERE rn = 1),
tags AS (
    SELECT DISTINCT test_uuid, statement_uuid, dimension, dimension_value, report_description_translation_code AS code
    FROM bi_silver_prod.lean_processed_conmat_reports.test_statements_tagged
    WHERE standard_structure = 'BUBBLE_TREE'
), props AS (
    SELECT DISTINCT test_uuid, statement_uuid FROM tags WHERE code = 'reports_conmat_contents_0007'
), rows AS (
    SELECT l.student_uuid, l.test_uuid, l.test_round_name, t.dimension, t.dimension_value AS tag, l.ok
    FROM last l JOIN tags t USING (test_uuid, statement_uuid)
    LEFT ANTI JOIN props p USING (test_uuid, statement_uuid)
    UNION ALL
    SELECT l.student_uuid, l.test_uuid, l.test_round_name, 'all', 'all', l.ok
    FROM last l LEFT ANTI JOIN props p USING (test_uuid, statement_uuid)
    UNION ALL
    SELECT l.student_uuid, l.test_uuid, l.test_round_name, 'check', 'mult_properties', l.ok
    FROM last l JOIN props p USING (test_uuid, statement_uuid)
)
SELECT r.student_uuid, r.test_uuid, t.course_age AS conmat_age,
       regexp_extract(r.test_round_name, '(pre|post)$', 1) AS phase,
       r.dimension, r.tag, count(*) AS n_items, sum(r.ok) AS n_correct
FROM rows r JOIN bi_silver_prod.lean_processed_conmat_reports.tests t USING (test_uuid)
GROUP BY 1, 2, 3, 4, 5, 6
