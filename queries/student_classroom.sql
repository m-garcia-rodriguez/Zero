-- ZERO study -- the classroom and school each student sits in, for clustering standard errors.
-- One row per student. In ES_2025 no student appears in more than one classroom (checked: 172,056
-- students, 172,056 student × classroom combinations), so the lookup is unambiguous.
-- Used by analysis/sequential_zero_consistency.ipynb, where the unit of analysis is the STUDENT
-- (one trial pair each) and the trial-level clustering of the other notebooks no longer applies:
-- students are nested in classrooms that sat the same test on the same day (DECISIONS #35).
-- The academic year is a marker: -- {YEAR}.
SELECT
    student_uuid,
    max(classroom_uuid)    AS classroom_uuid,
    max(organization_uuid) AS school_uuid
FROM bi_gold_prod.dm_research.fluency_test_metrics
WHERE academic_year_id  = 'ES_2025'   -- {YEAR}
  AND activity_codename = 'A998'
  AND classroom_course_age BETWEEN 8 AND 15
GROUP BY student_uuid
