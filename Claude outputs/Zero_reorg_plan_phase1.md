# Zero repository: reorganisation plan (Phase 1, read-only)

Reference point: commit `1b95d50` ("Further steps", 2026-11-10). The working tree is clean. Nothing has been edited, moved or deleted.

## 1. Inventory

### Notebooks (`analysis/`, 6.5 MB in total, of which 5.9 MB is stored cell output)

| Notebook | Size | Cells (code/md) | Outputs (images) | What it computes | Queries |
|---|---|---|---|---|---|
| `rt_error_scatter` | 1.36 MB | 14 / 14 | 1.33 MB (10) | Median correct RT vs error rate for each directed fact a×b, one panel per age, with 0-facts highlighted. Covers ES_2025, ES_2024 and 8 pooled country groups, plus country coverage. | country_coverage, fact_counts_by_country_age, rt_error_by_fact_age, rt_error_by_fact_age_country |
| `error_types` | 0.25 MB | 8 / 6 | 0.22 MB (3) | What wrong answers on 0-facts look like: 4-way taxonomy, answer histogram, and share of `0×n = n` by age, direction and operand (Wilson CIs) | answers_by_fact_age |
| `n_error_hypotheses` | 1.82 MB | 23 / 15 | 1.72 MB (5) | Five hypotheses on where `= n` comes from: H1 addition, H2 ×1 interference, H3 RT, H4 operand copy, H5 misconception vs slip (binomial mixtures) | n_error_zero_trials, n_error_student_facts, student_controls |
| `rt_zero_direction` | 0.18 MB | 7 / 6 | 0.16 MB (1) | RT of `0×n` vs `n×0`, compared within students (Wilcoxon, rank-biserial, Holm over 7 ages). Split-violin figure plus a `_log` variant. | rt_zero_facts_correct, rt_error_by_fact_age |
| `rt_direction_by_pair` | 0.36 MB | 8 / 8 | 0.33 MB (2) | Order asymmetry for all 45 pairs, grouped as 0-rule, 1-rule and retrieval, with a difficulty control and ES_2024 replication | rt_by_fact_correct, error_by_fact_student, rt_error_by_fact_age |
| `rt_order_robustness` | 0.20 MB | 17 / 11 | 0.15 MB (1) | `n×0 − 0×n` effect from six estimators (pyfixest FE, random-slope mixed model, per-student delta, operand-matched, one trial per student) plus accuracy | rt_zero_facts_trials, student_controls, rt_error_by_fact_age |
| `sequential_zero_consistency` | 1.54 MB | 17 / 6 | 1.47 MB (13) | Whether a correct first ×0 trial predicts a correct second one: adjusted OR, transfer conditions, specificity (tetrachoric), distance, latent classes vs continuum, class evolution (Sankey) | sequential_trials, sequential_trials_answers, student_controls, student_classroom, rt_error_by_fact_age |
| `zero_vs_one` | 0.19 MB | 20 / 12 | 0.11 MB (0, figures are saved and closed, not shown) | The main analyses repeated with 1-facts as their own group: scatter, data-collection wave, error types, RT order, FE model, sequential consistency, and data checks for reviewer Q1–Q3 | answers_by_fact_age, rt_by_fact_correct, rt_error_by_fact_age(_wave), rt_one/zero_facts_trials, check_*; also **reads `cache/sequential_trials__d107039f1884987b.parquet` directly** |
| `cognitive_predictors` | 0.63 MB | 12 / 12 | 0.57 MB (3) | Longitudinal study: COSMOS 2023-24 factors (PRE/POST/combined) predicting 0-rule and 1-rule accuracy at A998, with selection bias (IPW), the Mexico contrast and the COSMOS 2024-25 contrast | cosmos_2324_gold, cosmos_gold_by_year, a998_cosmos_2324_trials, a998_cosmos_student_counts |
| `zero_rule_correlates` | 0.12 MB | 10 / 11 | 0.09 MB (1) | ConMAT subscales, the socio-emotional test (exploratory) and COSMOS as correlates of the 0-rule. Discovery on ES_2024, confirmation on ES_2025. | zero_rule_student_outcomes, conmat_student_subscores, socioemotional_scores, cosmos_* , a998_cosmos_2324_trials |

### Scripts

| File | Size | Role | Used by |
|---|---|---|---|
| `databricks_connector.py` | 1.7 KB | OAuth M2M `query()` | cached_query |
| `cached_query.py` | 3.9 KB | Hash-keyed parquet cache around `query()` | all notebooks |
| `plot_style.py` | 2.1 KB | Fonts, palette, `paint_it_black` | all notebooks |
| `paired_stats.py` | 1.4 KB | Wilcoxon / rank-biserial / median CI / Holm / stars | 7 notebooks |
| `binary_stats.py` | 3.5 KB | Wilson, Newcombe, OR, tetrachoric | n_error, sequential, zero_vs_one |
| `latent_models.py` | 11.3 KB | LCA / IRT fitting | sequential only |
| `analysis/cognitive_predictors.py`, `n_error_hypotheses.py`, `zero_vs_one.py` | 49 / 57 / 41 KB | `# %%` copies of the notebooks. I checked them: every code cell is **byte-identical** to its notebook, so they add no information. | – |
| `queries/*.sql` (27 files) | 0.6–6.6 KB | One data question per file. All 27 are used by at least one notebook. | – |

### Other files

| File | Size | Notes |
|---|---|---|
| `cache/` (60 parquet) | **675 MB** | See §3. Ignored by git, so it exists only on this disk. |
| `analysis/plots/` (108 = 54 figures × PDF+PNG) | 21 MB (PNG 18.2, PDF 2.9) | 50 figures are produced by current code. 4 are orphans (§2). |
| `Claude outputs/` (6 PNG) | 3.2 MB | Older versions of six `seq_zero_*` figures. The checksums all differ from the current ones in `analysis/plots/`. |
| `Summary_2026-11-09.pdf`, `Zero_notebook_conclusions.pdf` | 2.4 MB, 75 KB | Reports generated from the results |
| `DECISIONS.txt` | 71 KB | Decision log |
| `analysis/.decisions_68-69_appended.txt` | 1.7 KB | Leftover. Its #68 and #69 appear verbatim in DECISIONS.txt. |
| `__pycache__/` | 84 KB | Built for both Python 3.10 and 3.14, so two interpreters have been used |
| `README.md`, `requirements.txt`, `.gitignore`, `.env`, `.gitkeep`s | small | Keep |

## 2. Dependencies and dead code

The notebooks are already fairly lean. Inside cells there is very little dead code. Most of the redundancy is across notebooks and in stale files.

**Dead or unused inside notebooks**
- Unused imports:
  - `n_error_hypotheses`: `stars`, `FONT_BIG`
  - `rt_error_scatter`: `pd`
  - `rt_zero_direction`: `median_ci`
  - `zero_vs_one`: `wilcoxon`
- Function defined but never called: `ame()` in `sequential_zero_consistency`
- Variables assigned and never read:
  - `MIN_N` in `rt_order_robustness` c1
  - `HO`, `LI`, `MNL` in `sequential_zero_consistency` c9
- `rt_error_scatter` repeats the same 4-line cell 8 times, once per country group. These could become one loop with the same calls.

**Helpers scattered across cells** (your convention puts them in the first code cell)
- `error_types`, `n_error_hypotheses`, `rt_direction_by_pair`, `rt_order_robustness`, `rt_zero_direction`, `zero_vs_one` and `zero_rule_correlates` define functions in cells further down.
- `cognitive_predictors` and `sequential_zero_consistency` already follow the convention.

**Duplicated across notebooks**
- Identical copies:
  - `_year()` in 3 notebooks (`rt_zero_direction`, `rt_direction_by_pair`, `rt_order_robustness`)
  - `to_ms()` in 2 notebooks (`rt_order_robustness`, `zero_vs_one`)
- Same name but different code:
  - `save` (4 versions)
  - `cq` / `_sub` (2 each)
  - `load_facts` / `plot_facts` (`rt_error_scatter` vs `zero_vs_one`)
  - `group_of`, `dummies`, `coverage`
- The "same name, different code" pairs are **not** safe to unify without changing behaviour.

**Content overlap.** `zero_vs_one` re-runs the 0-fact side of five other notebooks (scatter, error types, RT order, FE model, sequential) with the 1-facts added. It is a cross-cutting control analysis, not a duplicate.

**Figures with no producing code (orphans)**
- `cognitive_predictors_factor_corr`, `_forest_nofilter`, `_selection_balance`
  - Dated 2026-10-21. They seem to have been replaced by the current Figure 3, "Selection and estimator robustness".
- `n_error_h2_size_of_effect_ES_2025`
  - Dated 2026-11-10. It is described in DECISIONS #70 (second #70), but **its code is not in `n_error_hypotheses`**, so neither the figure nor its numbers (+1.18 pp, d .08) can be reproduced from the repo.

**Findings that matter for verification**
- `sequential_zero_consistency` was last run starting at execution count 57, and `rt_direction_by_pair` at 17. Neither was run top to bottom in a clean kernel, so their saved outputs may not match a fresh run.
- `sequential_zero_consistency` sections 2–3 (cells 4–8) **print nothing**. The headline numbers (OR 10.7, 93.8% vs 61.3%, tetrachoric .70/.42/.41) appear only inside figures, mainly Figure 13. A Conclusions bullet therefore can't be checked against a printed output yet.
- `zero_vs_one` §6 reads a hard-coded cache file (`sequential_trials__d107039f…`) whose hash no longer matches the current `sequential_trials.sql`. It is live data and must stay as it is.
- DECISIONS.txt has no #57 or #58, and uses #70 twice (latent classes, and H2 size of effect), the second time after #76. I won't touch the decision log, but you may want to fix this.

## 3. Where the disk space goes (747 MB total)

| Item | Size | Share | Reducible? |
|---|---|---|---|
| `cache/`, live files | ~641 MB | 86% | No. Regenerating them would mean re-querying Databricks, and upstream data may have changed. |
| `cache/`, stale (hash matches no current SQL + parameter combination) | **~34 MB** | 4.5% | Yes, to `_to_delete/` |
| `.git` | 39 MB | 5% | Not by us. It grows with every commit of image-heavy notebooks and PNGs. |
| `analysis/plots` | 21 MB | 3% | Orphans 1.2 MB |
| Notebook outputs | 5.9 MB (5.4 MB images) | 0.8% | Only by clearing image outputs (see Q4) |
| `Claude outputs/` + `__pycache__` + `.py` copies + leftover txt | 3.4 MB | 0.5% | Yes |

Stale cache files (the hash matches neither the current SQL nor any parameter value the notebooks use): `n_error_student_facts__efc5…` (17.5 MB), `student_classroom__39e8…` (5.8), `zero_rule_student_outcomes__e2ce…` (3.8), `conmat_student_subscores__b0f8…` (3.3), `__c199…` (2.0), `__4a03…` (0.3), `a998_cosmos_2324_trials__23e4…` (0.8), `rt_error_by_fact_age__47c9…`, `__9a56…`, `__b47f…`, `rt_error_by_fact_age_country__2780…`, `__f551…` (all < 30 KB).

**Realistic saving is ~40–45 MB (~6%).** The repository is large because of live cached data, not clutter.

## 4. Proposed structure

Notebooks stay **flat in `analysis/`**. Every notebook finds the repo root with `Path.cwd().parent if cwd.name == "analysis"`, so subfolders would break every path. The numbers group the notebooks by research question instead:

```
Zero/
├── README.md                  (updated: new names, one line per notebook, grouped by question)
├── DECISIONS.txt
├── *.py helpers               (unchanged)
├── queries/  cache/           (cache minus stale files)
├── reports/                   Summary_2026-11-09.pdf, Zero_notebook_conclusions.pdf
├── _to_delete/                (gitignored; everything removed, for you to check)
└── analysis/
    │  A. What happens on the 0-facts
    ├── 01_rt_vs_error_by_fact.ipynb
    ├── 02_error_types.ipynb
    ├── 03_n_error_origins.ipynb
    │  B. Operand-order asymmetry
    ├── 04_order_zero_direction.ipynb
    ├── 05_order_all_pairs.ipynb
    ├── 06_order_robustness.ipynb
    │  C. Is zero-rule use a stable student trait?
    ├── 07_sequential_consistency.ipynb
    │  D. Specific to zero? (control: 1-facts)
    ├── 08_zero_vs_one.ipynb
    │  E. Who uses the zero rule?
    ├── 09_cognitive_predictors_cosmos.ipynb
    ├── 10_zero_rule_correlates_conmat.ipynb
    └── plots/
```

**What goes into each new notebook.** Each one is a rename of exactly one current notebook, with only the in-notebook changes listed:

| New notebook | From | Changes inside |
|---|---|---|
| 01_rt_vs_error_by_fact | rt_error_scatter (all 28 cells) | Drop unused `pd`. Merge the 8 country cells into one loop. One Conclusions cell. |
| 02_error_types | error_types (all) | Move `load_answers/save/coverage/answer_hist/type_mix/type_figure/identity_share/wilson/identity_figure` to cell 1. Conclusions. |
| 03_n_error_origins | n_error_hypotheses (all) | Move the plot/stat helpers to cell 1. Drop `stars`, `FONT_BIG`. Merge mid-notebook "Conclusions" (c34) and "8. What we tested" (c37) into one final Conclusions. |
| 04_order_zero_direction | rt_zero_direction (all) | Move helpers to cell 1. Drop `median_ci`. Conclusions. |
| 05_order_all_pairs | rt_direction_by_pair (all) | Move `describe/groups_report/dot_panel/pair_figure/replication_figure` to cell 1. Conclusions. |
| 06_order_robustness | rt_order_robustness (all) | Move helpers to cell 1. Drop `MIN_N`. Merge "Conclusion — what the data show" into Conclusions. |
| 07_sequential_consistency | sequential_zero_consistency (all) | Drop `ame()` and `HO/LI/MNL`. **Add display of already-computed tables** (`desc`, `EST`, `by_type`) so the numbers are visible. "5. What this does and does not say" becomes Conclusions. |
| 08_zero_vs_one | zero_vs_one (all) | Move helpers to cell 1. Drop `wilcoxon`. Hard-coded parquet path untouched. Conclusions. |
| 09_cognitive_predictors_cosmos | cognitive_predictors (all) | Already structured. Conclusions cell rewritten to format. |
| 10_zero_rule_correlates_conmat | zero_rule_correlates (all) | Move `cosmos_z`, `beta_ci` to cell 1. The code cell under "Conclusions" becomes a "Summary" table above it. "Interpretation" is merged into Conclusions. |

Conclusions in all 10 notebooks will follow your format: 3–6 bullets, specific numbers, and Kraft (2020) labels.

## 5. Plan table: current file → action → reason

| Current file | Action | Reason |
|---|---|---|
| analysis/rt_error_scatter.ipynb | keep → rename 01_rt_vs_error_by_fact | Group A |
| analysis/error_types.ipynb | keep → rename 02_error_types | Group A |
| analysis/n_error_hypotheses.ipynb | keep → rename 03_n_error_origins | Group A |
| analysis/rt_zero_direction.ipynb | keep → rename 04_order_zero_direction | Group B |
| analysis/rt_direction_by_pair.ipynb | keep → rename 05_order_all_pairs | Group B |
| analysis/rt_order_robustness.ipynb | keep → rename 06_order_robustness | Group B |
| analysis/sequential_zero_consistency.ipynb | keep → rename 07_sequential_consistency | Group C |
| analysis/zero_vs_one.ipynb | keep → rename 08_zero_vs_one | Group D (see Q1) |
| analysis/cognitive_predictors.ipynb | keep → rename 09_cognitive_predictors_cosmos | Group E |
| analysis/zero_rule_correlates.ipynb | keep → rename 10_zero_rule_correlates_conmat | Group E |
| analysis/cognitive_predictors.py, n_error_hypotheses.py, zero_vs_one.py | **⚠ Q2** delete (→ _to_delete) | Byte-identical copies. They would go out of sync after Phase 2. |
| *.py helpers (6) | keep | All used |
| queries/*.sql (27) | keep | All used |
| cache: 12 stale files (~34 MB, listed in §3) | delete (→ _to_delete) | Hash matches no current query. Phase 3 will confirm that no notebook asks for them. |
| cache/sequential_trials__d107039f1884987b.parquet | keep | Stale hash, but read directly by zero_vs_one §6 |
| cache: all other files | keep | Live |
| analysis/plots: 50 produced figures (100 files) | keep | Regenerated by the notebooks |
| analysis/plots/cognitive_predictors_{factor_corr, forest_nofilter, selection_balance}.* | **⚠ Q3** archive (→ _to_delete) | No code produces them. Superseded in October. |
| analysis/plots/n_error_h2_size_of_effect_ES_2025.* | **⚠ Q3** keep | No code produces it, but DECISIONS #70 relies on it |
| Claude outputs/ (6 PNG) | delete (→ _to_delete) | Older versions of current figures |
| analysis/.decisions_68-69_appended.txt | delete (→ _to_delete) | Already in DECISIONS.txt verbatim |
| __pycache__/ | delete (→ _to_delete) | Regenerated automatically, ignored by git |
| Summary_2026-11-09.pdf, Zero_notebook_conclusions.pdf | keep → move to reports/ | Reports, not analysis |
| README.md | keep, update | New names and grouping |
| .gitignore | keep, add `_to_delete/` | So the archive isn't committed |
| DECISIONS.txt, requirements.txt, .env, .gitkeep | keep | – |

## 6. Questions for you (I have not decided these)

- **Q1. zero_vs_one.** My recommendation is to keep it as one notebook, 08. The alternative is to split its sections into 01/02/04/06/07. That would group strictly by question, but it means merging code that reuses names like `load_facts`, `plot_facts`, `cq` and `YEAR` with different definitions, which is a real risk of silently changing results.
- **Q2. The `.py` copies.** Should I archive them, or regenerate a `.py` copy for all 10 notebooks after Phase 2 (for example, if you use them for readable git diffs)?
- **Q3. Orphan figures.** Should I archive the three October `cognitive_predictors_*` figures? Should I keep `n_error_h2_size_of_effect` as it is? Its code is missing, so do you have it somewhere, for example in another session?
- **Q4. Cell outputs.** My recommendation is to keep all outputs, so the Conclusions are visibly backed by output. Clearing inline images would save only 5.4 MB, mostly helping git growth.
- **Q5. Shared helpers.** Should the identical `_year()` and `to_ms()` stay as one copy in each notebook's first cell (recommended, following your convention), or go into a small shared module?
- **Q6. sequential_zero_consistency.** Is it OK to add display-only lines for tables that are already computed, so the Conclusions numbers are visible? There would be no new calculations.

## 7. Verification plan (Phase 3)

1. **Baseline first.** Before any change, re-run the 10 committed notebooks top to bottom and record their key numbers. Two notebooks weren't run in a clean kernel, so the saved outputs aren't a reliable reference.
2. During the rerun, log every cache file that `cached_query` reads. This confirms the 12 stale files are never requested and that no notebook goes to Databricks.
3. Run the 10 new notebooks the same way. Compare all printed tables cell by cell and all key numbers with the baseline. Check every Conclusions bullet against an output.
4. Report the structure, before/after size, everything in `_to_delete/`, and any issues.

Open point: the Python on your machine's Linux VM doesn't have pandas yet. I'll install the `requirements.txt` stack there, or, if that's blocked, run on a copy in my cloud workspace. Both should use the same Python version you use. `__pycache__` shows both 3.10 and 3.14, so which one do you run the notebooks with?
