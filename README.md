# ZERO

Estudi del comportament del número zero en tests de multiplicació: com responen els alumnes a les operacions que inclouen el 0, tant en precisió (accuracy) com en velocitat de resposta (RT).

## Contingut del repositori

```
Zero/
├── databricks_connector.py   # connexió OAuth M2M a Databricks (query() -> pd.DataFrame)
├── cached_query.py           # wrapper de query() amb cache basat en hash del .sql
├── plot_style.py             # estil compartit de les figures (font, paleta, mides, paint_it_black)
├── paired_stats.py           # test aparellat compartit (Wilcoxon, rank-biserial, IC mediana, Holm)
├── binary_stats.py           # associació 2×2 compartida (Wilson, Newcombe, OR, tetracòrica)
├── latent_models.py          # classes latents (LCA) i IRT, usat per 07_sequential_consistency
├── .env                      # credencials (no versionat)
├── requirements.txt
├── DECISIONS.txt             # totes les decisions analítiques preses
├── queries/                  # .sql, una pregunta de dades per fitxer
├── cache/                    # resultats cachejats en .parquet (no versionat)
├── reports/
│   ├── Summary_2026-11-09.pdf
│   └── Zero_notebook_conclusions.pdf
├── _to_delete/               # fitxers retirats pendents d'esborrar (no versionat)
└── analysis/
    ├── 01_rt_vs_error_by_fact.ipynb
    ├── 02_error_types.ipynb
    ├── 03_n_error_origins.ipynb
    ├── 04_order_zero_direction.ipynb
    ├── 05_order_all_pairs.ipynb
    ├── 06_order_robustness.ipynb
    ├── 07_sequential_consistency.ipynb
    ├── 08_zero_vs_one.ipynb
    ├── 09_cognitive_predictors_cosmos.ipynb
    ├── 10_zero_rule_correlates_conmat.ipynb
    └── plots/                # figures generades (PDF + PNG)
```

## Decisions

Cada decisió analítica (font, any, filtres, tractament dels `Help`, mètrica de RT, mínims de mostra…)
queda escrita a `DECISIONS.txt`, numerada i datada. Si una decisió canvia, s'actualitza allà primer.

## Estil de les figures

Tots els notebooks importen `plot_style.py` i criden `set_style()`. Allà hi ha la paleta, les mides de
lletra i `paint_it_black()`. La font es resol en temps d'execució amb la primera família disponible de
`Helvetica → Arial → Nimbus Sans → TeX Gyre Heros → Liberation Sans → DejaVu Sans`: Helvetica no està
instal·lada a tot arreu i demanar-la directament fa que matplotlib avisi a cada text i caigui a DejaVu
Sans sense dir-ho. Si canvia un color o una mida, es canvia allà, no al notebook.

## Queries i cache

Cada pregunta de dades és un fitxer `.sql` dins `queries/`, mai una query escrita directament al notebook. S'executa amb `cached_query()`, que hasheja el text realment executat i desa el resultat a `cache_dir/<nom_sql>__<hash>.parquet`:

```python
from cached_query import cached_query

df = cached_query("queries/rt_error_by_fact_age.sql", cache_dir="cache")
```

Canviar la query genera un cache nou automàticament; `refresh=True` força tornar a consultar el warehouse. Les queries curtes de diagnòstic (`DESCRIBE TABLE`, comptatges de control) no es cachegen: criden `databricks_connector.query()` directament.


## Anàlisi

### A. Què passa amb els fets amb 0 (01–03)

- **`analysis/01_rt_vs_error_by_fact.ipynb`** — temps de resposta (mediana dels encerts) vs taxa d'error
  per cada multiplicació dirigida `a×b`, un panell per edat de curs (8–15), amb les operacions que
  contenen un 0 destacades. Una figura per `academic_year_id` (ES_2025 i ES_2024). Inclou una primera cel·la de cobertura
  per país (`queries/country_coverage.sql`).
  Query: `queries/rt_error_by_fact_age.sql`, amb l'any com a marcador `-- {YEAR}` que el notebook
  substitueix i passa a `cached_query(sql_text=...)`.
  La mateixa figura per grups de països (ES, MX, IT, US, EC-SI, EC-CO, CO, CL+BR — tots els anys agrupats) surt de
  `queries/rt_error_by_fact_age_country.sql`, amb el marcador `-- {COUNTRIES}`.

- **`analysis/02_error_types.ipynb`** — què responen els alumnes quan s'equivoquen en una operació amb un 0.
  Classifica cada error en quatre tipus (`= n` additiu, `= 1`, altre dígit, `≥ 10`), dibuixa l'histograma
  de respostes (operacions amb 0 vs la resta, un panell per edat) i reporta quin % dels errors és el
  típic `0×n = n` / `n×0 = n`, per edat, per direcció i per operació (figura amb IC 95% de Wilson,
  per edat i per operand no nul). El % principal és sobre TOTES les respostes al fet, no només sobre
  els errors. `0×0` queda fora dels %
  (la seva resposta correcta ja és 0) i es reporta a part.
  Query: `queries/answers_by_fact_age.sql`, una fila per (edat, operació, resultat, resposta),
  amb l'any com a marcador `-- {YEAR}`. Decisions #14-#17.

- **`analysis/03_n_error_origins.ipynb`** — d'on surt la resposta `0×n = n` (revisió de l'expert, novembre 2026).
  **H1 suma**: els alumnes que responen `n` no sumen més en altres multiplicacions un cop condicionat a l'error
  (`n+1` a les taules de l'1: 18,8% vs 18,5% dels errors; `a+b`: 3,1% vs 2,8%, ns). **H2 interferència de l'×1**: un
  fet amb 1 just abans puja el `= n` +0,5 pp sobre 8,5% (dins d'alumne i de test), només aquest error, no és
  perseveració, ni ho explica el nivell o la velocitat de l'alumne (#67); contribució petita. **H3 RT** (sense filtre de RT, perquè el signe en depenia): dins d'alumne `= n` és
  un 8% més ràpid que l'encert, només a 9-11 anys, i un 61% més lent que els altres errors (endevinalles, ~0,8 s). **H4 còpia de l'operand**: no; amb un error a un fet ×1, els alumnes `n` responen 1 *menys*
  (27% vs 35%), i el `= 1` sembla una intrusió de la regla del 0 en qui la té. **H5 creença o lapsus**: un 1,9% dels
  alumnes amb >= 4 fets amb 0 responen `n` gairebé sempre (86%) i fan el 19% de les respostes `n`; la resta és
  intermitent. Decisions #63-#69.
  Queries: `queries/n_error_zero_trials.sql`, `queries/n_error_student_facts.sql`.

### B. Asimetria d'ordre (04–06)

- **`analysis/04_order_zero_direction.ipynb`** — distribució del temps de resposta dels encerts a `0×n`
  comparada amb `n×0`, per edat de curs. Figura de dos panells: **A** violí partit (una parella per edat,
  eix y de 0 a 15 s — retall només visual; la variant `*_log` mostra tot el rang) i **B** la diferència dins de cada alumne (mediana de `n×0 − 0×n`
  amb IC 95%). El contrast de direcció és **aparellat per alumne** (Wilcoxon de rangs amb signe,
  mida de l'efecte rank-biserial, p corregits per Holm sobre les 7 edats, i t aparellada com a robustesa),
  perquè el mateix alumne respon les dues direccions al mateix test. El panell B porta les marques de
  significació; la variant `*_log` es desa però no es mostra al notebook.
  `0×0` queda fora de les dues sèries i es reporta a la taula de cobertura. Decisions #18-#21.
  Query: `queries/rt_zero_facts_correct.sql` (una fila per resposta), amb l'any com a marcador `-- {YEAR}`;
  la regla dels >= 30 s'aplica amb les cel·les de `queries/rt_error_by_fact_age.sql`.

- **`analysis/05_order_all_pairs.ipynb`** — la mateixa comparació d'ordre estesa als **45 parells**
  d'operands diferents: dins de cada parell, `petit×gran` contra `gran×petit`, test aparellat per alumne
  i p corregits per Holm sobre els 45. Figura de dos panells: **A** forest plot de la diferència mediana
  per parell (els que contenen un 0 destacats) i **B** la mida de l'efecte contra la dificultat del parell
  (taxa d'error, mesurada independentment del RT).
  Els parells es classifiquen en **regla-0** (els 9 amb un zero), **regla-1** (1×2…1×9) i **recuperació**
  (operands 2–9): `n×1 = n` és una regla igual que `n×0 = 0`.
  Resultat: l'asimetria d'ordre és un **efecte de regla**, no del zero ni de la dificultat. Les dues
  taules de regla es comporten igual (p = .321) i totes dues difereixen de les de recuperació (p < .001);
  dins dels parells de recuperació la dificultat no hi té cap relació (rho = +.03).
  **Replicat a ES_2024** (2,25M respostes): l'efecte de regla es manté (0-fets 9/9 parells, control de
  dificultat p < .001 als dos anys, rho = +.69 entre les mides d'efecte dels dos anys); el contrast
  sense controlar dificultat s'afebleix (p = .030) perquè el grup de recuperació es desplaça d'un any a
  l'altre mentre els de regla no. Decisions #22-#27.
  Query: `queries/rt_by_fact_correct.sql`, amb l'any com a marcador `-- {YEAR}`.

- **`analysis/06_order_robustness.ipynb`** — l'efecte d'ordre `n×0` − `0×n` atacat des de sis estimadors
  amb supòsits diferents, més la precisió com a segona variable. Models de trials amb efectes fixos
  d'alumne i de fet i errors clusteritzats per alumne (`pyfixest`), model mixt amb pendent aleatòria per
  alumne, delta per alumne, contrast aparellat alumne × operand, i remostreig d'un sol trial per alumne.
  Controls d'alumne (nivell i velocitat) mesurats **fora** dels problemes amb zero.
  Resultat: +32 a +40 ms (≈ +2% en log) a tots els estimadors, sense moderació per edat, nivell ni
  velocitat, i `n×0` també és **menys precís** (−2,4 punts, OR 0,73). Decisions #29-#33.
  Queries: `queries/rt_zero_facts_trials.sql` i `queries/student_controls.sql`.

### C. Consistència seqüencial (07)

- **`analysis/07_sequential_consistency.ipynb`** — dins d'un mateix alumne, encertar el primer
  problema `×0` prediu encertar el següent? Anàlisi **seqüencial i observacional, no causal**: la
  parella són els **dos primers assaigs `×0` vàlids** de cada alumne a cada edat, un parell per
  alumne (93.663 alumnes), amb els errors estàndard clusteritzats per **aula**.
  Brut: 93,3% contra 51,5% (+41,8 pp, OR 13,1). Amb competència general, velocitat, edat, RT del
  primer assaig, ordre i operand del segon i la distància controlats: OR 10,7 [10,0, 11,4]
  (93,8% contra 61,3%), i es manté dins de cada quintil de competència i de velocitat.
  No és repetició del fet: la cel·la **més feble** és justament el mateix fet repetit en el mateix
  ordre (OR ajustada 2,5) i l'associació es manté quan canvien **alhora** l'operand i l'ordre
  (`0×7 → 4×0`: OR 9,0, 43.511 alumnes). És **específica del zero**: correlació tetracòrica .70
  (`×0`) contra .42 (`×1`) i .41 (recuperació) sobre els mateixos 81.390 alumnes — la tetracòrica,
  i no l'OR, perquè els tres tipus tenen taxes base molt diferents.
  Cau amb la distància (.78 dins del mateix test, .43 en un test posterior, mediana 102 dies) però
  no desapareix, i l'A998 **no dona cap retroacció** després d'un error, de manera que el confusor
  habitual d'aquest disseny no hi és. Un efecte fix d'alumne **no és vàlid** amb un resultat
  retardat (biaix de Nickell: surt negatiu); això és al notebook, etiquetat, perquè no es torni a
  ajustar. Decisions #34-#41.
  Figures: `seq_zero_by_age`, `seq_zero_order_transfer`, `seq_zero_operand_transfer`,
  `seq_zero_four_conditions`, `seq_problem_type`, `seq_temporal_distance`.
  Queries: `queries/sequential_trials.sql` (cada assaig amb la seva posició dins la seqüència de
  l'alumne; marcadors `-- {YEAR}` i `-- {CAP}`), `queries/student_controls.sql` i
  `queries/student_classroom.sql`.

### D. Específic del zero? Control amb l'1 (08)

- **`analysis/08_zero_vs_one.ipynb`** — totes les inspeccions principals del 0 repetides amb els **fets amb 1**
  com a grup propi (revisió d'octubre de 2026): si un patró apareix amb el 0 i no amb l'1, no s'explica només
  per l'ús d'una regla. Seccions: dispersió RT–error amb tres colors (ES_2025, ES_2024), separació per
  **onada de recollida** (T1/T2/T3), tipus d'error (`0×n = n` contra `1×n = 1`), violí per ordre de `1×n` /
  `n×1`, model de trials amb efectes fixos amb 0 i 1 junts, consistència seqüencial per edat, i les
  comprovacions de dades (signes, `Help`, respostes mal registrades).
  Resultat: els fets amb 0 són tan ràpids com els d'1 però el **doble d'errors**; l'error "copiar l'altre
  operand" és 6 vegades més freqüent amb el 0; l'asimetria d'ordre en precisió és **només del 0** i en RT
  només ho és a partir dels 12 anys; la consistència seqüencial també és específica del 0. Decisions #42-#49.
  Queries noves: `rt_one_facts_trials.sql`, `rt_error_by_fact_age_wave.sql`, `check_operation_glyphs.sql`,
  `check_user_answer_quality.sql`.

### E. Qui fa servir la regla del 0 (09–10)

- **`analysis/09_cognitive_predictors_cosmos.ipynb`** — estudi longitudinal: els factors cognitius de **COSMOS 2023-24**
  (6 i 7 anys, PRE i POST) prediuen l'ús de la **regla del 0** (i de l'1) a l'A998 un o dos anys després,
  per sobre dels fets de recuperació? COSMOS 2023-24 es reconstrueix en format gold (percentils contra la
  norma 2024-25) i queda a la cache. La regla del 0 només es pot estudiar a la cohort de 7 anys als 9 anys
  (a 8 anys l'A998 no té fets amb 0).
  Resultat: el **raonament (matrius)** prediu la precisió amb el 0 controlant la recuperació (OR 1,35 per DE,
  PRE), la **memòria de treball** en la mateixa direcció i mida (1,28-1,38, no significativa després de Holm);
  cap factor prediu la regla de l'1. Efecte concentrat a Mèxic. Decisions #50-#56.
  Afegit (octubre 2026): **predictor combinat PRE+POST** (mateixa estimació, sense guany de potència), **biaix de selecció**
  (pesos IPW amb IC per bootstrap de classes i efectes fixos de classe: l'efecte no s'explica per la selecció), test formal
  **Mèxic vs resta** (no significatiu) i **contrast amb la cohort gran COSMOS 2024-25** a 8 anys: la regla de l'1 és un nul
  precís (OR 0,99-1,04, equivalent a negligible, ~21.800 alumnes) mentre la recuperació hi respon fort. Decisions #59-#62.
  Queries: `queries/cosmos_2324_gold.sql`, `queries/cosmos_gold_by_year.sql`, `queries/a998_cosmos_2324_trials.sql`,
  `queries/a998_cosmos_student_counts.sql`.

- **`analysis/10_zero_rule_correlates_conmat.ipynb`** — amb què va lligada la regla del 0 fora de l'A998? **ConMAT** del mateix
  curs (subescales de contingut i de procés), el test **socioemocional** (exploratori) i **COSMOS** al mateix model.
  Descobriment a ES_2024, confirmació a alumnes nous d'ES_2025.
  Resultat: la ConMAT total prediu la regla del 0 per sobre de la fluïdesa i la velocitat (β latent 0,11-0,13, Kraft mitjà,
  replicat) i gairebé no la de l'1; per sobre de la resta de la prova hi afegeixen poc totes les àrees (β 0,03-0,05),
  sobretot el **contingut conceptual**. WM i matrius comparteixen bona part del senyal amb la ConMAT. Decisions #71-#76.
  Queries: `zero_rule_student_outcomes.sql`, `conmat_student_subscores.sql`, `socioemotional_scores.sql`.
