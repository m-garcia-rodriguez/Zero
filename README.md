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
├── .env                      # credencials (no versionat)
├── requirements.txt
├── DECISIONS.txt             # totes les decisions analítiques preses
├── queries/                  # .sql, una pregunta de dades per fitxer
├── cache/                    # resultats cachejats en .parquet (no versionat)
└── analysis/
    ├── rt_error_scatter.ipynb
    ├── error_types.ipynb
    ├── rt_zero_direction.ipynb
    ├── rt_direction_by_pair.ipynb
    ├── rt_order_robustness.ipynb
    ├── sequential_zero_consistency.ipynb
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

- **`analysis/rt_error_scatter.ipynb`** — temps de resposta (mediana dels encerts) vs taxa d'error
  per cada multiplicació dirigida `a×b`, un panell per edat de curs (8–15), amb les operacions que
  contenen un 0 destacades. Una figura per `academic_year_id` (ES_2025 i ES_2024). Inclou una primera cel·la de cobertura
  per país (`queries/country_coverage.sql`).
  Query: `queries/rt_error_by_fact_age.sql`, amb l'any com a marcador `-- {YEAR}` que el notebook
  substitueix i passa a `cached_query(sql_text=...)`.
  La mateixa figura per grups de països (ES, MX, IT, US, EC-SI, EC-CO, CO, CL+BR — tots els anys agrupats) surt de
  `queries/rt_error_by_fact_age_country.sql`, amb el marcador `-- {COUNTRIES}`.

- **`analysis/error_types.ipynb`** — què responen els alumnes quan s'equivoquen en una operació amb un 0.
  Classifica cada error en quatre tipus (`= n` additiu, `= 1`, altre dígit, `≥ 10`), dibuixa l'histograma
  de respostes (operacions amb 0 vs la resta, un panell per edat) i reporta quin % dels errors és el
  típic `0×n = n` / `n×0 = n`, per edat, per direcció i per operació (figura amb IC 95% de Wilson,
  per edat i per operand no nul). El % principal és sobre TOTES les respostes al fet, no només sobre
  els errors. `0×0` queda fora dels %
  (la seva resposta correcta ja és 0) i es reporta a part.
  Query: `queries/answers_by_fact_age.sql`, una fila per (edat, operació, resultat, resposta),
  amb l'any com a marcador `-- {YEAR}`. Decisions #14-#17.

- **`analysis/rt_zero_direction.ipynb`** — distribució del temps de resposta dels encerts a `0×n`
  comparada amb `n×0`, per edat de curs. Figura de dos panells: **A** violí partit (una parella per edat,
  eix y de 0 a 15 s — retall només visual; la variant `*_log` mostra tot el rang) i **B** la diferència dins de cada alumne (mediana de `n×0 − 0×n`
  amb IC 95%). El contrast de direcció és **aparellat per alumne** (Wilcoxon de rangs amb signe,
  mida de l'efecte rank-biserial, p corregits per Holm sobre les 7 edats, i t aparellada com a robustesa),
  perquè el mateix alumne respon les dues direccions al mateix test. El panell B porta les marques de
  significació; la variant `*_log` es desa però no es mostra al notebook.
  `0×0` queda fora de les dues sèries i es reporta a la taula de cobertura. Decisions #18-#21.
  Query: `queries/rt_zero_facts_correct.sql` (una fila per resposta), amb l'any com a marcador `-- {YEAR}`;
  la regla dels >= 30 s'aplica amb les cel·les de `queries/rt_error_by_fact_age.sql`.

- **`analysis/rt_direction_by_pair.ipynb`** — la mateixa comparació d'ordre estesa als **45 parells**
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

- **`analysis/rt_order_robustness.ipynb`** — l'efecte d'ordre `n×0` − `0×n` atacat des de sis estimadors
  amb supòsits diferents, més la precisió com a segona variable. Models de trials amb efectes fixos
  d'alumne i de fet i errors clusteritzats per alumne (`pyfixest`), model mixt amb pendent aleatòria per
  alumne, delta per alumne, contrast aparellat alumne × operand, i remostreig d'un sol trial per alumne.
  Controls d'alumne (nivell i velocitat) mesurats **fora** dels problemes amb zero.
  Resultat: +32 a +40 ms (≈ +2% en log) a tots els estimadors, sense moderació per edat, nivell ni
  velocitat, i `n×0` també és **menys precís** (−2,4 punts, OR 0,73). Decisions #29-#33.
  Queries: `queries/rt_zero_facts_trials.sql` i `queries/student_controls.sql`.

- **`analysis/sequential_zero_consistency.ipynb`** — dins d'un mateix alumne, encertar el primer
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
