# ZERO

Estudi del comportament del número zero en tests de multiplicació: com responen els alumnes a les operacions que inclouen el 0, tant en precisió (accuracy) com en velocitat de resposta (RT).

## Contingut del repositori

```
Zero/
├── databricks_connector.py   # connexió OAuth M2M a Databricks (query() -> pd.DataFrame)
├── cached_query.py           # wrapper de query() amb cache basat en hash del .sql
├── .env                      # credencials (no versionat)
└── requirements.txt
```

## Queries i cache

Cada pregunta de dades és un fitxer `.sql` dins `queries/`, mai una query escrita directament al notebook. S'executa amb `cached_query()`, que hasheja el text realment executat i desa el resultat a `cache_dir/<nom_sql>__<hash>.parquet`:

```python
from cached_query import cached_query

df = cached_query("queries/zero_accuracy_by_age.sql", cache_dir="cache")
```

Canviar la query genera un cache nou automàticament; `refresh=True` força tornar a consultar el warehouse. Les queries curtes de diagnòstic (`DESCRIBE TABLE`, comptatges de control) no es cachegen: criden `databricks_connector.query()` directament.
