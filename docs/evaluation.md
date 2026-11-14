# Évaluation des recommandations

Module : `src/evaluation.py` · Tests : `tests/test_evaluation.py`

Ce module évalue une liste ordonnée de films recommandés par rapport à un ensemble
de films pertinents (vérité terrain).

## Fonctions

| Fonction | Formule |
|----------|---------|
| `precision_at_k(recommendations, relevant_movies, k)` | films pertinents dans le top K / K |
| `recall_at_k(recommendations, relevant_movies, k)` | films pertinents dans le top K / nombre de films pertinents |

## Format des entrées

- `recommendations` : DataFrame pandas avec les colonnes `movie_id` et `rank`
  (rangs numériques, positifs et uniques ; pas de `movie_id` dupliqué ou manquant).
- `relevant_movies` : ensemble ou liste d'identifiants de films.
- `k` : entier supérieur ou égal à 1.

Une `ValueError` est levée si une entrée est invalide.

## Exemple

```python
import pandas as pd
from src.evaluation import precision_at_k, recall_at_k

recs = pd.DataFrame({"movie_id": [10, 20, 30, 40, 50], "rank": [1, 2, 3, 4, 5]})
relevant = {20, 40, 99}

precision_at_k(recs, relevant, k=5)  # 2 / 5 = 0.4
recall_at_k(recs, relevant, k=5)     # 2 / 3 = 0.667
```

## Cas particuliers

- Moins de K recommandations : le dénominateur de Precision@K reste K.
- Liste de recommandations vide : les deux métriques valent 0.0.
- Vérité terrain vide : Recall@K vaut 0.0.

## Tests

```powershell
python -m pytest tests/test_evaluation.py -v
```