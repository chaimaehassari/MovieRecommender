# Création de la matrice utilisateurs-films

Module : `src/user_item_matrix.py` · Tests : `tests/test_user_item_matrix.py`

Ce module permet de transformer un DataFrame contenant les évaluations des utilisateurs en une matrice utilisateurs-films (User-Item Matrix).

## Fonction

| **Fonction** | **Description** |
|---|---|
| `create_user_item_matrix(ratings_df)` | Transforme les évaluations en une matrice où les lignes représentent les utilisateurs et les colonnes représentent les films. |

## Format des entrées

- `ratings_df` : DataFrame pandas contenant les colonnes obligatoires :
  - `userId` : identifiant de l'utilisateur.
  - `movieId` : identifiant du film.
  - `rating` : note attribuée au film.

Une `ValueError` est levée si une ou plusieurs colonnes obligatoires sont absentes.

## Format de sortie

La fonction retourne un DataFrame pandas :

- Index : identifiants des utilisateurs (`userId`).
- Colonnes : identifiants des films (`movieId`).
- Valeurs : notes attribuées par les utilisateurs (`rating`).
- Valeurs manquantes : représentées par `NaN` lorsqu'un utilisateur n'a pas évalué un film.

## Exemple

```python
import pandas as pd
from src.user_item_matrix import create_user_item_matrix

ratings = pd.DataFrame({
    "userId": [1, 1, 2, 2],
    "movieId": [10, 20, 10, 30],
    "rating": [4.0, 5.0, 3.0, 4.5]
})

matrix = create_user_item_matrix(ratings)

print(matrix)
```

Résultat :

```text
movieId  10   20   30
userId
1        4.0  5.0  NaN
2        3.0  NaN  4.5
```

## Tests

Pour exécuter le test unitaire :

```bash
python3 -m pytest tests/test_user_item_matrix.py -v
```