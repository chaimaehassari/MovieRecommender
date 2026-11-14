# Feature 3 — Similarity Calculation (User-Based Collaborative Filtering)

**Projet :** MovieRecommender  
**Fonctionnalité :** Feature 3 — Similarity Calculation  
**Responsable :** Hiba  
**Module :** `src/similarity.py`  
**Tests :** `tests/test_similarity.py`  
**Dépendances :** `pandas`, `numpy`, `scikit-learn`, `pytest`

## 1. Introduction

### 1.1 Présentation de la fonctionnalité

Similarity Calculation est la fonctionnalité qui calcule la similarité entre les utilisateurs d'un système de recommandation de films. Elle reçoit une matrice utilisateurs-films et produit une matrice de similarité utilisateurs-utilisateurs.

Cette fonctionnalité constitue une étape intermédiaire du système MovieRecommender :

1. les évaluations sont transformées en matrice utilisateurs-films par la fonctionnalité User-Item Matrix ;
2. Similarity Calculation mesure la proximité entre les utilisateurs à partir de cette matrice ;
3. la fonctionnalité Recommend exploite ces similarités pour identifier les voisins d'un utilisateur et produire des recommandations.

### 1.2 Objectif dans le User-Based Collaborative Filtering

Le User-Based Collaborative Filtering repose sur l'hypothèse que deux utilisateurs ayant attribué des notes similaires aux mêmes films ont des goûts proches. Les films appréciés par l'un ont donc une probabilité plus élevée d'être appréciés par l'autre.

La fonctionnalité répond à trois besoins :

- représenter chaque utilisateur par un vecteur de notes sur les films ;
- comparer ces vecteurs deux à deux ;
- fournir, pour un utilisateur donné, la liste de ses voisins les plus similaires.

### 1.3 Périmètre et responsabilités

Ce document décrit uniquement la fonctionnalité Similarity Calculation.

| Élément | Responsable | Rôle |
|---|---|---|
| User-Item Matrix | Safaa | Construit la matrice utilisateurs-films à partir des ratings |
| Similarity Calculation | Hiba | Calcule les similarités entre utilisateurs |
| Recommend | Chaimae | Exploite les similarités pour générer les recommandations |

Les fonctions `create_user_item_matrix` et les fonctions de Recommend ne sont pas documentées comme faisant partie de cette implémentation. Elles sont citées uniquement pour expliquer les interfaces d'entrée et de sortie.

## 2. Input Data

### 2.1 Structure de la `user_item_matrix`

La fonction `calculate_user_similarity` reçoit un `pandas.DataFrame` nommé `user_item_matrix` :

- **lignes** : utilisateurs, identifiés par `userId` ;
- **colonnes** : films, identifiés par `movieId` ;
- **valeurs** : notes numériques (`rating`) ;
- **valeurs manquantes** : `NaN`.

Exemple avec le jeu de données `data/sample/user_item_sample.csv` :

| user_id | M001 | M002 | M003 | M004 | M005 | M006 |
|---|---:|---:|---:|---:|---:|---:|
| U001 | 5 | 4 | 5 | 3 | 1 | 2 |
| U002 | 4 | 5 | 4 | 3 | 2 | 1 |
| U003 | NaN | NaN | NaN | NaN | 5 | 5 |
| U004 | 5 | 4 | 4 | 2 | 1 | NaN |
| U005 | NaN | NaN | 5 | 5 | NaN | NaN |
| U006 | NaN | NaN | NaN | NaN | NaN | NaN |

Ce fichier contient une matrice `6 × 6`. Il est utilisé par le test de compatibilité `test_sample_dataset_is_compatible_with_similarity_calculation`.

### 2.2 Contrat de données

La matrice doit respecter les conditions suivantes :

- être un `pandas.DataFrame` ;
- ne pas être vide ;
- avoir des identifiants utilisateurs uniques dans l'index ;
- avoir des identifiants de films uniques dans les colonnes ;
- contenir uniquement des colonnes numériques ;
- contenir des valeurs finies après traitement des `NaN`.

Le module documente ce contrat comme **provisoire** dans son docstring. Le cas nominal correspond toutefois à la sortie de la fonctionnalité User-Item Matrix.

### 2.3 Gestion des valeurs manquantes

Les `NaN` représentent les films qu'un utilisateur n'a pas notés. Ils sont remplacés par `0.0` **sur une copie** de la matrice avant le calcul de similarité.

Cette opération interne :

- ne modifie pas la matrice d'origine ;
- permet d'obtenir une matrice numérique complète pour `scikit-learn` ;
- traite un film non noté comme une absence de contribution dans le produit scalaire.

Le test `test_missing_ratings_are_treated_as_zero` vérifie qu'une matrice contenant des `NaN` produit le même résultat qu'une matrice où ces valeurs ont été remplacées par `0`.

Le test `test_original_input_dataframe_is_not_modified` vérifie que la matrice d'entrée conserve ses `NaN` après l'appel.

### 2.4 Compatibilité avec la fonctionnalité User-Item Matrix

La fonctionnalité User-Item Matrix, développée par Safaa, produit la matrice attendue à partir d'un DataFrame de ratings contenant les colonnes `userId`, `movieId` et `rating`.

La fonction fournie utilise `pivot` :

```python
matrix = ratings_df.pivot(
    index="userId",
    columns="movieId",
    values="rating",
)
```

Le résultat est directement compatible avec `calculate_user_similarity` :

| Point | Sortie de User-Item Matrix | Exigence de Similarity Calculation | Compatibilité |
|---|---|---|---|
| Type | `pandas.DataFrame` | `pandas.DataFrame` | Oui |
| Index | `userId` | identifiants utilisateurs uniques | Oui |
| Colonnes | `movieId` | identifiants de films uniques | Oui |
| Valeurs | ratings numériques | colonnes numériques | Oui si les ratings sont numériques |
| Valeurs manquantes | `NaN` | acceptées et remplacées par `0` | Oui |

Conditions à respecter en amont :

- chaque paire `(userId, movieId)` doit être unique, sinon `pivot` échoue avec une erreur de type `Index contains duplicate entries` ;
- la colonne `rating` doit être numérique ;
- le DataFrame de ratings ne doit pas être vide ;
- les identifiants `userId` et `movieId` ne doivent pas être manquants.

`pivot` ne contient que les utilisateurs et les films réellement présents dans les ratings. Si Recommend doit couvrir un catalogue complet de films, une réindexation devra être décidée par les fonctionnalités User-Item Matrix ou Recommend.

## 3. Méthode utilisée

### 3.1 Principe de la Cosine Similarity

Chaque utilisateur est représenté par un vecteur de notes :

\[
\mathbf{r}_u = (r_{u,1}, r_{u,2}, \ldots, r_{u,n})
\]

où \(n\) est le nombre de films et \(r_{u,i}\) est la note de l'utilisateur \(u\) pour le film \(i\). Si l'utilisateur n'a pas noté le film, la valeur utilisée est `0` après le traitement des `NaN`.

La similarité entre deux utilisateurs \(u\) et \(v\) est calculée par la similarité cosinus entre leurs vecteurs :

\[
\operatorname{sim}(u,v)
= \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{\lVert \mathbf{r}_u \rVert_2 \, \lVert \mathbf{r}_v \rVert_2}
= \frac{\sum_{i=1}^{n} r_{u,i}\,r_{v,i}}
{\sqrt{\sum_{i=1}^{n} r_{u,i}^2}\sqrt{\sum_{i=1}^{n} r_{v,i}^2}}
\]

Les composantes sont :

- \(\mathbf{r}_u \cdot \mathbf{r}_v\) : produit scalaire des deux vecteurs de notes ;
- \(\lVert \mathbf{r}_u \rVert_2\) : norme euclidienne du vecteur de l'utilisateur \(u\) ;
- \(\lVert \mathbf{r}_v \rVert_2\) : norme euclidienne du vecteur de l'utilisateur \(v\) ;
- le dénominateur normalise le score pour qu'il dépende de l'orientation des vecteurs et non de leur longueur.

### 3.2 Interprétation des scores

La similarité cosinus est bornée entre `-1` et `1` :

- `1` indique des vecteurs de même direction, donc des profils très proches ;
- `0` indique des vecteurs orthogonaux ou sans contribution commune ;
- une valeur négative indique des directions opposées.

Avec des ratings non négatifs, les scores obtenus sont compris entre `0` et `1`.

L'implémentation utilise `sklearn.metrics.pairwise.cosine_similarity` puis borne les résultats avec `numpy.clip` dans l'intervalle `[-1, 1]`.

### 3.3 Pourquoi cette méthode

La similarité cosinus est adaptée au User-Based Collaborative Filtering pour plusieurs raisons :

- elle compare la **direction** des profils de notes plutôt que leur échelle absolue ;
- elle est simple à calculer et efficace sur une matrice utilisateurs-films ;
- elle produit un score interprétable et borné ;
- elle est disponible directement dans `scikit-learn`.

Cette implémentation utilise les ratings bruts, sans centrage par utilisateur. Les `NaN` étant remplacés par `0`, cette convention doit être prise en compte dans l'interprétation des similarités.

### 3.4 Cas particuliers

- **Profil vide** : le vecteur de l'utilisateur est nul. L'implémentation produit une similarité de `0` avec tous les utilisateurs, y compris avec lui-même.
- **Aucun film commun** : le produit scalaire est nul, donc la similarité est `0`.
- **Utilisateur actif avec lui-même** : la similarité vaut `1`.
- **Utilisateurs avec peu de films communs** : la similarité peut être élevée artificiellement ; ce point est une limite méthodologique de la similarité cosinus avec des valeurs manquantes remplacées par `0`.

## 4. Implementation

### 4.1 Vue d'ensemble

Le module `src/similarity.py` contient trois fonctions :

| Fonction | Visibilité | Rôle | Retour |
|---|---|---|---|
| `_validate_user_item_matrix` | privée | Valider et normaliser la matrice d'entrée | `pandas.DataFrame` |
| `calculate_user_similarity` | publique | Calculer la matrice de similarité utilisateurs-utilisateurs | `pandas.DataFrame` |
| `get_most_similar_users` | publique | Retourner les voisins les plus similaires d'un utilisateur | `pandas.Series` |

### 4.2 `_validate_user_item_matrix`

```python
_validate_user_item_matrix(user_item_matrix: pd.DataFrame) -> pd.DataFrame
```

Cette fonction privée est appelée par `calculate_user_similarity`. Elle effectue les opérations suivantes :

1. vérifier que l'entrée est un `pandas.DataFrame` ;
2. vérifier que la matrice n'est pas vide ;
3. vérifier l'unicité des identifiants utilisateurs dans l'index ;
4. vérifier l'unicité des identifiants de films dans les colonnes ;
5. vérifier que toutes les colonnes ont un dtype numérique ;
6. remplacer les `NaN` par `0.0` et convertir les valeurs en `float` ;
7. vérifier que toutes les valeurs sont finies ;
8. retourner une copie normalisée.

La fonction ne modifie pas la matrice reçue.

### 4.3 `calculate_user_similarity`

```python
calculate_user_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame
```

**Paramètre :**

- `user_item_matrix` : `pandas.DataFrame` avec les utilisateurs en lignes, les films en colonnes et les notes en valeurs.

**Traitement :**

1. validation et normalisation par `_validate_user_item_matrix` ;
2. conversion de la matrice en tableau NumPy de type `float` ;
3. calcul de la similarité cosinus entre toutes les paires d'utilisateurs ;
4. bornage des scores entre `-1` et `1` ;
5. construction d'un `DataFrame` avec les identifiants utilisateurs en index et en colonnes.

**Retour :**

- un `pandas.DataFrame` carré ;
- index et colonnes contenant les identifiants utilisateurs, dans le même ordre que la matrice d'entrée ;
- valeurs numériques représentant les similarités.

### 4.4 `get_most_similar_users`

```python
get_most_similar_users(
    similarity_matrix: pd.DataFrame,
    user_id: Hashable,
    top_n: int = 5,
) -> pd.Series
```

**Paramètres :**

- `similarity_matrix` : matrice carrée produite par `calculate_user_similarity` ;
- `user_id` : identifiant de l'utilisateur cible ;
- `top_n` : nombre maximal de voisins à retourner, valeur par défaut `5`.

**Traitement :**

1. vérification du type et de la structure de la matrice ;
2. vérification de `top_n` ;
3. vérification de l'unicité, de la forme carrée et de la cohérence index/colonnes ;
4. vérification du caractère numérique, fini et borné des scores ;
5. vérification de la présence de `user_id` dans l'index et les colonnes ;
6. extraction de la ligne de l'utilisateur ;
7. suppression de l'utilisateur lui-même ;
8. tri des scores par ordre décroissant ;
9. limitation au nombre `top_n` ;
10. renommage de la Series en `"similarity"`.

**Retour :**

- une `pandas.Series` indexée par identifiant utilisateur ;
- nommée `"similarity"` ;
- triée par score décroissant ;
- ne contenant pas l'utilisateur demandé ;
- contenant au plus `top_n` utilisateurs.

Si `top_n` dépasse le nombre d'utilisateurs disponibles, la fonction retourne tous les voisins disponibles. Si un seul utilisateur est présent dans la matrice, la Series retournée est vide.

### 4.5 Validation des données

Les validations implémentées sont les suivantes.

Pour `calculate_user_similarity` :

| Condition | Exception |
|---|---|
| Entrée non `DataFrame` | `TypeError` |
| Matrice vide | `ValueError` |
| Identifiants utilisateurs dupliqués | `ValueError` |
| Identifiants de films dupliqués | `ValueError` |
| Colonnes non numériques | `TypeError` |
| Valeurs non finies après remplacement des `NaN` | `ValueError` |

Pour `get_most_similar_users` :

| Condition | Exception |
|---|---|
| `similarity_matrix` non `DataFrame` | `TypeError` |
| `similarity_matrix` vide | `ValueError` |
| `top_n` booléen ou non entier | `TypeError` |
| `top_n < 1` | `ValueError` |
| Identifiants utilisateurs dupliqués | `ValueError` |
| Identifiants de films dupliqués | `ValueError` |
| Matrice non carrée | `ValueError` |
| Index et colonnes différents ou dans un ordre différent | `ValueError` |
| Scores non numériques | `TypeError` |
| Scores manquants | `ValueError` |
| Scores non finis | `ValueError` |
| Scores hors de `[-1, 1]` | `ValueError` |
| `user_id` absent de l'index | `KeyError` |
| `user_id` absent des colonnes | `ValueError` |

`top_n` accepte les entiers Python et les entiers NumPy (`np.int32`, `np.int64`). Les booléens, les flottants, les chaînes et `None` sont rejetés.

### 4.6 Profils vides et absence de films communs

- Un utilisateur dont toutes les valeurs sont `NaN` devient un vecteur nul après normalisation.
- Sa similarité avec tous les utilisateurs, y compris lui-même, vaut `0`.
- Deux utilisateurs sans film commun ont un produit scalaire nul, donc une similarité de `0`.
- Un utilisateur actif a une similarité de `1` avec lui-même.

Ces comportements sont couverts par les tests `test_empty_user_profile_is_supported`, `test_empty_user_profile_self_similarity_is_zero`, `test_users_without_common_rated_movies_have_zero_similarity` et `test_self_similarity_is_one_for_active_users`.

### 4.7 Conservation des identifiants utilisateurs

`calculate_user_similarity` construit la matrice de sortie avec :

```python
index=matrix.index.copy()
columns=matrix.index.copy()
```

Les identifiants utilisateurs sont donc conservés à l'identique dans l'index et les colonnes, y compris leur dtype et leur nom. Les colonnes de films de la matrice d'entrée ne sont pas conservées dans la sortie : la matrice de similarité est une matrice utilisateurs-utilisateurs.

Le test `test_user_ids_are_preserved_as_index_and_columns` vérifie que les identifiants et le nom de l'index sont conservés sur les deux axes.

## 5. Output

### 5.1 Structure de la `similarity_matrix`

`calculate_user_similarity` retourne un `pandas.DataFrame` carré :

- **lignes** : utilisateurs (`userId`) ;
- **colonnes** : utilisateurs (`userId`) ;
- **valeurs** : scores de similarité entre `-1` et `1`, et entre `0` et `1` pour des ratings non négatifs.

La matrice est symétrique : `similarity.loc[u, v] == similarity.loc[v, u]`.

La diagonale contient la similarité de chaque utilisateur avec lui-même :

- `1` pour un utilisateur actif ;
- `0` pour un profil vide.

### 5.2 Exemple de sortie

L'exemple suivant est issu du test `test_known_cosine_similarity_example`.

Matrice d'entrée :

| user | M1 | M2 | M3 |
|---|---:|---:|---:|
| u1 | 1.0 | 2.0 | 0.0 |
| u2 | 2.0 | 1.0 | 0.0 |

Matrice de similarité obtenue :

| | u1 | u2 |
|---|---:|---:|
| u1 | 1.0 | 0.8 |
| u2 | 0.8 | 1.0 |

Le test vérifie que `similarity.loc["u1", "u2"]` vaut `0.8` à `pytest.approx` près.

### 5.3 Interprétation des scores

- Un score proche de `1` indique des utilisateurs ayant des profils de notes très proches.
- Un score proche de `0` indique une absence de contribution commune ou des profils orthogonaux.
- Un score négatif indique des directions opposées ; ce cas n'est pas attendu avec des ratings non négatifs.
- La similarité cosinus compare l'orientation des vecteurs, pas la magnitude absolue des notes.

### 5.4 Identification des utilisateurs les plus similaires

`get_most_similar_users` transforme la matrice de similarité en une Series de voisins :

- index : identifiants utilisateurs ;
- valeurs : scores de similarité ;
- nom de la Series : `"similarity"` ;
- tri décroissant ;
- exclusion de l'utilisateur demandé ;
- limite au paramètre `top_n`.

La fonction ne filtre pas les scores nuls. Si Recommend souhaite ignorer les voisins sans similarité positive, il doit appliquer un filtre supplémentaire, par exemple `neighbors[neighbors > 0]`.

## 6. Tests et validation

### 6.1 Exécution

Les tests unitaires se trouvent dans `tests/test_similarity.py`. Les **47 tests ont été exécutés avec succès** avec la commande :

```bash
python -m pytest tests/test_similarity.py -v
```

Résultat confirmé : `47 passed`.

### 6.2 Cas normaux

Les tests couvrent notamment :

- utilisateurs aux goûts similaires avec une similarité élevée ;
- utilisateurs aux goûts différents avec une similarité faible ;
- exemple de similarité cosinus connu, avec un score attendu de `0.8` ;
- similarité de soi-même égale à `1` pour les utilisateurs actifs ;
- symétrie de la matrice de similarité ;
- scores compris entre `-1` et `1` et valeurs finies.

### 6.3 Cas limites

Les tests couvrent également :

- remplacement des `NaN` par `0` ;
- non-modification de la matrice d'entrée ;
- profil vide avec similarité `0` ;
- similarité de soi-même égale à `0` pour un profil vide ;
- absence de films communs avec similarité `0` ;
- `top_n` supérieur au nombre d'utilisateurs disponibles ;
- matrice avec un seul utilisateur, retournant une Series vide ;
- scores aux bornes `-1.0` et `1.0` ;
- `top_n` de type `np.int32` et `np.int64` accepté ;
- conservation des identifiants utilisateurs dans l'index, les colonnes et les noms.

### 6.4 Validation des erreurs

Les tests vérifient le rejet des cas suivants :

- entrée qui n'est pas un `DataFrame` ;
- matrice vide ;
- ratings non numériques ;
- identifiants utilisateurs dupliqués ;
- identifiants de films dupliqués ;
- ratings infinis ;
- scores de similarité non numériques ;
- matrice de similarité non carrée ;
- index et colonnes contenant des identifiants différents ou dans un ordre différent ;
- scores de similarité manquants ;
- scores de similarité infinis ;
- scores de similarité hors de `[-1, 1]` ;
- utilisateur absent de la matrice ;
- `top_n` nul, négatif, booléen, non entier ou `None`.

### 6.5 Intégration interne et jeu de données d'exemple

Le test `test_calculate_and_get_most_similar_users_work_together` vérifie l'enchaînement :

```text
calculate_user_similarity → get_most_similar_users
```

Le test `test_sample_dataset_is_compatible_with_similarity_calculation` vérifie :

- la forme `6 × 6` du fichier `data/sample/user_item_sample.csv` ;
- la présence des utilisateurs `U001` à `U006` ;
- `U001`/`U002` supérieure à `0.9` ;
- `U003`/`U005` égale à `0` ;
- `U006`/`U001` égale à `0`.

Ces tests couvrent l'intégration interne des fonctions de Similarity Calculation. Ils ne testent pas directement la fonction `create_user_item_matrix` de Safaa ni les fonctions de Recommend de Chaimae.

## 7. Exemple d'utilisation

### 7.1 Enchaînement avec la fonctionnalité User-Item Matrix

L'exemple suivant suppose que la fonction `create_user_item_matrix` de Safaa est exposée dans `src/user_item_matrix.py`. Le chemin d'import peut être adapté à l'emplacement final du module.

```python
import pandas as pd

from src.user_item_matrix import create_user_item_matrix
from src.similarity import calculate_user_similarity, get_most_similar_users

# Données de ratings au format long
ratings_df = pd.DataFrame({
    "userId":  [1, 1, 1, 2, 2, 2, 3, 3],
    "movieId": [101, 102, 103, 101, 102, 103, 101, 103],
    "rating":  [5, 3, 4, 4, 2, 5, 1, 1],
})

# Feature 2 — construction de la matrice utilisateurs-films
user_item_matrix = create_user_item_matrix(ratings_df)

# Feature 3 — calcul de la matrice de similarité
similarity_matrix = calculate_user_similarity(user_item_matrix)

# Feature 3 — récupération des voisins de l'utilisateur 1
neighbors = get_most_similar_users(
    similarity_matrix,
    user_id=1,
    top_n=2,
)

print(similarity_matrix)
print(neighbors)
```

La matrice `user_item_matrix` obtenue est de forme `3 × 3` :

| userId | 101 | 102 | 103 |
|---|---:|---:|---:|
| 1 | 5 | 3 | 4 |
| 2 | 4 | 2 | 5 |
| 3 | 1 | NaN | 1 |

La similarité de l'utilisateur 1 avec l'utilisateur 2 est d'environ `0.97`, et avec l'utilisateur 3 d'environ `0.90`. La Series `neighbors` contient donc les utilisateurs `2` et `3`, triés par similarité décroissante.

### 7.2 Utilisation avec le jeu de données d'exemple

Le fichier `data/sample/user_item_sample.csv` contient déjà une matrice utilisateurs-films :

```python
import pandas as pd

from src.similarity import calculate_user_similarity, get_most_similar_users

user_item_matrix = pd.read_csv(
    "data/sample/user_item_sample.csv",
    index_col=0,
)

similarity_matrix = calculate_user_similarity(user_item_matrix)

neighbors = get_most_similar_users(
    similarity_matrix,
    user_id="U001",
    top_n=3,
)

print(similarity_matrix)
print(neighbors)
```

La matrice de similarité obtenue est de forme `6 × 6`, avec les utilisateurs `U001` à `U006` en index et en colonnes. La Series `neighbors` contient au plus trois utilisateurs, triés par similarité décroissante et sans `U001`.

## 8. Intégration dans le projet

### 8.1 Chaîne de traitement

```text
ratings_df
    │
    ▼
[Feature 2 — User-Item Matrix]  (Safaa)
create_user_item_matrix(ratings_df)
    │
    ▼
user_item_matrix : utilisateurs × films
    │
    ▼
[Feature 3 — Similarity Calculation]  (Hiba)
calculate_user_similarity(user_item_matrix)
    │
    ▼
similarity_matrix : utilisateurs × utilisateurs
    │
    ├──> get_most_similar_users(similarity_matrix, user_id, top_n)
    │
    ▼
[Feature 4 — Recommend]  (Chaimae)
```

### 8.2 Entrée depuis la fonctionnalité User-Item Matrix

La fonctionnalité de Safaa fournit la `user_item_matrix` attendue :

- `userId` en index ;
- `movieId` en colonnes ;
- ratings numériques en valeurs ;
- `NaN` pour les films non notés.

`calculate_user_similarity` peut recevoir cette matrice directement, sans adaptateur, si les conditions du contrat sont respectées.

### 8.3 Sortie vers la fonctionnalité Recommend

La fonctionnalité Recommend peut exploiter :

- `similarity_matrix` pour identifier les voisins d'un utilisateur ;
- `get_most_similar_users` pour obtenir les voisins triés ;
- `user_item_matrix` pour récupérer les notes et les films.

Points d'intégration importants :

- la matrice de similarité contient des identifiants utilisateurs sur les deux axes, pas des identifiants de films ;
- Recommend doit utiliser `user_item_matrix` pour accéder aux films et aux notes ;
- les `NaN` de `user_item_matrix` doivent être conservés pour distinguer les films non notés ;
- les scores de similarité nuls peuvent être filtrés par Recommend si nécessaire ;
- le cas sans voisin disponible doit être géré par Recommend.

L'interface exacte de la fonctionnalité Recommend n'est pas documentée ici : elle appartient à Chaimae et doit être confirmée avec elle lors de l'intégration.

## 9. Conclusion

La fonctionnalité Similarity Calculation implémente le calcul de similarité entre utilisateurs pour le User-Based Collaborative Filtering. Elle :

- reçoit une matrice utilisateurs-films ;
- valide et normalise les données ;
- remplace les `NaN` par `0` sans modifier l'entrée ;
- calcule les similarités avec Cosine Similarity ;
- retourne une matrice utilisateurs-utilisateurs ;
- permet d'identifier les voisins les plus similaires d'un utilisateur.

L'implémentation est validée par **47 tests unitaires réussis**, couvrant les cas normaux, les cas limites et la gestion des erreurs.

Cette fonctionnalité constitue la base de l'étape de recommandation : elle transforme les notes brutes en relations de proximité entre utilisateurs, que Recommend peut ensuite exploiter pour prédire des notes ou proposer des films. Elle ne construit pas la matrice utilisateurs-films et ne génère pas les recommandations finales ; ces responsabilités restent respectivement celles de la fonctionnalité User-Item Matrix et de la fonctionnalité Recommend.
