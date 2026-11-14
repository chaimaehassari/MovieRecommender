# Fonctionnalité `recommend`

## Objectif

La fonctionnalité `recommend` génère des recommandations de films pour un utilisateur à partir des notes des autres utilisateurs, selon une approche **User-Based Collaborative Filtering**.

## Fichiers concernés

- `src/recommend.py` : implémentation de la fonctionnalité.
- `tests/test_recommend.py` : tests unitaires.

## Interface

```python
recommend_movies(
    user_id,
    ratings_matrix,
    similarity_matrix,
    n=10,
)
```

### Entrées

- `user_id` : utilisateur cible, obligatoirement présent dans `ratings_matrix`.
- `ratings_matrix` : DataFrame utilisateurs × films, avec des notes numériques entre 1 et 5 ou `NaN`.
- `similarity_matrix` : DataFrame utilisateurs × utilisateurs, carré, symétrique, de diagonale 1 et de valeurs dans `[-1, 1]`.
- `n` : nombre maximum de recommandations, entier strictement positif.

### Sortie

Un DataFrame avec les colonnes :

- `movie_id` : identifiant du film recommandé ;
- `score` : score de recommandation ;
- `rank` : rang à partir de 1.

## Algorithme

1. Validation des entrées.
2. Récupération des films déjà notés par l’utilisateur cible.
3. Sélection des voisins ayant une similarité **strictement positive**.
4. Construction des films candidats à partir des voisins positifs, en excluant les films déjà notés par la cible.
5. Calcul du score de chaque film avec une moyenne pondérée des notes brutes :
   `score = Σ(similarité × note) / Σ(similarité)`.
6. Tri par score décroissant, puis par `movie_id` croissant en cas d’égalité.
7. Limitation à `n` résultats et ajout du rang.

## Validation et cas limites

La fonction vérifie notamment :

- le type DataFrame des matrices ;
- l’unicité des identifiants utilisateurs et films ;
- le caractère numérique des valeurs ;
- la plage des notes : 1 à 5 ou `NaN` ;
- la forme carrée, la symétrie, la diagonale et la plage de la matrice de similarité ;
- la présence de tous les utilisateurs notés dans la matrice de similarité ;
- la validité de `n` ;
- l’existence de l’utilisateur cible.

Les erreurs de validation lèvent une `ValueError`. Les cas sans notes, sans voisin positif, sans candidat ou avec tous les films déjà notés retournent un DataFrame vide avec les colonnes attendues.

## Tests

La suite `tests/test_recommend.py` contient **41 tests** couvrant notamment :

- la formule de moyenne pondérée ;
- la prise en compte des seules similarités positives ;
- l’exclusion des films déjà notés ;
- le dénominateur limité aux voisins ayant noté le film ;
- le classement et les égalités ;
- la limite `n` ;
- les erreurs de validation ;
- les cas limites ;
- la non-modification des DataFrames d’entrée.

Commande de vérification :

```powershell
py -m pytest tests/test_recommend.py -q
```

## Points d’intégration à aligner

- Décider si les similarités non définies sont représentées par `0` ou `NaN`.
- Vérifier que `user-item-matrix` utilise bien `NaN` pour les notes absentes, et non `0`.
- Vérifier que `similarity` fournit tous les utilisateurs ayant au moins une note, même en cas de Top-K.
- Homogénéiser les types des identifiants utilisateurs et films.
- Prévoir le comportement de `evaluation` face à une sortie vide, dont les dtypes sont actuellement `object`.
