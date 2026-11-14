# MovieRecommender

## Description
MovieRecommender est un projet Python de recommandation de films basé sur les évaluations des utilisateurs.

## Fonctionnalités
- Importation des évaluations (ratings-ingest)
- - Création de la matrice utilisateurs-films : génération de la User-Item Matrix à partir des évaluations, voir [docs/user-item-matrix.md](docs/user-item-matrix.md) (TOUKIL Safaa)
- Calcul des similarités (similarity)
- Génération des recommandations (recommend)
- Évaluation des recommandations : Precision@K et Recall@K (evaluation), voir [docs/evaluation.md](docs/evaluation.md) (SOUIDA Fahd)

## Technologies
- Python
- Git et GitHub
- Pytest

## Structure du projet
- `src/` : code source
- `tests/` : tests unitaires

## Ma contribution — `recommend`

J’ai développé `recommend_movies`, la recommandation User-Based Collaborative Filtering : sélection des voisins à similarité strictement positive, exclusion des films déjà notés et calcul d’une moyenne pondérée des notes.
La fonction valide les matrices d’entrée, gère les cas limites et retourne un DataFrame classé par score décroissant puis `movie_id` croissant.
Le module est couvert par `tests/test_recommend.py` avec 41 tests, sans modification des DataFrames reçus.