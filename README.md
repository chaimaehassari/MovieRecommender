# MovieRecommender

## Description
MovieRecommender est un projet Python de recommandation de films basé sur les évaluations des utilisateurs.

## Fonctionnalités
- Importation des évaluations (ratings-ingest)
- Création de la matrice utilisateurs-films (user-item-matrix)
- Calcul des similarités (similarity)
- Génération des recommandations (recommend)
- Évaluation du système (evaluation)

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