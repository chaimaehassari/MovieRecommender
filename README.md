# MovieRecommender

## Description
MovieRecommender est un projet Python de recommandation de films basé sur les évaluations des utilisateurs.

## Fonctionnalités
- Importation des évaluations (ratings-ingest)
- Création de la matrice utilisateurs-films (user-item-matrix)
- Calcul des similarités (similarity)
  - **Implémenté par Hiba :** Développement du calcul de similarité entre utilisateurs par Cosine Similarity, incluant la validation des entrées, la gestion des valeurs manquantes, la génération de la matrice de similarité et l'identification des utilisateurs les plus similaires.
  - 47 tests unitaires ajoutés et validés.
- Génération des recommandations (recommend)
- Évaluation du système (evaluation)

## Technologies
- Python
- Git et GitHub
- Pytest

## Structure du projet
- `src/` : code source
- `tests/` : tests unitaires
