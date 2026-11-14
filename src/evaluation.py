"""Evaluation metrics for a user-based movie recommendation system.

The functions in this module evaluate a ranked recommendation list against
a set of relevant movie IDs (ground truth).
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Iterable

import pandas as pd


def _validate_k(k: int) -> None:
    """Ensure k is a positive integer and not a boolean."""
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be an integer greater than or equal to 1.")


def _validate_recommendations(recommendations: pd.DataFrame) -> None:
    """Validate the recommendation DataFrame structure."""
    if not isinstance(recommendations, pd.DataFrame):
        raise ValueError("recommendations must be a pandas DataFrame.")

    required_columns = {"movie_id", "rank"}
    missing_columns = required_columns.difference(recommendations.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"recommendations is missing required column(s): {missing}.")

    if recommendations["movie_id"].isna().any():
        raise ValueError("recommendations must not contain missing movie IDs.")

    if recommendations["movie_id"].duplicated().any():
        raise ValueError("recommendations must not contain duplicate movie IDs.")

    ranks = recommendations["rank"]
    if ranks.isna().any():
        raise ValueError("recommendation ranks must not contain missing values.")

    if not pd.api.types.is_numeric_dtype(ranks):
        raise ValueError("recommendation ranks must be numeric.")

    if not ranks.is_unique:
        raise ValueError("recommendation ranks must be unique.")

    if (ranks <= 0).any():
        raise ValueError("recommendation ranks must be positive.")


def _normalize_relevant_movies(relevant_movies: Iterable[Any]) -> set[Any]:
    """Convert ground-truth movie IDs to a set and validate the input."""
    if relevant_movies is None:
        raise ValueError("relevant_movies must not be None.")

    try:
        relevant = set(relevant_movies)
    except TypeError as exc:
        raise ValueError("relevant_movies must be an iterable of movie IDs.") from exc

    if any(pd.isna(movie_id) for movie_id in relevant):
        raise ValueError("relevant_movies must not contain missing movie IDs.")

    return relevant


def _get_top_k(
    recommendations: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Return at most k recommendations in ascending rank order."""
    _validate_k(k)
    _validate_recommendations(recommendations)

    return recommendations.sort_values(
        by="rank",
        ascending=True,
        kind="mergesort",
    ).head(k)


def precision_at_k(
    recommendations: pd.DataFrame,
    relevant_movies: Iterable[Any],
    k: int,
) -> float:
    """Calculate Precision@K for a ranked recommendation list.

    Precision@K = relevant recommended movies in the top K / K.

    The denominator remains K, even if fewer than K recommendations exist.
    This follows the common information-retrieval definition.
    """
    relevant = _normalize_relevant_movies(relevant_movies)
    top_k = _get_top_k(recommendations, k)

    if top_k.empty:
        return 0.0

    hits = top_k["movie_id"].isin(relevant).sum()
    return float(hits / k)


def recall_at_k(
    recommendations: pd.DataFrame,
    relevant_movies: Iterable[Any],
    k: int,
) -> float:
    """Calculate Recall@K for a ranked recommendation list.

    Recall@K = relevant recommended movies in the top K / all relevant movies.

    Returns 0.0 when the ground-truth set is empty.
    """
    relevant = _normalize_relevant_movies(relevant_movies)
    top_k = _get_top_k(recommendations, k)

    if not relevant or top_k.empty:
        return 0.0

    hits = top_k["movie_id"].isin(relevant).sum()
    return float(hits / len(relevant))
