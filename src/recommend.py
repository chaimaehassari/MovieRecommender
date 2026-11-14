"""User-based collaborative filtering recommendation logic."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

_TOLERANCE = 1e-12
_OUTPUT_COLUMNS = ["movie_id", "score", "rank"]


def recommend_movies(
    user_id: Any,
    ratings_matrix: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Recommend unseen movies for a user with user-based collaborative filtering.

    Scores are weighted averages of raw ratings from users whose similarity with
    the target user is strictly positive. Movies already rated by the target user
    are excluded. Results are sorted by descending score, then ascending movie_id.
    """
    _validate_inputs(user_id, ratings_matrix, similarity_matrix, n)

    user_ratings = _get_user_ratings(user_id, ratings_matrix)
    if user_ratings.empty:
        return _empty_recommendations()

    positive_neighbors = _get_positive_neighbors(user_id, similarity_matrix)
    if positive_neighbors.empty:
        return _empty_recommendations()

    candidate_movies = _get_candidate_movies(
        positive_neighbors, user_ratings, ratings_matrix
    )
    if candidate_movies.empty:
        return _empty_recommendations()

    scored_movies = _compute_scores(
        candidate_movies, positive_neighbors, ratings_matrix
    )
    return _rank_recommendations(scored_movies, n)


def _validate_inputs(
    user_id: Any,
    ratings_matrix: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    n: int,
) -> None:
    """Validate all inputs before computing recommendations."""
    _validate_ratings_matrix(ratings_matrix)
    _validate_similarity_matrix(ratings_matrix, similarity_matrix)
    _validate_user_id(user_id, ratings_matrix)
    _validate_n(n)


def _validate_ratings_matrix(ratings_matrix: pd.DataFrame) -> None:
    """Validate the structure and values of the ratings matrix."""
    if not isinstance(ratings_matrix, pd.DataFrame):
        raise ValueError("ratings_matrix must be a pandas DataFrame.")
    if not ratings_matrix.index.is_unique:
        raise ValueError("ratings_matrix index must contain unique user IDs.")
    if not ratings_matrix.columns.is_unique:
        raise ValueError("ratings_matrix columns must contain unique movie IDs.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in ratings_matrix.dtypes):
        raise ValueError("ratings_matrix must contain numeric ratings.")

    try:
        values = ratings_matrix.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("ratings_matrix must contain numeric ratings.") from exc

    missing = np.isnan(values)
    finite = np.isfinite(values)
    valid = missing | (finite & (values >= 1.0) & (values <= 5.0))
    if not bool(valid.all()):
        raise ValueError("ratings must be between 1 and 5 or NaN.")


def _validate_similarity_matrix(
    ratings_matrix: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
) -> None:
    """Validate the structure, values and user coverage of the similarity matrix."""
    if not isinstance(similarity_matrix, pd.DataFrame):
        raise ValueError("similarity_matrix must be a pandas DataFrame.")
    if not similarity_matrix.index.is_unique:
        raise ValueError("similarity_matrix index must contain unique user IDs.")
    if not similarity_matrix.columns.is_unique:
        raise ValueError("similarity_matrix columns must contain unique user IDs.")
    if len(similarity_matrix.index) != len(similarity_matrix.columns):
        raise ValueError("similarity_matrix must be square.")

    index_users = set(similarity_matrix.index)
    column_users = set(similarity_matrix.columns)
    if index_users != column_users:
        raise ValueError(
            "similarity_matrix index and columns must contain the same users."
        )

    rating_users = set(ratings_matrix.index)
    if not index_users.issubset(rating_users):
        raise ValueError(
            "similarity_matrix must not contain users absent from ratings_matrix."
        )

    rated_users = set(ratings_matrix.index[ratings_matrix.notna().any(axis=1)])
    if not rated_users.issubset(index_users):
        raise ValueError(
            "similarity_matrix must contain every user with at least one rating."
        )

    if not all(
        pd.api.types.is_numeric_dtype(dtype) for dtype in similarity_matrix.dtypes
    ):
        raise ValueError("similarity_matrix must contain numeric values.")

    try:
        aligned = similarity_matrix.reindex(
            index=similarity_matrix.index, columns=similarity_matrix.index
        )
        values = aligned.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("similarity_matrix must contain numeric values.") from exc

    if not bool(np.isfinite(values).all()):
        raise ValueError("similarity_matrix must contain finite values.")
    if not bool(np.all((values >= -1.0) & (values <= 1.0))):
        raise ValueError("similarity values must be between -1 and 1.")
    if not np.allclose(values, values.T, rtol=0.0, atol=_TOLERANCE):
        raise ValueError("similarity_matrix must be symmetric.")
    if not np.allclose(np.diag(values), 1.0, rtol=0.0, atol=_TOLERANCE):
        raise ValueError("similarity_matrix diagonal must be 1.")


def _validate_user_id(user_id: Any, ratings_matrix: pd.DataFrame) -> None:
    """Validate that the target user exists in the ratings matrix."""
    try:
        exists = user_id in ratings_matrix.index
    except TypeError as exc:
        raise ValueError("user_id must identify a known user.") from exc
    if not exists:
        raise ValueError(f"Unknown user_id: {user_id!r}.")


def _validate_n(n: int) -> None:
    """Validate that n is a positive integer and not a boolean."""
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be an integer greater than or equal to 1.")


def _get_user_ratings(user_id: Any, ratings_matrix: pd.DataFrame) -> pd.Series:
    """Return the non-missing ratings of the target user."""
    return ratings_matrix.loc[user_id].dropna()


def _get_positive_neighbors(
    user_id: Any,
    similarity_matrix: pd.DataFrame,
) -> pd.Series:
    """Return similarities of other users with strictly positive similarity."""
    similarities = similarity_matrix.loc[user_id].drop(labels=[user_id])
    return similarities[similarities > 0]


def _get_candidate_movies(
    positive_neighbors: pd.Series,
    user_ratings: pd.Series,
    ratings_matrix: pd.DataFrame,
) -> pd.Index:
    """Return movies rated by positive neighbors but not by the target user."""
    candidate_movies: set[Any] = set()

    for neighbor_id in positive_neighbors.index:
        neighbor_ratings = ratings_matrix.loc[neighbor_id].dropna()
        candidate_movies.update(neighbor_ratings.index)

    candidate_movies.difference_update(user_ratings.index)
    return pd.Index(sorted(candidate_movies))


def _weighted_average(similarities: pd.Series, ratings: pd.Series) -> float:
    """Compute a similarity-weighted average of raw neighbor ratings."""
    numerator = float((similarities * ratings).sum())
    denominator = float(similarities.sum())
    return numerator / denominator


def _compute_candidate_score(
    movie_id: Any,
    positive_neighbors: pd.Series,
    ratings_matrix: pd.DataFrame,
) -> float | None:
    """Compute one candidate score, or None when no positive neighbor rated it."""
    neighbor_ratings = ratings_matrix.loc[positive_neighbors.index, movie_id].dropna()
    if neighbor_ratings.empty:
        return None

    similarities = positive_neighbors.loc[neighbor_ratings.index]
    return _weighted_average(similarities, neighbor_ratings)


def _compute_scores(
    candidate_movies: pd.Index,
    positive_neighbors: pd.Series,
    ratings_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compute scores for all candidate movies."""
    scored_rows: list[dict[str, Any]] = []

    for movie_id in candidate_movies:
        score = _compute_candidate_score(
            movie_id, positive_neighbors, ratings_matrix
        )
        if score is not None:
            scored_rows.append({"movie_id": movie_id, "score": score})

    return pd.DataFrame(scored_rows, columns=["movie_id", "score"])


def _rank_recommendations(scored_movies: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sort scored movies and add a one-based rank column.

    Scores equal within the configured tolerance are ordered by movie_id.
    """
    if scored_movies.empty:
        return _empty_recommendations()

    ordered = (
        scored_movies.assign(_score_key=scored_movies["score"].round(12))
        .sort_values(
            by=["_score_key", "movie_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(int(n))
        .reset_index(drop=True)
        .copy()
    )
    ordered["rank"] = range(1, len(ordered) + 1)

    return ordered[_OUTPUT_COLUMNS]


def _empty_recommendations() -> pd.DataFrame:
    """Return an empty recommendations DataFrame with the expected columns."""
    return pd.DataFrame(columns=_OUTPUT_COLUMNS)
