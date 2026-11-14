"""User-based collaborative filtering similarity calculation.

Provisional data contract
-------------------------
The input is a pandas DataFrame where:

* each row represents a user (the DataFrame index contains user IDs);
* each column represents a movie (the DataFrame columns contain movie IDs);
* each cell is a numeric rating;
* missing ratings may be represented by ``NaN`` or ``0``.

This contract is provisional and must be confirmed with the other team
members before the modules are integrated.

Missing ratings are replaced by ``0`` before cosine similarity is computed.
Cosine similarity is bounded between ``-1`` and ``1`` (``0`` to ``1`` for
non-negative ratings). An empty profile is an all-zero vector, so its cosine
similarity with every user is ``0``.
"""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _validate_user_item_matrix(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a user-item matrix without mutating the input."""
    if not isinstance(user_item_matrix, pd.DataFrame):
        raise TypeError("user_item_matrix must be a pandas DataFrame")

    if user_item_matrix.empty:
        raise ValueError(
            "user_item_matrix must contain at least one user and one movie"
        )

    if user_item_matrix.index.has_duplicates:
        raise ValueError("user IDs (the DataFrame index) must be unique")

    if user_item_matrix.columns.has_duplicates:
        raise ValueError("movie IDs (the DataFrame columns) must be unique")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in user_item_matrix.dtypes
    ):
        raise TypeError("user_item_matrix must contain numeric ratings only")

    matrix = user_item_matrix.fillna(0.0).astype(float)
    values = matrix.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("ratings must contain only finite values")

    return matrix


def calculate_user_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise user similarity with cosine similarity.

    Parameters
    ----------
    user_item_matrix:
        DataFrame with users as rows, movies as columns and ratings as values.
        Missing ratings may be ``NaN`` or ``0``.

    Returns
    -------
    pandas.DataFrame
        A square similarity matrix whose index and columns are the user IDs
        from ``user_item_matrix``, in the same order.
    """
    matrix = _validate_user_item_matrix(user_item_matrix)
    similarities = cosine_similarity(matrix.to_numpy(dtype=float))
    similarities = np.clip(similarities, -1.0, 1.0)

    return pd.DataFrame(
        similarities,
        index=matrix.index.copy(),
        columns=matrix.index.copy(),
    )


def get_most_similar_users(
    similarity_matrix: pd.DataFrame,
    user_id: Hashable,
    top_n: int = 5,
) -> pd.Series:
    """Return the users most similar to ``user_id``.

    Parameters
    ----------
    similarity_matrix:
        Square user similarity matrix, typically produced by
        :func:`calculate_user_similarity`.
    user_id:
        ID of the user whose neighbours should be returned.
    top_n:
        Maximum number of neighbours to return. Must be at least ``1``.
        Python integers and NumPy integer types are accepted.

    Returns
    -------
    pandas.Series
        Similarity scores indexed by user ID, sorted in descending order.
        The requested user is excluded from the result.

    Notes
    -----
    ``similarity_matrix`` must be square, must have identical user IDs in
    its index and columns in the same order, and must contain only finite
    scores between ``-1`` and ``1``.
    """
    if not isinstance(similarity_matrix, pd.DataFrame):
        raise TypeError("similarity_matrix must be a pandas DataFrame")

    if similarity_matrix.empty:
        raise ValueError("similarity_matrix must not be empty")

    if isinstance(top_n, bool) or not isinstance(top_n, Integral):
        raise TypeError("top_n must be an integer")

    top_n = int(top_n)

    if top_n < 1:
        raise ValueError("top_n must be greater than or equal to 1")

    if similarity_matrix.index.has_duplicates:
        raise ValueError("similarity matrix user IDs must be unique")

    if similarity_matrix.columns.has_duplicates:
        raise ValueError("similarity matrix columns must be unique")

    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("similarity_matrix must be square")

    if not similarity_matrix.index.equals(similarity_matrix.columns):
        raise ValueError(
            "similarity_matrix index and columns must contain the same "
            "user IDs in the same order"
        )

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in similarity_matrix.dtypes
    ):
        raise TypeError("similarity_matrix must contain numeric scores only")

    if similarity_matrix.isna().to_numpy().any():
        raise ValueError("similarity scores must not contain missing values")

    score_values = similarity_matrix.to_numpy(dtype=float)

    if not np.isfinite(score_values).all():
        raise ValueError("similarity scores must be finite")

    if ((score_values < -1.0) | (score_values > 1.0)).any():
        raise ValueError("similarity scores must be between -1 and 1")

    if user_id not in similarity_matrix.index:
        raise KeyError(f"unknown user ID: {user_id!r}")

    if user_id not in similarity_matrix.columns:
        raise ValueError(
            "similarity_matrix must contain user_id in both its index and columns"
        )

    scores = similarity_matrix.loc[user_id].drop(labels=[user_id])
    scores = pd.to_numeric(scores, errors="raise")
    scores = scores.sort_values(ascending=False, kind="mergesort")

    return scores.head(top_n).rename("similarity")
