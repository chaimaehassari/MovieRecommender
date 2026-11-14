"""Tests unitaires pour src/evaluation.py (precision@k et recall@k)."""

import pandas as pd
import pytest

from src.evaluation import precision_at_k, recall_at_k


def make_recs(movie_ids):
    """Construit une liste de recommandations avec des rangs 1..n."""
    return pd.DataFrame(
        {"movie_id": list(movie_ids), "rank": range(1, len(movie_ids) + 1)}
    )


def empty_recs():
    """DataFrame vide mais avec les bons types de colonnes."""
    return pd.DataFrame(
        {
            "movie_id": pd.Series([], dtype="int64"),
            "rank": pd.Series([], dtype="int64"),
        }
    )


# --------------------------------------------------------------------------
# Precision@K
# --------------------------------------------------------------------------
def test_precision_basic():
    recs = make_recs([1, 2, 3, 4, 5])
    assert precision_at_k(recs, {1, 3, 9}, k=5) == pytest.approx(2 / 5)


def test_precision_uses_only_top_k():
    recs = make_recs([1, 2, 3, 4, 5])
    # Top 3 = [1, 2, 3] -> 1 et 3 sont pertinents
    assert precision_at_k(recs, {1, 3, 5}, k=3) == pytest.approx(2 / 3)


def test_precision_respects_rank_order_not_row_order():
    recs = pd.DataFrame({"movie_id": [10, 20, 30], "rank": [3, 1, 2]})
    # Ordre réel : 20, 30, 10 -> top 1 = 20
    assert precision_at_k(recs, {20}, k=1) == 1.0
    assert precision_at_k(recs, {10}, k=1) == 0.0


def test_precision_denominator_stays_k_when_fewer_recommendations():
    recs = make_recs([1, 2])
    assert precision_at_k(recs, {1, 2}, k=5) == pytest.approx(2 / 5)


def test_precision_perfect_and_zero():
    recs = make_recs([1, 2, 3])
    assert precision_at_k(recs, {1, 2, 3}, k=3) == 1.0
    assert precision_at_k(recs, {7, 8, 9}, k=3) == 0.0


def test_precision_empty_recommendations():
    assert precision_at_k(empty_recs(), {1, 2}, k=3) == 0.0


def test_precision_empty_relevant_set():
    assert precision_at_k(make_recs([1, 2, 3]), set(), k=3) == 0.0


# --------------------------------------------------------------------------
# Recall@K
# --------------------------------------------------------------------------
def test_recall_basic():
    recs = make_recs([1, 2, 3, 4, 5])
    assert recall_at_k(recs, {1, 3, 9, 10}, k=5) == pytest.approx(2 / 4)


def test_recall_uses_only_top_k():
    recs = make_recs([1, 2, 3, 4, 5])
    assert recall_at_k(recs, {1, 5}, k=2) == pytest.approx(1 / 2)


def test_recall_k_larger_than_list():
    recs = make_recs([1, 2])
    assert recall_at_k(recs, {1, 2, 3, 4}, k=10) == pytest.approx(2 / 4)


def test_recall_empty_relevant_set_returns_zero():
    assert recall_at_k(make_recs([1, 2, 3]), set(), k=3) == 0.0


def test_recall_empty_recommendations():
    assert recall_at_k(empty_recs(), {1, 2}, k=3) == 0.0


def test_recall_accepts_list_and_ignores_duplicates():
    recs = make_recs([1, 2, 3])
    # [1, 1, 3] -> ensemble {1, 3} : 2 films pertinents
    assert recall_at_k(recs, [1, 1, 3], k=3) == 1.0


def test_recall_accepts_generator():
    recs = make_recs([1, 2, 3])
    assert recall_at_k(recs, (m for m in [1, 2]), k=1) == pytest.approx(1 / 2)


# --------------------------------------------------------------------------
# Validation de k
# --------------------------------------------------------------------------
@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k])
@pytest.mark.parametrize("bad_k", [0, -1, 1.5, True, "3", None])
def test_invalid_k_raises(metric, bad_k):
    with pytest.raises(ValueError, match="k must be"):
        metric(make_recs([1, 2, 3]), {1}, bad_k)


# --------------------------------------------------------------------------
# Validation du DataFrame de recommandations
# --------------------------------------------------------------------------
@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k])
@pytest.mark.parametrize(
    "bad_recs, message",
    [
        ([1, 2, 3], "must be a pandas DataFrame"),
        (pd.DataFrame({"movie_id": [1, 2]}), "missing required column"),
        (pd.DataFrame({"rank": [1, 2]}), "missing required column"),
        (pd.DataFrame({"movie_id": [1, 1], "rank": [1, 2]}), "duplicate movie IDs"),
        (
            pd.DataFrame({"movie_id": [1, None], "rank": [1, 2]}),
            "missing movie IDs",
        ),
        (pd.DataFrame({"movie_id": [1, 2], "rank": [1, 1]}), "must be unique"),
        (
            pd.DataFrame({"movie_id": [1, 2], "rank": [1.0, None]}),
            "must not contain missing",
        ),
        (pd.DataFrame({"movie_id": [1, 2], "rank": ["a", "b"]}), "must be numeric"),
        (pd.DataFrame({"movie_id": [1, 2], "rank": [0, 1]}), "must be positive"),
        (pd.DataFrame({"movie_id": [1, 2], "rank": [-1, 1]}), "must be positive"),
    ],
)
def test_invalid_recommendations_raise(metric, bad_recs, message):
    with pytest.raises(ValueError, match=message):
        metric(bad_recs, {1}, 2)


# --------------------------------------------------------------------------
# Validation de relevant_movies
# --------------------------------------------------------------------------
@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k])
def test_relevant_none_raises(metric):
    with pytest.raises(ValueError, match="must not be None"):
        metric(make_recs([1, 2]), None, 2)


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k])
def test_relevant_not_iterable_raises(metric):
    with pytest.raises(ValueError, match="must be an iterable"):
        metric(make_recs([1, 2]), 5, 2)


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k])
def test_relevant_with_nan_raises(metric):
    with pytest.raises(ValueError, match="missing movie IDs"):
        metric(make_recs([1, 2]), [1, float("nan")], 2)
