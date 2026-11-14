"""Unit tests for User-Based Collaborative Filtering similarity calculation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.similarity import calculate_user_similarity, get_most_similar_users

SAMPLE_DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample"
    / "user_item_sample.csv"
)


def test_users_with_similar_tastes_have_high_similarity():
    matrix = pd.DataFrame(
        {
            "M1": [5, 4, 5],
            "M2": [4, 5, 4],
            "M3": [5, 4, 5],
        },
        index=["u1", "u2", "u3"],
    )

    similarity = calculate_user_similarity(matrix)

    assert similarity.loc["u1", "u2"] > 0.95
    assert similarity.loc["u1", "u3"] > 0.95


def test_users_with_different_tastes_have_low_similarity():
    matrix = pd.DataFrame(
        {
            "M1": [5, 1, 5],
            "M2": [5, 1, 5],
            "M3": [1, 5, 2],
            "M4": [1, 5, 2],
        },
        index=["u1", "u2", "u3"],
    )

    similarity = calculate_user_similarity(matrix)

    assert similarity.loc["u1", "u2"] < 0.5
    assert similarity.loc["u1", "u3"] > similarity.loc["u1", "u2"]


def test_known_cosine_similarity_example():
    matrix = pd.DataFrame(
        {
            "M1": [1.0, 2.0],
            "M2": [2.0, 1.0],
            "M3": [0.0, 0.0],
        },
        index=["u1", "u2"],
    )

    similarity = calculate_user_similarity(matrix)

    assert similarity.loc["u1", "u2"] == pytest.approx(0.8)


def test_similarity_scores_are_between_minus_one_and_one():
    matrix = pd.DataFrame(
        {
            "M1": [5.0, 1.0, 3.0],
            "M2": [1.0, 5.0, 3.0],
            "M3": [3.0, 3.0, 3.0],
            "M4": [np.nan, 4.0, 0.0],
        },
        index=["u1", "u2", "u3"],
    )

    similarity = calculate_user_similarity(matrix)
    values = similarity.to_numpy()

    assert (values >= -1.0).all()
    assert (values <= 1.0).all()
    assert np.isfinite(values).all()


def test_missing_ratings_are_treated_as_zero():
    matrix_with_missing = pd.DataFrame(
        {
            "M1": [5.0, 4.0, np.nan],
            "M2": [np.nan, 5.0, 1.0],
            "M3": [1.0, np.nan, 5.0],
        },
        index=["u1", "u2", "u3"],
    )
    matrix_with_zeros = matrix_with_missing.fillna(0.0)

    similarity_with_missing = calculate_user_similarity(matrix_with_missing)
    similarity_with_zeros = calculate_user_similarity(matrix_with_zeros)

    pd.testing.assert_frame_equal(similarity_with_missing, similarity_with_zeros)
    assert np.isfinite(similarity_with_missing.to_numpy()).all()


def test_original_input_dataframe_is_not_modified():
    matrix = pd.DataFrame(
        {
            "M1": [5.0, np.nan, 3.0],
            "M2": [np.nan, 4.0, 2.0],
            "M3": [1.0, 2.0, np.nan],
        },
        index=["u1", "u2", "u3"],
    )
    original = matrix.copy(deep=True)

    calculate_user_similarity(matrix)

    pd.testing.assert_frame_equal(matrix, original)
    assert matrix.isna().sum().sum() == 3


def test_empty_user_profile_is_supported():
    matrix = pd.DataFrame(
        {
            "M1": [5.0, 4.0, np.nan],
            "M2": [4.0, 5.0, np.nan],
            "M3": [5.0, 4.0, np.nan],
        },
        index=["u1", "u2", "empty"],
    )

    similarity = calculate_user_similarity(matrix)

    assert similarity.loc["empty", "u1"] == pytest.approx(0.0)
    assert similarity.loc["empty", "u2"] == pytest.approx(0.0)
    assert np.isfinite(similarity.to_numpy()).all()


def test_empty_user_profile_self_similarity_is_zero():
    matrix = pd.DataFrame(
        {
            "M1": [5.0, np.nan, 4.0],
            "M2": [4.0, np.nan, 5.0],
        },
        index=["u1", "empty", "u2"],
    )

    similarity = calculate_user_similarity(matrix)

    assert similarity.loc["empty", "empty"] == pytest.approx(0.0)


def test_users_without_common_rated_movies_have_zero_similarity():
    matrix = pd.DataFrame(
        {
            "M1": [5.0, np.nan],
            "M2": [4.0, np.nan],
            "M3": [np.nan, 5.0],
            "M4": [np.nan, 4.0],
        },
        index=["u1", "u2"],
    )

    similarity = calculate_user_similarity(matrix)

    assert similarity.loc["u1", "u2"] == pytest.approx(0.0)


def test_self_similarity_is_one_for_active_users():
    matrix = pd.DataFrame(
        {
            "M1": [5, 4, 1],
            "M2": [4, 5, 5],
            "M3": [1, 2, 4],
        },
        index=["u1", "u2", "u3"],
    )

    similarity = calculate_user_similarity(matrix)

    np.testing.assert_allclose(np.diag(similarity.to_numpy()), 1.0)
    pd.testing.assert_frame_equal(similarity, similarity.T)


def test_user_ids_are_preserved_as_index_and_columns():
    users = ["alice", "bob", "charlie"]
    matrix = pd.DataFrame(
        [[5, 4], [4, 5], [1, 1]],
        index=users,
        columns=["M1", "M2"],
    )
    matrix.index.name = "user_id"

    similarity = calculate_user_similarity(matrix)

    assert list(similarity.index) == users
    assert list(similarity.columns) == users
    assert similarity.index.name == "user_id"
    assert similarity.columns.name == "user_id"


def test_get_most_similar_users_excludes_self_and_sorts_descending():
    users = ["u1", "u2", "u3", "u4"]
    similarity = pd.DataFrame(
        [
            [1.0, 0.9, 0.7, 0.2],
            [0.9, 1.0, 0.8, 0.3],
            [0.7, 0.8, 1.0, 0.4],
            [0.2, 0.3, 0.4, 1.0],
        ],
        index=users,
        columns=users,
    )

    result = get_most_similar_users(similarity, "u1", top_n=2)

    assert list(result.index) == ["u2", "u3"]
    assert result.iloc[0] >= result.iloc[1]
    assert "u1" not in result.index
    assert len(result) == 2


def test_get_most_similar_users_returns_all_available_when_top_n_is_large():
    users = ["u1", "u2", "u3", "u4"]
    similarity = pd.DataFrame(
        [
            [1.0, 0.9, 0.7, 0.2],
            [0.9, 1.0, 0.8, 0.3],
            [0.7, 0.8, 1.0, 0.4],
            [0.2, 0.3, 0.4, 1.0],
        ],
        index=users,
        columns=users,
    )

    result = get_most_similar_users(similarity, "u1", top_n=10)

    assert list(result.index) == ["u2", "u3", "u4"]


def test_get_most_similar_users_with_single_user_returns_empty_series():
    similarity = pd.DataFrame([[1.0]], index=["u1"], columns=["u1"])

    result = get_most_similar_users(similarity, "u1")

    assert result.empty


def test_calculate_and_get_most_similar_users_work_together():
    matrix = pd.DataFrame(
        {
            "M1": [5, 4, 1, 2],
            "M2": [4, 5, 1, 2],
            "M3": [5, 4, 5, 1],
            "M4": [1, 2, 5, 5],
        },
        index=["target", "close", "different", "other"],
    )

    similarity = calculate_user_similarity(matrix)
    result = get_most_similar_users(similarity, "target", top_n=2)

    assert list(result.index) == ["close", "different"]
    assert "target" not in result.index
    assert len(result) == 2
    assert result.iloc[0] >= result.iloc[1]


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        [1, 2, 3],
        "not a dataframe",
        np.array([[1.0, 2.0]]),
        pd.Series([1.0, 2.0]),
    ],
)
def test_calculate_rejects_non_dataframe_inputs(invalid_input):
    with pytest.raises(TypeError):
        calculate_user_similarity(invalid_input)


def test_calculate_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        calculate_user_similarity(pd.DataFrame())


def test_calculate_rejects_non_numeric_ratings():
    matrix = pd.DataFrame(
        {"M1": ["five", "four"]},
        index=["u1", "u2"],
    )

    with pytest.raises(TypeError):
        calculate_user_similarity(matrix)


def test_calculate_rejects_duplicate_users():
    matrix = pd.DataFrame(
        {"M1": [5, 4]},
        index=["u1", "u1"],
    )

    with pytest.raises(ValueError):
        calculate_user_similarity(matrix)


def test_calculate_rejects_duplicate_movies():
    matrix = pd.DataFrame(
        [[5, 4], [4, 5]],
        index=["u1", "u2"],
        columns=["M1", "M1"],
    )

    with pytest.raises(ValueError):
        calculate_user_similarity(matrix)


def test_calculate_rejects_infinite_ratings():
    matrix = pd.DataFrame(
        {"M1": [np.inf, 4.0]},
        index=["u1", "u2"],
    )

    with pytest.raises(ValueError):
        calculate_user_similarity(matrix)


def test_get_most_similar_users_rejects_non_dataframe():
    with pytest.raises(TypeError):
        get_most_similar_users({"u1": [1.0]}, "u1")


def test_get_most_similar_users_rejects_empty_matrix():
    with pytest.raises(ValueError):
        get_most_similar_users(pd.DataFrame(), "u1")


def test_get_most_similar_users_rejects_non_numeric_scores():
    similarity = pd.DataFrame(
        [["self", "other"], ["other", "self"]],
        index=["u1", "u2"],
        columns=["u1", "u2"],
    )

    with pytest.raises(TypeError):
        get_most_similar_users(similarity, "u1")


def test_get_most_similar_users_rejects_non_square_matrix():
    similarity = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0], [0.2, 0.3]],
        index=["u1", "u2", "u3"],
        columns=["u1", "u2"],
    )

    with pytest.raises(ValueError, match="square"):
        get_most_similar_users(similarity, "u1")


def test_get_most_similar_users_rejects_mismatched_index_and_columns():
    similarity = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=["u1", "u2"],
        columns=["u2", "u1"],
    )

    with pytest.raises(ValueError, match="same user IDs"):
        get_most_similar_users(similarity, "u1")


def test_get_most_similar_users_rejects_different_ids_in_index_and_columns():
    similarity = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=["u1", "u2"],
        columns=["u1", "u3"],
    )

    with pytest.raises(ValueError, match="same user IDs"):
        get_most_similar_users(similarity, "u1")


def test_get_most_similar_users_rejects_missing_scores():
    similarity = pd.DataFrame(
        [[1.0, np.nan], [np.nan, 1.0]],
        index=["u1", "u2"],
        columns=["u1", "u2"],
    )

    with pytest.raises(ValueError, match="missing"):
        get_most_similar_users(similarity, "u1")


def test_get_most_similar_users_rejects_infinite_scores():
    similarity = pd.DataFrame(
        [[1.0, np.inf], [np.inf, 1.0]],
        index=["u1", "u2"],
        columns=["u1", "u2"],
    )

    with pytest.raises(ValueError, match="finite"):
        get_most_similar_users(similarity, "u1")


@pytest.mark.parametrize("invalid_score", [-1.1, 1.1])
def test_get_most_similar_users_rejects_scores_outside_bounds(invalid_score):
    similarity = pd.DataFrame(
        [[1.0, invalid_score], [invalid_score, 1.0]],
        index=["u1", "u2"],
        columns=["u1", "u2"],
    )

    with pytest.raises(ValueError, match="between -1 and 1"):
        get_most_similar_users(similarity, "u1")


def test_get_most_similar_users_accepts_boundary_scores():
    similarity = pd.DataFrame(
        [[1.0, -1.0], [-1.0, 1.0]],
        index=["u1", "u2"],
        columns=["u1", "u2"],
    )

    result = get_most_similar_users(similarity, "u1")

    assert result.loc["u2"] == pytest.approx(-1.0)


@pytest.mark.parametrize("top_n", [np.int32(1), np.int64(1)])
def test_get_most_similar_users_accepts_numpy_integer_top_n(top_n):
    users = ["u1", "u2", "u3"]
    similarity = pd.DataFrame(
        [[1.0, 0.9, 0.7], [0.9, 1.0, 0.8], [0.7, 0.8, 1.0]],
        index=users,
        columns=users,
    )

    result = get_most_similar_users(similarity, "u1", top_n=top_n)

    assert list(result.index) == ["u2"]


def test_get_most_similar_users_rejects_unknown_user():
    users = ["u1", "u2"]
    similarity = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=users,
        columns=users,
    )

    with pytest.raises(KeyError):
        get_most_similar_users(similarity, "missing-user")


@pytest.mark.parametrize("top_n", [0, -1])
def test_get_most_similar_users_rejects_non_positive_top_n(top_n):
    users = ["u1", "u2"]
    similarity = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=users,
        columns=users,
    )

    with pytest.raises(ValueError):
        get_most_similar_users(similarity, "u1", top_n=top_n)


@pytest.mark.parametrize("top_n", [1.5, "5", True, np.bool_(True), None])
def test_get_most_similar_users_rejects_non_integer_top_n(top_n):
    users = ["u1", "u2"]
    similarity = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=users,
        columns=users,
    )

    with pytest.raises(TypeError):
        get_most_similar_users(similarity, "u1", top_n=top_n)


def test_sample_dataset_is_compatible_with_similarity_calculation():
    matrix = pd.read_csv(SAMPLE_DATA, index_col=0)

    assert matrix.shape == (6, 6)
    assert set(matrix.index) == {"U001", "U002", "U003", "U004", "U005", "U006"}

    similarity = calculate_user_similarity(matrix)

    assert similarity.loc["U001", "U002"] > 0.9
    assert similarity.loc["U003", "U005"] == pytest.approx(0.0)
    assert similarity.loc["U006", "U001"] == pytest.approx(0.0)
