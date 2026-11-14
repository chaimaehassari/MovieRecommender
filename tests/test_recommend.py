import numpy as np
import pandas as pd
import pytest

import src.recommend as recommender

USERS = ["U1", "U2", "U3", "U4", "U5", "U6"]
MOVIES = [101, 102, 103, 104, 105, 106, 107, 108]


@pytest.fixture
def ratings():
    return pd.DataFrame(
        [
            [5.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            [4.0, np.nan, np.nan, 2.0, np.nan, 5.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 5.0, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [5.0, 4.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2.0],
        ],
        index=pd.Index(USERS, name="user_id"),
        columns=pd.Index(MOVIES, name="movie_id"),
    )


@pytest.fixture
def similarities():
    return pd.DataFrame(
        [
            [1.0, 0.8, 0.6, 0.0, 0.4, -0.3],
            [0.8, 1.0, 0.1, 0.0, 0.2, 0.0],
            [0.6, 0.1, 1.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.4, 0.2, 0.3, 0.0, 1.0, 0.5],
            [-0.3, 0.0, 0.0, 0.0, 0.5, 1.0],
        ],
        index=pd.Index(USERS, name="user_id"),
        columns=pd.Index(USERS, name="user_id"),
    )


@pytest.fixture
def similarities_without_unrated_user():
    users = ["U1", "U2", "U3", "U5", "U6"]
    return pd.DataFrame(
        [
            [1.0, 0.8, 0.6, 0.4, -0.3],
            [0.8, 1.0, 0.1, 0.2, 0.0],
            [0.6, 0.1, 1.0, 0.3, 0.0],
            [0.4, 0.2, 0.3, 1.0, 0.5],
            [-0.3, 0.0, 0.0, 0.5, 1.0],
        ],
        index=pd.Index(users, name="user_id"),
        columns=pd.Index(users, name="user_id"),
    )


@pytest.fixture
def similarities_without_positive_neighbors_for_u1():
    return pd.DataFrame(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, -0.3],
            [0.0, 1.0, 0.1, 0.0, 0.2, 0.0],
            [0.0, 0.1, 1.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.2, 0.3, 0.0, 1.0, 0.5],
            [-0.3, 0.0, 0.0, 0.0, 0.5, 1.0],
        ],
        index=pd.Index(USERS, name="user_id"),
        columns=pd.Index(USERS, name="user_id"),
    )


def drop_user(similarity_matrix, user_id):
    return similarity_matrix.drop(index=user_id, columns=user_id)


def test_recommendations_use_raw_weighted_average_and_ranking(ratings, similarities):
    result = recommender.recommend_movies("U1", ratings, similarities, n=5)

    assert list(result.columns) == ["movie_id", "score", "rank"]
    assert result["movie_id"].tolist() == [106, 105, 104, 108, 107]
    assert result["rank"].tolist() == [1, 2, 3, 4, 5]
    assert result["score"].tolist() == pytest.approx(
        [14.0 / 3.0, 3.8, 3.0, 3.0, 1.0]
    )


def test_movies_already_rated_by_target_are_excluded(ratings, similarities):
    result = recommender.recommend_movies("U1", ratings, similarities, n=10)

    assert set(result["movie_id"]).isdisjoint({101, 102, 103})


def test_movies_already_rated_by_target_are_excluded_from_empty_candidates(
    ratings, similarities
):
    result = recommender.recommend_movies("U5", ratings, similarities, n=10)

    assert result.empty
    assert list(result.columns) == ["movie_id", "score", "rank"]


def test_only_strictly_positive_neighbors_contribute(ratings, similarities):
    result = recommender.recommend_movies("U6", ratings, similarities, n=10)

    assert result["movie_id"].tolist() == [101, 104, 102, 106, 103, 105, 107]
    assert result["score"].tolist() == pytest.approx(
        [5.0, 5.0, 4.0, 4.0, 3.0, 2.0, 1.0]
    )


def test_scores_tied_are_ordered_by_movie_id(ratings, similarities):
    result = recommender.recommend_movies("U1", ratings, similarities, n=5)

    tied = result[result["movie_id"].isin([104, 108])]

    assert tied["movie_id"].tolist() == [104, 108]
    assert tied["score"].iloc[0] == pytest.approx(tied["score"].iloc[1])


def test_denominator_only_includes_neighbors_who_rated_the_candidate(
    ratings, similarities
):
    result = recommender.recommend_movies("U1", ratings, similarities, n=5)

    score_105 = result.loc[result["movie_id"] == 105, "score"].iloc[0]

    assert score_105 == pytest.approx(3.8)


@pytest.mark.parametrize(
    ("n", "expected_movies"),
    [
        (1, [106]),
        (3, [106, 105, 104]),
        (100, [106, 105, 104, 108, 107]),
    ],
)
def test_n_limits_recommendations(ratings, similarities, n, expected_movies):
    result = recommender.recommend_movies("U1", ratings, similarities, n=n)

    assert result["movie_id"].tolist() == expected_movies
    assert result["rank"].tolist() == list(range(1, len(expected_movies) + 1))


def test_default_n_is_10(ratings, similarities):
    result = recommender.recommend_movies("U1", ratings, similarities)

    assert len(result) == 5


def test_unknown_user_raises_value_error(ratings, similarities):
    with pytest.raises(ValueError):
        recommender.recommend_movies("U99", ratings, similarities, n=3)


def test_user_without_ratings_returns_empty_even_if_present_in_similarity(
    ratings, similarities
):
    result = recommender.recommend_movies("U4", ratings, similarities, n=3)

    assert result.empty
    assert list(result.columns) == ["movie_id", "score", "rank"]


def test_user_without_ratings_returns_empty_when_omitted_from_similarity(
    ratings, similarities_without_unrated_user
):
    result = recommender.recommend_movies(
        "U4", ratings, similarities_without_unrated_user, n=3
    )

    assert result.empty
    assert list(result.columns) == ["movie_id", "score", "rank"]


def test_unrated_user_may_be_omitted_from_similarity(
    ratings, similarities_without_unrated_user
):
    result = recommender.recommend_movies(
        "U1", ratings, similarities_without_unrated_user, n=3
    )

    assert result["movie_id"].tolist() == [106, 105, 104]


def test_no_positive_neighbors_returns_empty(
    ratings, similarities_without_positive_neighbors_for_u1
):
    result = recommender.recommend_movies(
        "U1", ratings, similarities_without_positive_neighbors_for_u1, n=3
    )

    assert result.empty
    assert list(result.columns) == ["movie_id", "score", "rank"]


@pytest.mark.parametrize("bad_n", [0, -1, 1.5, "10", True, None])
def test_invalid_n_raises_value_error(ratings, similarities, bad_n):
    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", ratings, similarities, n=bad_n)


def test_ratings_matrix_must_be_a_dataframe(ratings, similarities):
    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", ratings.to_numpy(), similarities, n=3)


def test_similarity_matrix_must_be_a_dataframe(ratings, similarities):
    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", ratings, similarities.to_numpy(), n=3)


@pytest.mark.parametrize("bad_rating", [-1.0, 0.0, 5.5, 6.0, np.inf, -np.inf])
def test_ratings_must_be_between_1_and_5(ratings, similarities, bad_rating):
    invalid = ratings.copy()
    invalid.loc["U2", 104] = bad_rating

    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", invalid, similarities, n=3)


def test_duplicate_users_are_rejected(ratings, similarities):
    invalid = pd.concat([ratings, ratings.iloc[[0]]])

    with pytest.raises(ValueError):
        recommender.recommend_movies("U2", invalid, similarities, n=3)


def test_duplicate_movies_are_rejected(ratings, similarities):
    invalid = pd.concat([ratings, ratings.iloc[:, [0]]], axis=1)

    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", invalid, similarities, n=3)


def test_similarity_matrix_must_be_square(ratings, similarities):
    invalid = similarities.iloc[:-1]

    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", ratings, invalid, n=3)


def test_similarity_matrix_must_be_symmetric(ratings, similarities):
    invalid = similarities.copy()
    invalid.loc["U2", "U3"] = 0.4

    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", ratings, invalid, n=3)


def test_similarity_matrix_diagonal_must_be_one(ratings, similarities):
    invalid = similarities.copy()
    invalid.loc["U3", "U3"] = 0.9

    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", ratings, invalid, n=3)


@pytest.mark.parametrize("bad_similarity", [1.1, -1.1])
def test_similarity_values_must_be_within_minus_one_and_one(
    ratings, similarities, bad_similarity
):
    invalid = similarities.copy()
    invalid.loc["U2", "U3"] = bad_similarity
    invalid.loc["U3", "U2"] = bad_similarity

    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", ratings, invalid, n=3)


def test_similarity_matrix_must_not_contain_nan(ratings, similarities):
    invalid = similarities.copy()
    invalid.loc["U2", "U3"] = np.nan
    invalid.loc["U3", "U2"] = np.nan

    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", ratings, invalid, n=3)


def test_similarity_index_and_columns_must_have_the_same_users(ratings, similarities):
    invalid = similarities.rename(columns={"U6": "U7"})

    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", ratings, invalid, n=3)


def test_similarity_must_contain_every_user_with_at_least_one_rating(
    ratings, similarities
):
    invalid = drop_user(similarities, "U1")

    with pytest.raises(ValueError):
        recommender.recommend_movies("U2", ratings, invalid, n=3)


def test_similarity_must_not_contain_users_absent_from_ratings(ratings, similarities):
    invalid = similarities.rename(index={"U4": "U99"}, columns={"U4": "U99"})

    with pytest.raises(ValueError):
        recommender.recommend_movies("U1", ratings, invalid, n=3)


def test_inputs_are_not_mutated(ratings, similarities):
    ratings_before = ratings.copy(deep=True)
    similarities_before = similarities.copy(deep=True)

    recommender.recommend_movies("U1", ratings, similarities, n=5)

    pd.testing.assert_frame_equal(ratings, ratings_before)
    pd.testing.assert_frame_equal(similarities, similarities_before)
