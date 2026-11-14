import pandas as pd
from src.user_item_matrix import create_user_item_matrix


def test_create_user_item_matrix():
    ratings = pd.DataFrame({
        "userId": [1, 1, 2, 2],
        "movieId": [10, 20, 10, 30],
        "rating": [4.0, 5.0, 3.0, 4.5]
    })

    matrix = create_user_item_matrix(ratings)

    assert matrix.shape == (2, 3)
    assert matrix.loc[1, 10] == 4.0
    assert matrix.loc[1, 20] == 5.0
    assert pd.isna(matrix.loc[1, 30])
    assert matrix.loc[2, 10] == 3.0