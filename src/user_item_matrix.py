import pandas as pd


def create_user_item_matrix(ratings_df):
    """
    Create a User-Item Matrix from a ratings DataFrame.
    """

    required_columns = {"userId", "movieId", "rating"}

    if not required_columns.issubset(ratings_df.columns):
        raise ValueError("Missing required columns")

    matrix = ratings_df.pivot(
        index="userId",
        columns="movieId",
        values="rating"
    )

    return matrix