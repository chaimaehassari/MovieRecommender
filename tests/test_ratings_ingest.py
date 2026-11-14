import pandas as pd
import pytest
from src.ratings_ingest import clean_ratings, load_ratings, save_ratings


def test_load_and_clean_valid_ratings(tmp_path):
    csv_file = tmp_path / "ratings.csv"
    csv_file.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,10,4.0,1000\n"
        "1,10,5.0,1001\n"
        "2,20,,1002\n"
        "3,30,9.5,1003\n"
        "4,40,3.5,1004\n",
        encoding="utf-8",
    )

    df = load_ratings(csv_file)

    assert len(df) == 2
    assert list(df["userId"]) == [1, 4]
    assert list(df["rating"]) == [5.0, 3.5]


def test_clean_empty_dataframe():
    empty_df = pd.DataFrame(columns=["userId", "movieId", "rating"])
    result = clean_ratings(empty_df)
    assert result.empty


def test_missing_columns_raises_error():
    bad_df = pd.DataFrame({"userId": [1], "movieId": [10]})
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        clean_ratings(bad_df)


def test_file_not_found_raises_error():
    with pytest.raises(FileNotFoundError):
        load_ratings("chemin/inexistant/ratings.csv")


def test_clean_keeps_only_useful_columns():
    df = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
            "rating": [4.0],
            "y": [-2.5],
            "movieId_orig": [1689],
        }
    )

    result = clean_ratings(df)

    assert list(result.columns) == ["userId", "movieId", "rating"]


def test_clean_keeps_timestamp_when_present():
    df = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
            "rating": [4.0],
            "timestamp": [1000],
            "extra": ["x"],
        }
    )

    result = clean_ratings(df)

    assert list(result.columns) == ["userId", "movieId", "rating", "timestamp"]


def test_save_ratings_writes_csv_and_creates_folder(tmp_path):
    df = pd.DataFrame(
        {
            "userId": [1, 4],
            "movieId": [10, 40],
            "rating": [5.0, 3.5],
        }
    )
    output = tmp_path / "processed" / "ratings_clean.csv"

    saved = save_ratings(df, output)

    assert saved == output
    assert output.exists()
    reloaded = pd.read_csv(output)
    assert list(reloaded.columns) == ["userId", "movieId", "rating"]
    assert len(reloaded) == 2


def test_saved_ratings_can_be_loaded_again(tmp_path):
    df = pd.DataFrame(
        {
            "userId": [1, 4],
            "movieId": [10, 40],
            "rating": [5.0, 3.5],
        }
    )
    output = tmp_path / "ratings_clean.csv"

    save_ratings(df, output)
    reloaded = load_ratings(output)

    assert list(reloaded["userId"]) == [1, 4]
    assert list(reloaded["movieId"]) == [10, 40]
    assert list(reloaded["rating"]) == [5.0, 3.5]