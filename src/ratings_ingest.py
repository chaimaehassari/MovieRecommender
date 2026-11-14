from pathlib import Path
from typing import Union
import pandas as pd

REQUIRED_COLUMNS = {"userId", "movieId", "rating"}


def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie un DataFrame d'évaluations MovieLens."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    if df.empty:
        return df.copy()

    cleaned = df.copy()

    cleaned["userId"] = pd.to_numeric(cleaned["userId"], errors="coerce")
    cleaned["movieId"] = pd.to_numeric(cleaned["movieId"], errors="coerce")
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")

    cleaned = cleaned.dropna(subset=["userId", "movieId", "rating"])

    cleaned["userId"] = cleaned["userId"].astype(int)
    cleaned["movieId"] = cleaned["movieId"].astype(int)
    cleaned["rating"] = cleaned["rating"].astype(float)

    cleaned = cleaned[(cleaned["rating"] >= 0.5) & (cleaned["rating"] <= 5.0)]
    cleaned = cleaned.drop_duplicates(subset=["userId", "movieId"], keep="last")

    return cleaned.reset_index(drop=True)


def load_ratings(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Charge un fichier CSV d'évaluations et le nettoie."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    raw_df = pd.read_csv(path)
    return clean_ratings(raw_df)