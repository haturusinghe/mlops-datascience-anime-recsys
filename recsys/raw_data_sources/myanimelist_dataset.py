import polars as pl
from pathlib import Path

def extract_anime_data() -> pl.DataFrame:
    """
    Extracts anime data from the MyAnimeList dataset.

    Returns:
        pl.DataFrame: A DataFrame containing the anime data.
    """
    # __file__ is the current file, so go two directories up to the project root.
    base_dir = Path(__file__).resolve().parent.parent.parent
    csv_path = base_dir / "kaggle" / "anime.csv"
    synopsis_csv_path = base_dir / "kaggle" / "anime_with_synopsis.csv"

    df = pl.read_csv(csv_path, null_values=["Unknown"])
    df_synopsis = pl.read_csv(synopsis_csv_path, null_values=["Unknown"])
    joined_df = df.join(df_synopsis, on="MAL_ID", suffix="_dupe")

    column_names: list[str] = joined_df.columns

    columns_to_drop = [name for name in column_names if "_dupe" in name]
    joined_df = joined_df.drop(columns_to_drop)

    # rename column synopsis to synopsis
    joined_df = joined_df.rename({"sypnopsis": "Synopsis"})

    return joined_df
