import polars as pl
from pathlib import Path
import kagglehub
from loguru import logger

from recsys.helpers.kaggle import download_and_extract_from_kaggle


def check_files_exists():
    """
    Check if the required CSV files exist in the specified directory.

    Raises:
        FileNotFoundError: If any of the required CSV files are not found.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    animes_csv_path = base_dir / "kaggle" / "anime.csv"
    synopsis_csv_path = base_dir / "kaggle" / "anime_with_synopsis.csv"
    user_csv_path = base_dir / "kaggle" / "rating_complete.csv"
    ratings_csv_path = base_dir / "kaggle" / "animelist.csv"

    if not animes_csv_path.exists():
        raise FileNotFoundError(f"{animes_csv_path} does not exist.")
    if not synopsis_csv_path.exists():
        raise FileNotFoundError(f"{synopsis_csv_path} does not exist.")
    if not user_csv_path.exists():
        raise FileNotFoundError(f"{user_csv_path} does not exist.")
    if not ratings_csv_path.exists():
        raise FileNotFoundError(f"{ratings_csv_path} does not exist.")


def extract_anime_data() -> pl.DataFrame:
    """
    Extracts anime data from the MyAnimeList dataset.

    Returns:
        pl.DataFrame: A DataFrame containing the anime data.
    """
    # __file__ is the current file, so go two directories up to the project root.

    # Check if the files exist
    try:
        check_files_exists()
    except FileNotFoundError:
        logger.info("Files do not exist. Downloading from Kaggle...")
        download_and_extract_from_kaggle()
        check_files_exists()
        logger.info("Files downloaded successfully.")

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

def extract_user_data() -> pl.DataFrame:
    """
    Extracts user data from the MyAnimeList dataset.

    Returns:
        pl.DataFrame: A DataFrame containing the user data.
    """
    # Check if the files exist
    try:
        check_files_exists()
    except FileNotFoundError:
        logger.info("Files do not exist. Downloading from Kaggle...")
        download_and_extract_from_kaggle()
        check_files_exists()
        logger.info("Files downloaded successfully.")
        
    # Use resolve() to ensure absolute path
    base_dir = Path(__file__).resolve().parent.parent.parent
    csv_path = base_dir / "kaggle" / "rating_complete.csv"

    df = pl.read_csv(csv_path, null_values=["Unknown"])

    return df


def extract_ratings_data() -> pl.DataFrame:
    """
    Extracts ratings data from the MyAnimeList dataset.

    Returns:
        pl.DataFrame: A DataFrame containing the ratings data.
    """
    # Check if the files exist
    try:
        check_files_exists()
    except FileNotFoundError:
        logger.info("Files do not exist. Downloading from Kaggle...")
        download_and_extract_from_kaggle()
        check_files_exists()
        logger.info("Files downloaded successfully.")

    # Use resolve() to ensure absolute path
    base_dir = Path(__file__).resolve().parent.parent.parent
    csv_path = base_dir / "kaggle" / "animelist.csv"

    df = pl.read_csv(csv_path, null_values=["Unknown"])

    return df