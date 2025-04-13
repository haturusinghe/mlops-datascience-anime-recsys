import polars as pl
from pathlib import Path
import kagglehub
from loguru import logger


def check_files_exists():
    """
    Check if the required CSV files exist in the specified directory.

    Raises:
        FileNotFoundError: If any of the required CSV files are not found.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    csv_path = base_dir / "kaggle" / "anime.csv"
    synopsis_csv_path = base_dir / "kaggle" / "anime_with_synopsis.csv"
    user_csv_path = base_dir / "kaggle" / "rating_complete.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} does not exist.")
    if not synopsis_csv_path.exists():
        raise FileNotFoundError(f"{synopsis_csv_path} does not exist.")
    if not user_csv_path.exists():
        raise FileNotFoundError(f"{user_csv_path} does not exist.")

def download_and_extract_from_kaggle():
    import shutil
    import os
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    kaggle_dir = base_dir / "kaggle"

    # Create the directory if it doesn't exist
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    # Download to the cache (default behavior)
    download_path = kagglehub.dataset_download(
        "hernan4444/anime-recommendation-database-2020",
    )
    
    logger.info(f"Downloaded dataset to {download_path}")
    
    # Copy required CSV files to the kaggle_dir
    required_files = ["anime.csv", "anime_with_synopsis.csv", "rating_complete.csv"]
    for file in required_files:
        source_file = Path(download_path) / file
        if source_file.exists():
            target_file = kaggle_dir / file
            shutil.copy2(source_file, target_file)
            logger.info(f"Copied {file} to {target_file}")
        else:
            logger.error(f"Required file {file} not found in {download_path}")
    
    return kaggle_dir

    

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
    # __file__ is the current file, so go two directories up to the project root.
    base_dir = Path(__file__).parent.parent.parent
    # csv_path = base_dir / "kaggle" / "animelist.csv"
    csv_path = base_dir / "kaggle" / "rating_complete.csv"

    df = pl.read_csv(csv_path, null_values=["Unknown"])

    return df