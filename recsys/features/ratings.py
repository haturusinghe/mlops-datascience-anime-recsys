import polars as pl
from recsys.features.anime import get_anime_id


def create_watched_episodes_ratio(df: pl.DataFrame) -> pl.Series:
    """
    Calculate the ratio of watched episodes to total episodes for each anime.

    Args:
        df (pl.DataFrame): The DataFrame containing 'watched_episodes' and 'Episodes' columns.

    Returns:
        pl.Series: A Series containing the ratio of watched episodes to total episodes.
    """
    # Avoid division by zero by replacing zeros with 1
    episodes = df["Episodes"].fill_null(0).replace(0, 1)
    
    return (df["watched_episodes"] / episodes).cast(pl.Float32).alias("watched_episodes_ratio")


def compute_features_of_ratings(df: pl.DataFrame, anime_df: pl.DataFrame, default_episodes: int = 50) -> pl.DataFrame:
    """
    Compute additional features based on ratings data.
    
    Args:
        df (pl.DataFrame): The ratings DataFrame.
        anime_df (pl.DataFrame): The anime DataFrame with episode information.
        default_episodes (int, optional): Default value for missing episode counts. Defaults to 50.
        
    Returns:
        pl.DataFrame: The ratings DataFrame with additional computed features.
    """
    # Store original columns to preserve them in the output
    rating_df_original_column_list = df.columns

    # Ensure anime_id is in the correct format
    df = df.with_columns(
        get_anime_id(df, id_column="anime_id"),  # No need to alias if the column name stays the same
    )

    # Join with anime data to get episode information
    df_joined = df.join(
        anime_df, 
        on="anime_id", 
        how="left",
        suffix="_anime"
    ).select([
        pl.col("anime_id"),
        pl.col("watched_episodes"),
        pl.col("Episodes")
    ])

    # Add the ratio feature and select required columns
    return df.with_columns([
        create_watched_episodes_ratio(df_joined),
        create_total_episodes(df_joined, default_episodes)
    ]).select(rating_df_original_column_list + ["total_episodes","watched_episodes_ratio"])


def create_total_episodes(df: pl.DataFrame, default_episodes: int) -> pl.Series:
    """
    Create a Series for total episodes, replacing null values with a default value.

    Args:
        df (pl.DataFrame): The DataFrame containing 'Episodes' column.
        default_episodes (int): Default value for missing episode counts.

    Returns:
        pl.Series: A Series containing the total episodes.
    """
    return df["Episodes"].fill_null(default_episodes).cast(pl.Int32).alias("total_episodes")