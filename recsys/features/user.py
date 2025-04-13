import polars as pl

def compute_features_of_user(df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare user data by performing several data cleaning and transformation steps.

    This function does the following:
    1. Checks for required columns in the input DataFrame
    2. Sorts data by user_id and by rating (descending)
    3. Groups data by user_id
    4. For each user, aggregates the top 20 anime IDs and their ratings

    Parameters:
    - df (pl.DataFrame): Input DataFrame containing user rating data.

    Returns:
    - pl.DataFrame: Processed DataFrame with cleaned and transformed user data.

    Raises:
    - ValueError: If any of the required columns are missing from the input DataFrame.
    """

    required_columns = ['user_id', 'anime_id', 'rating']

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing {','.join(missing_columns)} columns in dataframe")
    
    df = df.sort('user_id').sort('rating', descending=True).group_by('user_id').agg(
        pl.col("anime_id").list.head(20).alias("top_anime"), 
        pl.col('rating').list.head(20).alias("top_ratings")
    )

    return df