import polars as pl

def compute_features_of_user(df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare user data by performing several data cleaning and transformation steps.

    This function does the following:
    1. Checks for required columns in the input DataFrame
    2. Groups data by user_id
    3. For each user, sorts by rating (descending) and aggregates the top 20 anime IDs and ratings

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
    
    # First sort the entire dataframe by user_id and then by rating (descending)
    # Then group by user_id
    result = df.group_by('user_id').agg(
        # Sort anime_ids and ratings by rating in descending order within each group
        pl.col('anime_id').sort_by(pl.col('rating'), descending=True).list.head(20).alias("top_anime"),
        pl.col('rating').sort_by(pl.col('rating'), descending=True).list.head(20).alias("top_ratings")
    )
    
    return result