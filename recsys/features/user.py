# Configure Polars to display the full content of list columns
import polars as pl



def compute_features_of_user(df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare user data by performing several data cleaning and transformation steps.

    This function does the following:
    1. Checks for required columns in the input DataFrame
    2. Sorts data by user_id and rating (descending)
    3. Groups data by user_id
    4. For each user, aggregates the top 20 anime IDs and their ratings

    Parameters:
    - df (pl.DataFrame): Input DataFrame containing user rating data.

    Returns:
    - pl.DataFrame: Processed DataFrame with cleaned and transformed user data.

    Raises:
    - ValueError: If any of the required columns are missing from the input DataFrame.
    """

    # Set the max_columns width to display full list content
    pl.Config.set_tbl_width_chars(1000)

    # Set the max list elements displayed
    pl.Config.set_fmt_str_lengths(100)

    required_columns = {'user_id', 'anime_id', 'rating'}

    # Check for missing columns
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the DataFrame: {', '.join(missing_columns)}")

    # Sort by user_id (ascending) and rating (descending)
    df_sorted = df.sort(["user_id", "rating"], descending=[False, True])

    # Group by user_id and collect the anime_id and rating values as lists
    # Use keyword argument syntax for Polars 1.9.0+ list aggregation
    result = df_sorted.group_by('user_id').agg(
        top_anime=pl.col('anime_id'),
        top_ratings=pl.col('rating')
    )

    # Limit each list to at most 20 items using .list.head() for Polars 1.9.0
    result = result.with_columns([
        pl.col('top_anime').list.head(20).alias('top_anime'),
        pl.col('top_ratings').list.head(20).alias('top_ratings')
    ])

    return result