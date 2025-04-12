import contextlib
import io
import sys

import polars as pl
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

def get_anime_id(df : pl.DataFrame) -> pl.Series:
    return df['MAL_ID'].cast(pl.Utf8)


def create_anime_description(row):
    description = f"{row['Name']} is a {row['Type']} anime with a rating of {row['Score']}/10.\n"
    description += f"It has {row['Episodes']} episodes and was released in {row['Aired']}. "
    description += f"It is rated {row['Rating']} and it is ranked {row['Popularity']}, of Most Popular Anime.\n"
    description += f"It belongs to the {row['Genres']} genres.\n"
    description += f"It has a synopsis of {row['Synopsis']}\n"

    return description


def drop_null_rows_for_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    Drop rows with null values for the specified columns.

    Args:
        df (pl.DataFrame): The DataFrame to process.
        columns (list[str]): The list of column names to check for null values.

    Returns:
        pl.DataFrame: The DataFrame with rows containing null values in the specified columns dropped.
    """
    return df.drop_nulls(columns)


def compute_features_of_anime(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        get_anime_id(df).alias("anime_id"),
        # map each row to get description
        pl.struct(df.columns).map_elements(
            lambda row: create_anime_description(row)
        ).alias("description"),
    )

    #list of columns 
    selected_columns = [
                "Name",
                "Type",
                "Score",
                "Episodes",
                "Aired",
                "Rating",
                "Popularity",
                "Genres",
                "Synopsis",
            ]
    
    # for the selected_columns, drop rows with null values 
    df = df.drop_nulls(selected_columns)

    return df