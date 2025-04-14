import contextlib
import io
import sys

import polars as pl
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

def get_anime_id(df : pl.DataFrame, id_column:str = 'MAL_ID') -> pl.Series:
    return df[id_column].cast(pl.Utf8)


def create_anime_description(row):
    description = f"This is a {row['Type']} anime.\n"
    description += f"It belongs to the {row['Genres']} genres.\n"
    description += f"It has a synopsis : {row['Synopsis']}\n"

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

def generate_embeddings_for_dataframe(
    df: pl.DataFrame,
    embedding_column_name: str,
    model: SentenceTransformer,
    batch_size: int = 32,
) -> pl.DataFrame:
    """
    Generate embeddings for a specified text column in a DataFrame using a SentenceTransformer model.

    Args:
        df (pl.DataFrame): The DataFrame containing the text data.
        text_col (str): The name of the column containing the text data.
        model (SentenceTransformer): The SentenceTransformer model to use for generating embeddings.
        batch_size (int, optional): The batch size for processing. Defaults to 32.

    Returns:
        pl.DataFrame: The DataFrame with an additional column for embeddings.
    """

    @contextlib.contextmanager
    def suppress_stdout():
        new_stdout = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = new_stdout
        try:
            yield new_stdout
        finally:
            sys.stdout = old_stdout

    total_rows = len(df)
    progress_bar = tqdm(total=total_rows, desc="Generating embeddings", unit="row")

    texts = df[embedding_column_name].to_list()
    all_embeddings = []

    for i in range(0,total_rows, batch_size):
        batch = texts[i:i + batch_size]
        with suppress_stdout():
            embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings) # the .extend actually extends the list by adding the elements of the iterable
        progress_bar.update(len(batch))

    df_with_embeddings = df.with_columns(
        pl.Series("embeddings", all_embeddings)
    )
    progress_bar.close()
    return df_with_embeddings
