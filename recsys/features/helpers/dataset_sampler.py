import random
import polars as pl
from loguru import logger

from recsys.config import UserDatasetSize

class DatasetSampler:
    _SIZES = {
        UserDatasetSize.SMALL : 1_000,
        UserDatasetSize.MEDIUM : 5_000,
        UserDatasetSize.LARGE : 10_000,
    }

    def __init__(self, size:UserDatasetSize, seed: int = 42):
        self._size = size
        self._seed = seed

    @classmethod
    def get_supported_sized(cls) -> dict:
        return cls._SIZES
    
    def sample_dataset(self, ratings_df: pl.DataFrame, users_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
        random.seed(self._seed)

        n_users =self._SIZES[self._size]
        logger.info(f"Sampling {n_users} users from the dataset")
        users_df = users_df.sample(n_users, seed=self._seed)

        logger.info(f"Total Ratings for the sampled users: {ratings_df.height}")
        
        ratings_df = ratings_df.join(users_df, on="user_id")
        logger.info(f"Total Ratings for the sampled users: {ratings_df.height}")

        
        return {"users_df" : users_df, "ratings_df": ratings_df}
