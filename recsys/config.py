from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class UserDatasetSize(Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"


# Define the settings for the application using Pydantic's BaseSettings
class Settings(BaseSettings):
    # Pydantic's settings configuration, specifying how to load environment variables.
    # This will load environment variables from a .env file and set the encoding.
    model_config = SettingsConfigDict(
        env_file=".env",             # Read environment variables from a .env file.
        env_file_encoding="utf-8",     # Specify the encoding of the .env file.
        # env_prefix="RECSYS_",         # Optional prefix for environment variables (currently commented out).
    )
    
    # HOPSWORKS_API_KEY is used to authenticate with the Hopsworks API.
    # It is stored securely as a SecretStr and can be None if not provided.
    HOPSWORKS_API_KEY: SecretStr | None = None

    # USER_DATASET_SIZE determines the size of the user dataset, with a default value of LARGE.
    USER_DATASET_SIZE: UserDatasetSize = UserDatasetSize.LARGE

    # SentenceTransformer model ID for generating embeddings.
    FEATURES_EMBEDDING_MODEL_ID: str = "all-MiniLM-L6-v2"



settings = Settings()  # Create an instance of the Settings class to access the configuration.