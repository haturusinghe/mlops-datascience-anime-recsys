from pathlib import Path
import kagglehub
from loguru import logger

def download_and_extract_from_kaggle():
    """
    Downloads and extracts the required anime dataset files from Kaggle.
    
    Returns:
        Path: Path to the kaggle directory where files were extracted
        
    Raises:
        FileNotFoundError: If any required file couldn't be found in the downloaded dataset
        RuntimeError: If the download fails for any reason
    """
    import shutil
    import os
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    kaggle_dir = base_dir / "kaggle"

    # Create the directory if it doesn't exist
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download to the cache (default behavior)
        download_path = kagglehub.dataset_download(
            "hernan4444/anime-recommendation-database-2020",
        )
        
        logger.info(f"Downloaded dataset to {download_path}")
        
        # Copy required CSV files to the kaggle_dir
        required_files = ["anime.csv", "anime_with_synopsis.csv", "rating_complete.csv"]
        missing_files = []
        
        for file in required_files:
            source_file = Path(download_path) / file
            if source_file.exists():
                target_file = kaggle_dir / file
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied {file} to {target_file}")
            else:
                missing_files.append(file)
                logger.error(f"Required file {file} not found in {download_path}")
        
        # Check if all required files were downloaded and copied successfully
        if missing_files:
            raise FileNotFoundError(f"The following required files could not be found in the downloaded dataset: {', '.join(missing_files)}")
    
    except Exception as e:
        logger.error(f"Failed to download dataset from Kaggle: {str(e)}")
        
        # Try using kaggle CLI as fallback
        try:
            logger.info("Attempting to download using kaggle CLI as fallback...")
            import subprocess
            
            # Create temporary directory for download
            import tempfile
            temp_dir = tempfile.mkdtemp()
            
            # Run kaggle command to download dataset
            cmd = ["kaggle", "datasets", "download", "-d", "hernan4444/anime-recommendation-database-2020", "--path", temp_dir, "--unzip"]
            subprocess.run(cmd, check=True)
            
            # Copy required files from temp directory to kaggle directory
            required_files = ["anime.csv", "anime_with_synopsis.csv", "rating_complete.csv", "animelist.csv"]
            missing_files = []
            
            for file in required_files:
                source_file = Path(temp_dir) / file
                if source_file.exists():
                    target_file = kaggle_dir / file
                    shutil.copy2(source_file, target_file)
                    logger.info(f"Copied {file} to {target_file}")
                else:
                    missing_files.append(file)
                    logger.error(f"Required file {file} not found in {temp_dir}")
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            # Check if all required files were downloaded and copied successfully
            if missing_files:
                raise FileNotFoundError(f"The following required files could not be found in the downloaded dataset: {', '.join(missing_files)}")
                
        except Exception as inner_e:
            logger.error(f"Fallback download also failed: {str(inner_e)}")
            raise RuntimeError(f"Failed to download dataset using both kagglehub and kaggle CLI: {str(e)} and {str(inner_e)}")
    
    return kaggle_dir