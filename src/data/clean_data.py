import os
import warnings

import pandas as pd
from loguru import logger

from src.config import settings

warnings.filterwarnings("ignore")


# Step 1: Load the Dataset
def load_dataset(file_path):
    """
    Load a parenting dataset from a CSV or JSON file.
    """
    if not os.path.exists(file_path):
        logger.error(
            f"Dataset not found at {file_path}. Please provide the correct path."
        )
        raise FileNotFoundError(f"Dataset not found at {file_path}.")

    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with {len(data)} records from CSV.")
    elif file_path.endswith(".json"):
        data = pd.read_json(file_path, lines=True)
        logger.info(f"Loaded dataset with {len(data)} records from JSON.")
    else:
        logger.error("Unsupported file format. Use CSV or JSON.")
        raise ValueError("Unsupported file format. Use CSV or JSON.")

    return data


# Step 2: Preprocess the Dataset
def preprocess_data(data, full_data=False):
    """
    Clean and prepare the parenting dataset for RAG use.
    """
    # Adjusted to match dataset column names
    required_columns = [
        "QuestionTitle",
        "QuestionBody",
        "QuestionTags",
        "QuestionScore",
        "AnswerBody",
    ]
    for col in required_columns:
        if col not in data.columns:
            logger.error(f"Column {col} not found in the dataset.")
            raise KeyError(f"Column {col} not found in the dataset.")

    # Clean the data
    logger.info("Cleaning and preprocessing data.")
    data = data[required_columns]
    data["QuestionBody"] = data["QuestionBody"].str.replace(
        r"<[^>]*>", "", regex=True
    )  # Remove HTML tags
    data["AnswerBody"] = data["AnswerBody"].str.replace(
        r"<[^>]*>", "", regex=True
    )  # Remove HTML tags
    data["QuestionTitle"] = data["QuestionTitle"].str.strip()
    data["QuestionBody"] = data["QuestionBody"].str.strip()
    data["AnswerBody"] = data["AnswerBody"].str.strip()

    # Filter by score (optional, to get high-quality content)
    data = data[data["QuestionScore"] > 0]
    if full_data:
        logger.info(f"Preprocessed dataset with {len(data)} records.")
    else:
        data = data.head(100)
        logger.info(f"Preprocessed dataset with {len(data)} records (sample).")

    return data


# Step 3: Save Preprocessed Data
def save_preprocessed_data(data, output_path):
    """
    Save the preprocessed data to a file for later use.
    """
    data.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to {output_path}")


# Main Function
def process_qa_data():
    """
    Main function to load, preprocess, and save the parenting dataset.
    """
    # Update this path with your dataset's location
    input_file_path = settings.RAW_DATA_PATH
    output_file_path = settings.PROCESSED_DATA_PATH

    # Configure loguru logger
    logger.add("preprocessing.log", level="INFO", format="{time} {level} {message}")

    try:
        logger.info("Starting data processing pipeline.")
        raw_data = load_dataset(input_file_path)
        processed_data = preprocess_data(raw_data)
        save_preprocessed_data(processed_data, output_file_path)
        logger.info("Data processing pipeline completed successfully.")
    except Exception as e:
        logger.exception(f"Error during data processing: {e}")


if __name__ == "__main__":
    process_qa_data()
