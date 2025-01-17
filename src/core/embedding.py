"""
This script generates embeddings for a parenting dataset and indexes them using FAISS and BM25.

"""

import os
import pickle

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from loguru import logger

from src.config import settings


# pylint: disable=C0103,W0718
# Step 1: Load the Dataset
def load_dataset(file_path):
    """
    Load the preprocessed parenting dataset.
    """
    try:
        loader = CSVLoader(file_path, encoding="utf-8")

        data = loader.load()
        logger.info(f"Loaded dataset with {len(data)} records from CSV.")
        return data
    except Exception as e:
        logger.exception("Failed to load dataset.")
        raise e


# Step 2: Initialize the Embedding Model
def initialize_embeddings_model():
    """
    Initialize the HuggingFaceEmbeddings.
    """
    try:
        model_name = settings.EMBEDDINGS_MODEL_NAME
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Successfully initialized embeddings model: {model_name}")
        return embeddings
    except Exception as e:
        logger.exception("Failed to initialize embeddings model.")
        raise e


# Step 3: Generate and Index Embeddings
def generate_and_index_embeddings(data, embeddings_model, faiss_index_path):
    """
    Generate embeddings for question-answer pairs and index them in FAISS.
    """
    try:
        # Create FAISS vector store
        logger.info("Indexing embeddings in FAISS...")
        vector_store = FAISS.from_documents(data, embeddings_model)

        # Save FAISS index
        vector_store.save_local(faiss_index_path)
        logger.info(f"FAISS index saved to {faiss_index_path}")

    except Exception as e:
        logger.exception("Failed to generate and index embeddings.")
        raise e


# Step 4: Generate BM25 Index
def generate_bm25_index(data, file_path=settings.BM25_INDEX_PATH):
    """
    Generate BM25 index for question-answer pairs.
    """
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    else:
        logger.info("Generating BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(documents=data)
    with open(file_path, "wb") as file:
        pickle.dump(bm25_retriever, file)
    logger.info(f"BM25 index saved to {file_path}")


# Step 5: Main Function
def embed_qa_data():
    """
    Main function to load, preprocess, and save the parenting dataset.
    """
    # Update this path with your dataset's location
    input_file_path = settings.PROCESSED_DATA_PATH
    faiss_index_path = settings.FAISS_INDEX_PATH

    # Configure loguru logger
    logger.add(
        "embedding_generation.log", level="INFO", format="{time} {level} {message}"
    )
    data = load_dataset(input_file_path)

    if os.path.exists(faiss_index_path):
        logger.info(f"FAISS index already exists at {faiss_index_path}.")
    else:
        try:
            logger.info("Starting embedding generation and FAISS indexing pipeline.")
            embeddings_model = initialize_embeddings_model()
            generate_and_index_embeddings(data, embeddings_model, faiss_index_path)
            logger.info(
                "Embedding generation and FAISS indexing pipeline completed successfully."
            )
        except Exception as e:
            logger.exception(
                "Error in embedding generation and FAISS indexing pipeline."
            )
    if os.path.exists(settings.BM25_INDEX_PATH):
        logger.info(f"BM25 index already exists at {settings.BM25_INDEX_PATH}.")
    else:
        try:
            logger.info("Starting BM25 index generation pipeline.")
            generate_bm25_index(data)
            logger.info("BM25 index generation pipeline completed successfully.")
        except Exception as e:
            logger.exception("Error in BM25 index generation pipeline.")


# Main Function
if __name__ == "__main__":
    # make the notebook run correctyly by setting the correct path
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    embed_qa_data()
