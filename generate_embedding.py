from langchain.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

from config import settings


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


# Step 4: Main Function
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

    try:
        logger.info("Starting embedding generation and FAISS indexing pipeline.")
        data = load_dataset(input_file_path)
        embeddings_model = initialize_embeddings_model()
        generate_and_index_embeddings(data, embeddings_model, faiss_index_path)
        logger.info(
            "Embedding generation and FAISS indexing pipeline completed successfully."
        )
    except Exception as e:
        logger.exception("Error in embedding generation and FAISS indexing pipeline.")


# Main Function
if __name__ == "__main__":
    embed_qa_data()
