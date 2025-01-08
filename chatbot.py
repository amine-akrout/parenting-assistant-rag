"""
This module implements a chatbot for assisting parents with their questions using
a retrieval-based approach.
"""

import pickle

from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from optimum.intel.openvino import OVModelForSequenceClassification
from transformers import AutoTokenizer

from config import settings
from monitoring import create_langfuse_handler


# pylint: disable=W0621,C0103
# Step 1: Load the FAISS index and retriever
def load_retriever():
    """
    Load the FAISS and BM25 retrievers.
    """
    embeddings_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL_NAME)
    vector_store = FAISS.load_local(
        settings.FAISS_INDEX_PATH,
        embeddings_model,
        allow_dangerous_deserialization=True,
    )
    embedding_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.FAISS_TOP_K},
    )

    # Load the BM25 Retriever
    with open(settings.BM25_INDEX_PATH, "rb") as file:
        bm25_retriever = pickle.load(file)
    bm25_retriever.k = settings.BM25_TOP_K

    # Create an ensemble retriever

    base_retriever = EnsembleRetriever(
        retrievers=[embedding_retriever, bm25_retriever],
        weights=settings.RETRIEVER_WEIGHTS,
        top_k=settings.RETTRIEVER_TOP_K,
    )

    model_name = settings.CROSS_ENCODER_MODEL_NAME

    ov_model = OVModelForSequenceClassification.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ov_compressor = OpenVINOReranker(
        model_name_or_path=model_name,
        ov_model=ov_model,
        tokenizer=tokenizer,
        top_n=3,
        model_kwargs={},
    )

    # Add Cross Encoder Reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=ov_compressor, base_retriever=base_retriever
    )

    return compression_retriever


# Step 2: Initialize the Retrieval Chain
def create_chatbot_chain(retriever):
    """
    Create a retrieval chain-based chatbot.
    """
    # Define the prompt template
    prompt_template = """You are a knowledgeable and empathetic assistant helping parents with
    their questions.

    {context}

    Question: {question}
    Answer:"""
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Initialize LLM
    set_llm_cache(InMemoryCache())
    llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME, temperature=0, max_tokens=100, cache=True
    )

    def format_docs(docs):
        # return "\n".join([doc.page_content for doc in docs])
        answer_bodies = []
        for doc in docs:
            content = doc.page_content
            if "AnswerBody:" in content:
                # Extract the AnswerBody content
                answer_start = content.find("AnswerBody:") + len("AnswerBody:")
                answer_body = content[answer_start:].strip()
                answer_bodies.append(answer_body)
        return "\n\n".join(answer_bodies)

    # Create chain to combine documents
    qa_chain = (
        RunnableParallel(
            {
                "context": (lambda x: x["question"]) | retriever | format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain


# Step 3: Chatbot Function
def chatbot_response(question, qa_chain):
    """
    Get a chatbot response for a given query.

    Args:
        question (str): The user's question or input.
        retriever: The document retriever.
        qa_chain: The retrieval QA chain.

    Returns:
        str: The chatbot's response.
    """
    # create the langfuse handler
    langfuse_handler = create_langfuse_handler()

    if not question.strip():
        return "Please ask a valid question."

    # Get the response
    result = qa_chain.invoke(
        {"question": question}, config={"callbacks": [langfuse_handler]}
    )
    return result


if __name__ == "__main__":
    retriever = load_retriever()
    qa_chain = create_chatbot_chain(retriever)

    # Test a query
    question = "How do I manage my toddler's tantrums?"
    question = "How do I get my child to sleep"
    question = "How do I get my child to eat vegetables"
    question = "How do I get my child to stop hitting"
    response = chatbot_response(question, qa_chain)
    print(f"User: {question}\nChatbot: {response}")
