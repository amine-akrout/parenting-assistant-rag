from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config import settings
from monitoring import create_langfuse_handler


# Step 1: Load the FAISS index and retriever
def load_retriever():
    """
    Load the FAISS retriever.
    """
    embeddings_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL_NAME)
    vector_store = FAISS.load_local(
        settings.FAISS_INDEX_PATH,
        embeddings_model,
        allow_dangerous_deserialization=True,
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "lambda_mult": 0.99, "fetch_k": 50},
    )
    return retriever


# Step 2: Initialize the Retrieval Chain
def create_chatbot_chain(retriever):
    """
    Create a retrieval chain-based chatbot.
    """
    # Define the prompt template
    prompt_template = """You are a knowledgeable and empathetic assistant helping parents with their questions.

    {context}

    Question: {input}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "input"]
    )

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create chain to combine documents
    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)

    # Create the retrieval chain
    qa_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=combine_docs_chain
    )
    return qa_chain


# Step 3: Chatbot Function
def chatbot_response(user_query, retriever, qa_chain):
    """
    Get a chatbot response for a given query.

    Args:
        user_query (str): The user's question or input.
        retriever: The document retriever.
        qa_chain: The retrieval QA chain.

    Returns:
        str: The chatbot's response.
    """
    # create the langfuse handler
    langfuse_handler = create_langfuse_handler()

    if not user_query.strip():
        return "Please ask a valid question."

    # Get the response
    result = qa_chain.invoke(
        {"input": user_query}, config={"callbacks": [langfuse_handler]}
    )
    return result["answer"]


if __name__ == "__main__":
    retriever = load_retriever()
    qa_chain = create_chatbot_chain(retriever)

    # Test a query
    query = "How do I manage my toddler's tantrums?"
    response = chatbot_response(query, retriever, qa_chain)
    print(f"User: {query}\nChatbot: {response}")
