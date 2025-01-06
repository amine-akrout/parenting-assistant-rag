from operator import itemgetter

from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from optimum.intel.openvino import OVModelForSequenceClassification
from transformers import AutoTokenizer

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
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10, "lambda_mult": 0.99, "fetch_k": 50},
    )

    model_name = settings.CROSS_ENCODER_MODEL_NAME

    ov_model = OVModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Create your reranker.
    #    The key difference is including `model_kwargs={}` to avoid Pydantic's Field issue.
    ov_compressor = OpenVINOReranker(
        model_name_or_path=model_name,
        ov_model=ov_model,
        tokenizer=tokenizer,
        top_n=3,
        model_kwargs={},  # <--- This ensures **model_kwargs is a normal dict, not a Field
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
    prompt_template = """You are a knowledgeable and empathetic assistant helping parents with their questions.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create chain to combine documents
    qa_chain = (
        {
            "context": (lambda x: x["question"]) | retriever,
            "question": itemgetter("question"),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return qa_chain


# Step 3: Chatbot Function
def chatbot_response(question, retriever, qa_chain):
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
    response = chatbot_response(question, retriever, qa_chain)
    print(f"User: {question}\nChatbot: {response}")
