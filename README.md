# 🤖 AI-Powered Parenting Assistant

## 📌 Overview
The **AI-Powered Parenting Assistant** is an intelligent chatbot designed to provide safe, reliable, and empathetic responses to parenting-related questions. Built using **LangChain**, **LLM Guard**, and **FastAPI**, the system leverages **Retrieval-Augmented Generation (RAG)** to enhance its responses with relevant knowledge from curated sources.

The chatbot also integrates **advanced content filtering** to ensure the safety and appropriateness of responses, making it a powerful demonstration of AI engineering best practices.

---

## 🚀 Features
### ✅ **Retrieval-Augmented Generation (RAG) System**
- Uses **FAISS vector search** combined with **BM25 retrieval** for accurate contextual knowledge retrieval.
- Implements **OpenVINO Reranker** for document ranking and relevance filtering.

### 🛡 **AI Safety with LLM Guard**
- **Input Filtering**: Blocks **toxic, political, religious, and self-harm topics** using `llm_guard`.
- **Output Validation**: Scans model responses for **language safety, sensitivity, and relevance**.
- **Routing Mechanism**: Ensures inappropriate queries are handled gracefully.

### ⚡ **High-Performance AI Pipeline**
- **LangChain Pipelines**: Modular **runnable chains** for seamless data flow.
- **Prompt Engineering**: Optimized **few-shot prompting** for structured AI outputs.
- **FastAPI Backend**: Fully asynchronous **REST API** for real-time chatbot interactions.
- **Langfuse Monitoring**: Enables **real-time LLM performance tracking**.

### 🔥 **Job-Worthy Tech Stack**
✅ **AI & LLMs**: OpenAI GPT Models, LangChain, HuggingFace Embeddings  
✅ **Data Retrieval**: FAISS, BM25, OpenVINO Reranker  
✅ **Safety**: LLM Guard (Input & Output Filtering)  
✅ **Backend API**: FastAPI, Pydantic, Loguru  
✅ **Monitoring**: Langfuse Callbacks  
✅ **Dockerized Pipeline**: Multi-stage AI data processing stack  

---

## 📁 Project Structure
```
📂 src/
 ├── core/
 │   ├── chatbot.py       # Main AI chatbot pipeline (LLM + Retrieval + Guard)
 │   ├── embedding.py     # FAISS/BM25 index generation
 │   ├── evaluation.py    # AI performance and accuracy checks
 │   ├── filters.py       # LLM Guard configurations for input/output scanning
 │── config.py            # Application settings & environment variables
 │
 ├── api/
 │   ├── routers/
 │   │   └── chat.py      # FastAPI routes for chatbot integration
 │   │── __init__.py      # API initializer
 │   └── main.py          # FastAPI app initialization
 │
 ├── data/
 │   ├── clean_data.py    # Data preprocessing scripts
 │
 ├── monitoring/
 │   └── monitoring.py    # Langfuse monitoring setup
 │
 ├── Dockerfile           # Containerization setup
 ├── docker-compose.yml   # Multi-container deployment setup
 ├── requirements.txt     # Python dependencies
 ├── README.md            # Project documentation (this file)
```

---

## 🏗 Setup & Installation
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/amine-akrout/parenting_assistant_rag.git
cd parenting_assistant_rag
```

### 2️⃣ **Create Virtual Environment & Install Dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3️⃣ **Set Up Environment Variables**
Create a `.env` file in the root directory:
```ini
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_DB=your_db_name
OPENAI_API_KEY=your-openai-key
FAISS_INDEX_PATH=./data/faiss_index.faiss
BM25_INDEX_PATH=./data/bm25_index.pkl
CROSS_ENCODER_MODEL_NAME=your-cross-encoder-model
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
```

### 4️⃣ **Run the API Server**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5️⃣ **Test the API**
Open **Swagger UI** to test endpoints:  
📌 `http://localhost:8000/docs`

---

## 🐳 Dockerized Workflow
This project includes a **Dockerized data pipeline** that prepares, indexes, and serves the chatbot API.  

### 1️⃣ **Start All Services**
```bash
docker-compose up --build
```
### 2️⃣ **Stop All Services**
```bash
docker-compose down
```

### **Docker Services Overview**
- **`langfuse-server`** → Logs chatbot requests/responses for monitoring
- **`db`** → PostgreSQL database for logging and indexing
- **`clean-data`** → Prepares and cleans datasets before indexing
- **`preprocess`** → Creates FAISS/BM25 indexes for fast retrieval
- **`chatbot-api`** → Runs the **FastAPI** chatbot backend


---

## 📈 Performance Monitoring (Langfuse)
Enable **Langfuse** for real-time monitoring:
```bash
export LANGFUSE_API_KEY=your-langfuse-key
```
View logs on **Langfuse Dashboard**.

---

## 🎯 Future Enhancements
✅ Add **multi-turn conversations** (memory support)  
✅ Implement **whisper model for speech-to-text**  
✅ Add **more retrieval sources (vector & hybrid search)**  
✅ Integrate **more LLM Guard features** (e.g., sentiment analysis)  
✅ Add **Unit Tests** for API endpoints and core functions


---

## 🤝 Contributing
Feel free to **open an issue** or **submit a pull request**.

---


