🏥 MediChat – Medical Document Assistant

# 🏥 MediChat – Medical Document Assistant

**MediChat** is a **Streamlit-based application** that enables users to:

- Upload **medical PDF documents**.
- Process them into a **vector database** for semantic search.
- Chat with an **AI assistant** to extract **insights, summaries, and medical diagnoses**.

This project demonstrates a practical implementation of a **Retrieval-Augmented Generation (RAG) pipeline** using:

- **LangChain**  
- **FAISS**  
- **HuggingFace embeddings**  
- **EuriAI LLMs**



🚀 Features
📂 Upload multiple PDFs of medical records or research papers.
🔎 Extract text from PDFs and split into smaller chunks.
🧠 Vectorize documents with FAISS and sentence-transformers for semantic search.
🤖 Chat with an AI assistant (EuriAI LLM) to get medical insights.
🎯 Context-aware answers grounded in your uploaded documents.
💬 Interactive Streamlit chat interface with message history.


📂 Project Structure
medichat/
│
├── app/                      # Core application logic
│   ├── __init__.py           # Package initializer
│   ├── chat_utils.py         # LLM integration (EuriAI chat model)
│   ├── config.py             # API key configuration
│   ├── pdf_utils.py          # PDF text extraction
│   ├── ui.py                 # Streamlit UI components (file uploader, etc.)
│   ├── vectorstore_utils.py  # Vector database creation & retrieval
│
├── example_data/             # Sample medical PDFs for testing
│
├── main.py                   # Streamlit entry point
├── requirements.txt          # Project dependencies


## ⚙️ Installation & Setup

Follow these steps to set up and run MediChat locally:

1. **Clone the Repository**
```bash
git clone https://github.com/<your-username>/medichat.git
cd medichat


# Create a Virtual Environment (Recommended)

python -m venv venv         # Create virtual environment
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows



#Install Dependencies

pip install -r requirements.txt

# Configure API Key
# Edit app/config.py and add your EuriAI API key:


EURI_API_KEY = "your_api_key_here"

#Run the Application

streamlit run main.py



✅ Features of this formatting:  
- Numbered steps for clear sequential guidance.  
- Commands and code are in code blocks for readability.  
- File names and key values are highlighted with backticks.  

If you want, I can now **combine all your sections** (Project Overview, Features, Key Modules, Installation, How It Works, Workflow, Requirements, Future Improvements, Author, Architecture) into **one complete README.md** ready to use.  

Do you want me to do that next?


## 🛠️ How It Works

1. **Upload PDFs** → Users can upload one or more medical PDF documents.
2. **Text Extraction** → Extracts raw text using `pypdf`.
3. **Chunking** → Splits text into smaller segments with `RecursiveCharacterTextSplitter`.
4. **Embedding & Indexing** → Converts text into embeddings using `HuggingFaceEmbeddings` and stores them in a **FAISS vector database**.
5. **Semantic Search** → Retrieves the most relevant chunks based on user queries.
6. **LLM Response** → Sends query + retrieved context to an **EuriAI GPT model (`gpt-4.1-nano`)** for a grounded, context-aware answer.
7. **Interactive Chat** → Streamlit displays a chat-like interface where user and assistant messages are stored in session state.



## 📜 Key Modules

### 🔹 `chat_utils.py`
- Creates the EuriAI chat model (`get_chat_model`).
- Defines function `ask_chat_model` to send prompts and retrieve responses.

### 🔹 `pdf_utils.py`
- Extracts text from PDF files using `PdfReader`.

### 🔹 `ui.py`
- Provides a simple Streamlit file uploader for PDFs.

### 🔹 `vectorstore_utils.py`
- Creates FAISS vector index from text chunks.
- Retrieves relevant documents for queries.
- Includes a fallback method in case FAISS fails.

### 🔹 `main.py`
- Defines the Streamlit app workflow:
  - PDF upload in sidebar.
  - Document processing pipeline.
  - Chat interface with timestamped messages.
  - AI-powered responses using context from documents.


## 📊 Example Workflow

1. **Upload** one or more medical research papers.
2. **Click** "Process Documents" → Extract & index text.
3. **Ask questions** such as:
   - “Summarize the diagnosis mentioned in the report.”
   - “What medications are recommended for this patient?”
   - “Does this report mention hypertension?”
4. **Receive** precise, document-grounded answers with medical insights.

---

## ✅ Requirements

- Python 3.9+
- Streamlit
- FAISS
- HuggingFace sentence-transformers
- LangChain
- EuriAI API key

---

## 🔮 Future Improvements

- Add **chat memory persistence** across sessions.
- Support for **other document formats** (Word, TXT).
- Enhanced **medical terminology handling**.
- Option to **export chat history** as PDF.

---

## 👨‍💻 Author

### Developed by **Shewan Dagne**  
🤖 Powered by **EuriAI & LangChain**  
- Built with ❤️ using **RAG (Retrieval-Augmented Generation)**

## 📊 MediChat Architecture (RAG Pipeline)


flowchart TD
    A[📂 PDF Upload] --> B[🔎 Text Extraction with PyPDF]
    B --> C[✂️ Text Splitting (RecursiveCharacterTextSplitter)]
    C --> D[🧠 Embeddings (Sentence-Transformers)]
    D --> E[📦 FAISS Vector Store]
    E --> F[🔍 Retrieve Relevant Chunks]
    F --> G[🤖 EuriAI LLM (gpt-4.1-nano)]
    G --> H[💬 Answer Displayed in Streamlit Chat]



## 📊 How MediChat Works (RAG Pipeline)


flowchart TD
    A[📂 PDF Upload] --> B[🔎 Text Extraction with PyPDF]
    B --> C[✂️ Text Splitting (RecursiveCharacterTextSplitter)]
    C --> D[🧠 Embeddings (Sentence-Transformers)]
    D --> E[📦 FAISS Vector Store]
    E --> F[🔍 Retrieve Relevant Chunks]
    F --> G[🤖 EuriAI LLM (gpt-4.1-nano)]
    G --> H[💬 Answer Displayed in Streamlit Chat]
