
# RAG Document Chatbot

An AI-powered document chatbot built with LangChain, Groq, and Streamlit. Upload any PDF document and have an intelligent conversation with it. The chatbot answers questions based strictly on the document content — never making up information.

## Live Demo
[Click here to try the app]() ← We will add this link after Streamlit deployment

## What It Does

Upload any PDF document and ask questions about it in plain English. The chatbot reads your document, understands it, and answers your questions accurately. It remembers the conversation history so you can ask natural follow up questions. If the answer is not in the document it honestly says so rather than inventing an answer.

## How It Works — RAG Pipeline

This app uses Retrieval Augmented Generation (RAG) — a technique that gives the LLM access to specific document content rather than relying on general training knowledge.

**Step 1 — Indexing**
When you upload a PDF the app extracts all text, splits it into chunks of 1000 characters with 200 character overlap, converts each chunk into numerical embeddings using HuggingFace all-MiniLM-L6-v2, and stores them in a FAISS vector database.

**Step 2 — Retrieval**
When you ask a question the app converts your question into embeddings and searches the FAISS database for the 4 most relevant chunks from your document.

**Step 3 — Generation**
The retrieved chunks plus your question are sent to Groq LLM which generates an accurate answer based strictly on your document content.

## Tech Stack

- **LangChain** — AI document processing and chain orchestration
- **Groq** — ultra fast LLM inference platform
- **Llama 3.3 70B** — large language model by Meta
- **FAISS** — vector database by Facebook for document search
- **HuggingFace** — sentence embeddings model
- **Streamlit** — web application framework
- **Python** — core programming language

## Project Structure
```
rag-chatbot/
├── config.py        # LLM connection and centralized settings
├── rag_engine.py    # RAG pipeline — PDF processing and Q&A
├── app.py           # Streamlit web interface
├── requirements.txt # Project dependencies
└── .gitignore       # Git ignore rules
```

## Key Settings

- Chunk Size: 1000 characters
- Chunk Overlap: 200 characters
- Retrieval Count: Top 4 most relevant chunks
- LLM Temperature: 0.3 (optimized for factual accuracy)
- Supported Format: PDF

## How to Run Locally

**1. Clone the repository**
```
git clone https://github.com/aamirqadeerdev/rag-chatbot.git
cd rag-chatbot
```

**2. Create and activate virtual environment**
```
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```
pip install -r requirements.txt
pip install sentence-transformers
pip install langchain-text-splitters
```

**4. Create .env file**
```
GROQ_API_KEY=your_groq_api_key_here
```

**5. Run the app**
```
streamlit run app.py
```

## Professional Documentation

This project includes professional client-ready documentation:

- **Chatbot Development Guidelines** — coding standards and best practices
- **AI Chatbot Compliance Checklist** — PIPEDA, GDPR, ISO 42001, ISO 27001, HIPAA, SOC 2

## Compliance Features

- No permanent storage of user documents
- Session data cleared after use
- API keys secured in environment variables
- Hallucination prevention implemented
- AI system clearly identified to users
- Honest responses when answer not found in document

## Author

Aamir Qadeer — Full Stack Developer and AI Engineer
- Available for Canadian remote opportunities
- Open to relocation to Canada
