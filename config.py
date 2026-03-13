

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load API key from .env file
load_dotenv()

# LLM Settings
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3

# Document Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_COUNT = 4

# Connect to Groq LLM
llm = ChatGroq(
    model=LLM_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=LLM_TEMPERATURE
)
