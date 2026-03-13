from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import llm, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_COUNT

def process_pdf(pdf_path):
    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    
    # Convert chunks to embeddings and store in FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

def create_conversation_chain(vector_store):
    # Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_kwargs={"k": RETRIEVAL_COUNT}
    )
    
    # Create prompt template
    prompt = PromptTemplate.from_template("""
    You are a helpful assistant that answers questions based on the provided document context only.
    If the answer is not in the context, say "I could not find the answer to your question in the uploaded document."
    Never make up information.
    
    Context: {context}
    
    Chat History: {chat_history}
    
    Question: {question}
    
    Answer:""")
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    return chain, retriever

def get_answer(conversation_chain, question, chat_history):
    # Check if question is empty
    if not question.strip():
        return "Please ask a question about the document."
    
    try:
        chain, retriever = conversation_chain
        
        # Format chat history as string
        history_text = ""
        for human, assistant in chat_history:
            history_text += f"Human: {human}\nAssistant: {assistant}\n"
        
        # Get relevant documents
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Get answer using correct input format
        answer = chain.invoke({
            "context": context,
            "chat_history": history_text,
            "question": question
        })
        
        return answer
    
    except Exception as e:
        return f"Error: {str(e)}"
