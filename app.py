
import streamlit as st
import tempfile
import os
from rag_engine import process_pdf, create_conversation_chain, get_answer

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="",
    layout="centered"
)

st.title("RAG Document Chatbot")
st.markdown("Powered by **LangChain** + **Groq** + **FAISS**")
st.markdown("Upload a PDF document and ask questions about it.")
st.divider()


# Initialize session state variables
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# File upload section
with st.sidebar:
    st.subheader("Upload Your Document")
    uploaded_file = st.file_uploader(
        label="Choose a PDF file",
        type="pdf",
        help="Upload any PDF document to start chatting with it."
    )
    
    if uploaded_file is not None and not st.session_state.pdf_processed:
        with st.spinner("Processing your document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
            
            # Process PDF and create conversation chain
            vector_store = process_pdf(temp_path)
            st.session_state.conversation_chain = create_conversation_chain(vector_store)
            st.session_state.pdf_processed = True
            
            # Delete temporary file
            os.remove(temp_path)
            
        st.success("Document processed successfully! Start asking questions.")
    
    if st.session_state.pdf_processed:
        st.info("Document is ready to chat with.")
        if st.button("Upload New Document"):
            st.session_state.conversation_chain = None
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.session_state.pdf_processed = False
            st.rerun()


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.pdf_processed:
    question = st.chat_input("Ask a question about your document...")
    
    if question:
        # Add user message to display
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get answer from RAG engine
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(
                    st.session_state.conversation_chain,
                    question,
                    st.session_state.chat_history
                )
            st.markdown(answer)
        
        # Update chat history and messages
        st.session_state.chat_history.append((question, answer))
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info("Please upload a PDF document from the sidebar to start chatting.")




