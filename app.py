import sqlite_fix

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Constants
DB_DIRECTORY = "chroma_db_recent"
COLLECTION_NAME = "mix_faq_collection"

# Load environment variables
load_dotenv()

# Document loader
def load_documents(doc_path="./data/"):
    """
    Load documents from specified path.
    Supports PDF and text files.
    
    Args:
        document_path: Path to documents directory
    
    Returns:
        List of documents
    """
    documents = []
    
    # Create a Path object
    path = Path(doc_path)
    
    # Check if path exists
    if not path.exists():
        print(f"Path {doc_path} does not exist!")
        return documents
    
    # Load PDFs
    pdf_loader = DirectoryLoader(doc_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    # Load text files
    text_loader = DirectoryLoader(doc_path, glob="**/*.txt", loader_cls=TextLoader)
    # Load CSV files
    csv_loader = DirectoryLoader(doc_path, glob="**/*.csv", loader_cls=CSVLoader)
    
    # Load documents
    pdf_documents = pdf_loader.load() if list(path.glob("**/*.pdf")) else []
    text_documents = text_loader.load() if list(path.glob("**/*.txt")) else []
    csv_documents = csv_loader.load() if list(path.glob("**/*.csv")) else []
    
    # Combine documents
    documents = pdf_documents + text_documents + csv_documents
    
    print(f"Loaded {len(documents)} documents")
    return documents

# Text Splitter
def split_documents(documents, chunk_size=500, chunk_overlap=100):
    """
    Split documents into chunks for better embedding.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of document chunks
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # Split documents into chunks
    document_chunks = text_splitter.split_documents(documents)
    
    print(f"Split {len(documents)} documents into {len(document_chunks)} chunks")
    return document_chunks

# Vector Store
def initialize_vector_store(document_chunks=None, recreate=False):
    """
    Initialize or load the vector store.
    
    Args:
        document_chunks: List of document chunks to embed
        recreate: Whether to recreate the vector store
    
    Returns:
        Vector store
    """
    # Get OpenAI API key from environment
    MESOLITICA_API_KEY = os.environ.get("MESOLITICA_API_KEY")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="base",
        api_key=MESOLITICA_API_KEY,
        base_url="https://api.mesolitica.com"
    )
    
    # Check if vector store exists
    if os.path.exists(DB_DIRECTORY) and not recreate:
        print(f"Loading existing vector store from {DB_DIRECTORY}")
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=DB_DIRECTORY
        )
        return vector_store
    
    # Create vector store if it doesn't exist or recreate is True
    print("Creating new vector store")
    if document_chunks is None:
        raise ValueError("Document chunks must be provided to create a new vector store")
    
    # Create vector store
    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIRECTORY
    )

    # Then add documents in batches
    batch_size = 75
    for i in range(0, len(document_chunks), batch_size):
        batch = document_chunks[i:i + batch_size]
        vector_store.add_documents(documents=batch)
        print(f"Added batch {i//batch_size + 1}, documents {i} to {min(i+batch_size, len(document_chunks))}")
    
    return vector_store

# Retriever
def retrieve_documents(query, vector_store, top_k=5):
    """
    Retrieve relevant documents from vector store.
    
    Args:
        query: User query
        vector_store: Vector store to search
        top_k: Number of documents to retrieve
    
    Returns:
        List of retrieved documents
    """
    # Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    # Retrieve documents
    retrieved_docs = retriever.invoke(query)
    
    return retrieved_docs

# Retrieved docs formatter
def format_documents(docs):
    """
    Format retrieved documents for prompt.
    
    Args:
        docs: List of retrieved documents
    
    Returns:
        Formatted string
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        formatted_docs.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(formatted_docs)

# Create MaLLaM (LLM) for RAG Chain
def create_langchain_llm():
    """
    Create a LangChain LLM for MaLLaM.
    
    Returns:
        LangChain ChatOpenAI instance
    """
    # Get MaLLaM API key
    MESOLITICA_API_KEY = os.environ.get("MESOLITICA_API_KEY")
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="mallam-small",
        temperature=0.3,
        max_tokens=500,
        api_key=MESOLITICA_API_KEY,
        base_url="https://api.mesolitica.com"
    )
    
    return llm

# Prompt Template
def create_chat_prompt():
    """
    Create a prompt template for the chatbot.
    
    Returns:
        ChatPromptTemplate
    """
    template = """
    Anda adalah pembantu AI yang membantu menjawab soalan mengenai PADU (Pangkalan Data Utama Malaysia).
    
    Berikut adalah maklumat mengenai PADU:
    
    {context}
    
    Sejarah perbualan:
    {chat_history}
    
    Soalan pengguna terkini: {question}
    
    Jawab soalan dengan jelas berdasarkan maklumat di atas sahaja. Jika maklumat tidak mencukupi untuk menjawab soalan, sila nyatakan bahawa anda tidak mempunyai maklumat tersebut dan cadangkan pengguna untuk merujuk kepada laman web PADU di https://www.padu.gov.my atau menghubungi talian bantuan PADU.
    
    Jawapan:
    """
    
    return ChatPromptTemplate.from_template(template)

# Response Generator
def generate_response(query, vector_store, chat_history=""):
    """
    Generate a response to the user query.
    
    Args:
        query: User query
        vector_store: Vector store for retrieval
        chat_history: Chat history as a string
    
    Returns:
        Generated response
    """
    # Set up LLM
    llm = create_langchain_llm()
    
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, vector_store, top_k=5)
    
    # Format documents
    formatted_docs = format_documents(retrieved_docs)
    
    # Create prompt
    prompt = create_chat_prompt()
    
    # Create RAG chain
    rag_chain = (
        {"context": lambda _: formatted_docs, 
         "question": RunnablePassthrough(),
         "chat_history": lambda _: chat_history}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Generate response
    try:
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Maaf, saya tidak dapat menjawab soalan anda pada masa ini. Sila layari https://www.padu.gov.my untuk maklumat lanjut."

# Chat history handler
def format_chat_history(messages):
    """
    Format the chat history for the prompt.
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        Formatted chat history string
    """
    formatted_history = ""
    for message in messages:
        if message["role"] == "user":
            formatted_history += f"User: {message['content']}\n"
        else:
            formatted_history += f"Assistant: {message['content']}\n"
    
    return formatted_history

# Main Streamlit app
def main():
    """Main Streamlit app for the FAQ Chatbot"""
    # Set page title and favicon
    st.set_page_config(
        page_title="FAQ Chatbot",
        page_icon="🌙",
        layout="wide"
    )
    
    # Add header
    st.title("🌙 FAQ Chatbot with MaLLaM")
    st.write("Tanya soalan tentang Pangkalan Data Utama (PADU) Malaysia")
    
    # Reset chat history
    if st.button("Reset Chat History"):
        st.session_state.messages = []
        st.success("Chat history reset!")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Upload documents
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Upload PDF, CSV or Text files", type=["pdf", "csv", "txt"], accept_multiple_files=True)
        
        if uploaded_files:
            # Create data directory if it doesn't exist
            Path("./data").mkdir(exist_ok=True)
            
            # Save all uploaded files
            for uploaded_file in uploaded_files:
                with open(f"./data/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                st.success(f"File {uploaded_file.name} uploaded successfully!")
        
        # Process documents button
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Load and process documents
                documents = load_documents()
                
                if documents:
                    document_chunks = split_documents(documents)
                    st.session_state.vector_store = initialize_vector_store(document_chunks, recreate=True)
                    st.success(f"Processed {len(documents)} documents into {len(document_chunks)} chunks!")
                else:
                    st.error("No documents found to process!")
        
        
        # Clear vector store button
        if st.button("Clear Vector Store", help="This will completely remove the vector database"):
            import shutil
            # Delete the vector store directory if it exists
            if os.path.exists(DB_DIRECTORY):
                shutil.rmtree(DB_DIRECTORY)
                st.session_state.vector_store = None
                st.success("Vector store cleared successfully! You'll need to upload and process documents again.")
            else:
                st.info("No vector store found to clear.")
                
        # Remove uploaded files button
        if st.button("Remove Uploaded Files", help="This will delete all files in the upload directory"):
            data_directory = "./data/"
            if os.path.exists(data_directory):
                for filename in os.listdir(data_directory):
                    file_path = os.path.join(data_directory, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                st.success("All uploaded files have been removed.")
            else:
                st.info("No uploaded files found to remove.")
    
    # Initialize vector store
    try:
        vector_store = initialize_vector_store()
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        
        # If vector store doesn't exist, prompt user to upload documents
        st.info("Please upload documents to create a vector store.")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    if prompt := st.chat_input("Ask a question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
            
                # Format chat history (use last 5 messages to avoid context window issues)
                chat_history = format_chat_history(st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages)
                
                # Generate response
                response = generate_response(prompt, vector_store, chat_history)
                
                # Update placeholder with response
                message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
if __name__ == "__main__":
    main()