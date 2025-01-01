import streamlit as st
import openai
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
import requests
import fitz  # PyMuPDF
from langchain.schema import Document

# Load API Key
def load_api_key():
    with open('.config', 'r') as f:
        return f.read().strip()

# Function to download and extract text from PDF URL
def extract_text_from_pdf(url):
    try:
        # Download the PDF
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad HTTP status codes
        
        # Save the PDF to a temporary file
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        
        # Open the PDF file using PyMuPDF
        doc = fitz.open("temp.pdf")
        
        # Extract text from each page
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        return text
    
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Function to process URLs (including PDFs)
def process_urls(urls):
    if not urls:
        st.error("No URLs provided.")
        return None

    all_docs = []
    
    for url in urls:
        if url.endswith(".pdf"):
            # Handle PDF URL
            text = extract_text_from_pdf(url)
            if text:
                # Wrap the extracted text in a Document object
                all_docs.append(Document(page_content=text, metadata={"source": url}))
        else:
            # For non-PDF URLs, use the UnstructuredURLLoader
            loader = UnstructuredURLLoader([url])
            try:
                docs = loader.load()
                if docs:
                    all_docs.extend(docs)
            except Exception as e:
                st.error(f"Error loading URL: {e}")
    
    if not all_docs:
        st.error("No documents loaded from the provided URLs.")
        return None
    
    st.success(f"Loaded {len(all_docs)} documents from URLs.")
    
    # Now split the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = splitter.split_documents(all_docs)
    
    if not docs_split:
        st.error("No documents were split after loading.")
        return None
    
    # Use the new embedding models (text-embedding-3-small or text-embedding-3-large)
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Or "text-embedding-3-large" based on preference
    except Exception as e:
        st.error(f"Error accessing embedding model: {e}")
        return None

    try:
        vector_store = FAISS.from_documents(docs_split, embeddings)
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None
    
    # Save the FAISS index
    with open("faiss_store_openai.pkl", "wb") as f:
        pickle.dump(vector_store, f)
    
    return vector_store

# Load existing FAISS index
def load_faiss_index():
    if os.path.exists("faiss_store_openai.pkl"):
        with open("faiss_store_openai.pkl", "rb") as f:
            return pickle.load(f)
    return None

# Query the vector store
def query_vector_store(vector_store, query):
    result = vector_store.similarity_search_with_score(query, k=1)
    answer = result[0][0].page_content
    return answer

# Streamlit App
st.title("Scheme Research Application")
st.sidebar.header("Input Options")
input_option = st.sidebar.radio("Choose Input Type:", ("Enter URLs", "Upload URL File"))

if input_option == "Enter URLs":
    urls = st.sidebar.text_area("Enter URLs (one per line):").splitlines()
elif input_option == "Upload URL File":
    uploaded_file = st.sidebar.file_uploader("Upload Text File with URLs")
    # Modify the decoding to ignore invalid characters
    if uploaded_file:
        urls = uploaded_file.read().decode("utf-8", errors='ignore').splitlines()

# Process URLs
if st.sidebar.button("Process URLs"):
    if urls:
        try:
            vector_store = process_urls(urls)
            st.success("URLs processed and FAISS index created!")
        except ValueError as e:
            st.error(f"Error processing URLs: {e}")
    else:
        st.error("No URLs provided!")

# Load FAISS index
vector_store = load_faiss_index()

if vector_store:
    st.header("Query Section")
    query = st.text_input("Enter your query:")
    if st.button("Search"):
        answer = query_vector_store(vector_store, query)
        st.write("Answer:", answer)

# Footer
st.sidebar.info("Developed by lavish Gangwanii")
