import os
import sys
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")

SOURCE_DOCUMENTS_DIR = "documents"
VECTOR_DB_PATH = "vectorstore/db_faiss"
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

# --- DATABASE AND QA CHAIN SETUP ---
def build_vector_database():
    """Builds or updates the vector database."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME, model_kwargs={'device': 'cpu'})
    
    if os.path.exists(VECTOR_DB_PATH):
        # Load the existing database
        print("Loading existing vector database...")
        db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Get a list of already indexed documents
        indexed_files = [os.path.basename(doc.metadata['source']) for doc in db.docstore._dict.values()]
        
        # Check for new documents
        new_documents_to_add = []
        print("Checking for new documents...")
        for filename in os.listdir(SOURCE_DOCUMENTS_DIR):
            if filename not in indexed_files:
                filepath = os.path.join(SOURCE_DOCUMENTS_DIR, filename)
                if os.path.isdir(filepath): continue
                try:
                    loader = PyMuPDFLoader(filepath) if filename.endswith(".pdf") else TextLoader(filepath)
                    documents = loader.load()
                    new_documents_to_add.extend(documents)
                    print(f"Found new document to add: {filename}")
                except Exception as e:
                    print(f"Warning: Could not load new file '{filename}'. Reason: {e}. Skipping.")

        # If new documents were found, add them to the database
        if new_documents_to_add:
            print("Adding new documents to the database...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(new_documents_to_add)
            db.add_documents(texts)
            db.save_local(VECTOR_DB_PATH)
            print("Database updated successfully!")
        else:
            print("No new documents found. Database is up to date.")
            
        return db

    else:
        # Build database from scratch if it doesn't exist
        print("Building new vector database from files...")
        # (This part is the same as before)
        all_documents = []
        for filename in os.listdir(SOURCE_DOCUMENTS_DIR):
            filepath = os.path.join(SOURCE_DOCUMENTS_DIR, filename)
            if os.path.isdir(filepath): continue
            try:
                loader = PyMuPDFLoader(filepath) if filename.endswith(".pdf") else TextLoader(filepath)
                documents = loader.load()
                all_documents.extend(documents)
                print(f"Successfully loaded {filename}")
            except Exception as e:
                print(f"Warning: Could not load file '{filename}'. Reason: {e}. Skipping.")
        
        if not all_documents:
            raise ValueError("Error: No documents were loaded. Check the 'documents' directory.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(all_documents)
        
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(VECTOR_DB_PATH)
        print("New vector database built successfully!")
        return db

def setup_qa_chain(db):
    """Sets up and returns the Retrieval QA chain."""
    retriever = db.as_retriever(search_kwargs={'k': 3})
    prompt_template = """
    Use the following pieces of context to answer the user's question. If you don't know the answer from the context, just say that you don't know. Provide a detailed and helpful answer based on the information given.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = GoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- MAIN GRADIO APP ---
print("Initializing chatbot...")
db = build_vector_database()
qa_chain = setup_qa_chain(db)
print("Chatbot is ready.")

def chatbot_response(message, history):
    """The core chatbot function for Gradio."""
    try:
        result = qa_chain.invoke({"query": message})
        return result["result"]
    except Exception as e:
        return f"An error occurred: {e}"

ui = gr.ChatInterface(
    fn=chatbot_response,
    title="RAG Document Chatbot 🤖",
    description="Ask any question about the documents you've provided. The chatbot will use them as its knowledge base.",
    theme="soft"
)

if __name__ == "__main__":
    ui.launch(share=True)