import os
import gradio as gr
from dotenv import load_dotenv

# Redirect Hugging Face cache to writable directory
os.environ['HF_HOME'] = '/app/cache'
os.environ['TRANSFORMERS_CACHE'] = '/app/cache/huggingface'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/app/cache/sentence_transformers'

# --- Load environment ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")

# --- LangChain imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# --- Configuration ---
VECTOR_DB_PATH = "vectorstore/db_faiss"
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
DOCUMENTS_PATH = "documents"  # Folder to store PDFs

def build_or_load_vector_database():
    """Builds or loads the FAISS vector database based on PDF documents."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME, model_kwargs={'device': 'cpu'})

    if os.path.exists(VECTOR_DB_PATH):
        print("✅ Loading existing vector database...")
        db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("📂 Vector database not found. Scanning documents...")
        if not os.path.exists(DOCUMENTS_PATH):
            raise FileNotFoundError(f"'{DOCUMENTS_PATH}' folder not found. Add your PDFs there.")
        loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            raise ValueError("No documents found in the 'documents' folder.")
        print(f"📄 Loaded {len(documents)} documents. Building vector DB...")
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(VECTOR_DB_PATH)
        print("✅ Vector database created and saved.")
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
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

# --- Start Application ---
print("🚀 Initializing chatbot...")
try:
    db = build_or_load_vector_database()
    qa_chain = setup_qa_chain(db)
    print("🤖 Chatbot is ready.")
except Exception as e:
    print(f"❌ Error during startup: {e}")
    raise

def chatbot_response(message, history):
    """The core chatbot function for Gradio."""
    try:
        result = qa_chain.invoke({"query": message})
        return result["result"]
    except Exception as e:
        return f"An error occurred: {e}"

# --- Gradio UI ---
ui = gr.ChatInterface(
    fn=chatbot_response,
    title="RAG Document Chatbot 🤖",
    description="Ask any question about the documents you've provided. The chatbot will use them as its knowledge base.",
    theme="soft",
    chatbot=gr.Chatbot(type="messages")
)

if __name__ == "__main__":
    ui.launch(share=True)
