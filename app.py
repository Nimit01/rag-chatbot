import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURATION ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")

VECTOR_DB_PATH = "vectorstore/db_faiss"
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

# --- LOAD THE PRE-BUILT DATABASE ---
def load_vector_database():
    """Loads the pre-built vector database."""
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(f"Vector database not found at {VECTOR_DB_PATH}. Please ensure it was included in the build.")
    
    print("Loading existing vector database...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Database loaded successfully.")
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

# --- MAIN GRADIO APP ---
print("Initializing chatbot...")
db = load_vector_database()
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
    theme="soft",
    chatbot=gr.Chatbot(type="messages") # Added to resolve deprecation warning
)

if __name__ == "__main__":
    ui.launch(share=True)