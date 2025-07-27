import os
import gradio as gr
from dotenv import load_dotenv

# --- Load environment ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Error: GOOGLE_API_KEY not found. Please set it as a secret in your deployment environment.")

# --- LangChain imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuration ---
VECTOR_DB_PATH = "vectorstore/db_faiss"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Using the full name is best practice
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

def load_prebuilt_database():
    """Loads the pre-built FAISS vector database."""
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(f"Vector database not found at {VECTOR_DB_PATH}. Ensure it was included in the build.")
    
    print("✅ Loading existing vector database...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ Database loaded successfully.")
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
db = load_prebuilt_database()
qa_chain = setup_qa_chain(db)
print("🤖 Chatbot is ready.")

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

# --- Production Launch ---
if __name__ == "__main__":
    # Get the port from the environment variable, default to 7860
    port = int(os.environ.get("PORT", 7860))
    # Launch the app to be accessible on the network
    ui.launch(server_name="0.0.0.0", server_port=port)