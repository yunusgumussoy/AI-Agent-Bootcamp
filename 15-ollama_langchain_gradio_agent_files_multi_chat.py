# pip install -U langchain langchain-community langchain-huggingface langchain-ollama 
# pip install faiss-cpu gradio duckduckgo-search wikipedia
# pip install langgraph pypdf
# pip install torchvision torchaudio
# pip install torch==2.3.0 transformers==4.42.4 sentence-transformers==3.0.1 protobuf==3.20.3


# python ollama_langchain_gradio_agent_files_multi_chat.py
# Summarize the main arguments from all documents.


import os
import gradio as gr
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import SQLChatMessageHistory

# ==============================
# CONFIG
# ==============================
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3"  # or "mistral", "phi3", etc.
VECTORSTORE = None  # will hold the FAISS index
# MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ============================================
# 🧠 Persistent Chat Memory (Step 1 upgrade)
# ============================================

# SQLite file will be created in working directory
DB_PATH = "chat_history.sqlite"
SESSION_ID = "main_session"

history = SQLChatMessageHistory(
    session_id=SESSION_ID,
    connection_string=f"sqlite:///{DB_PATH}" 
)

MEMORY = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=history,
    return_messages=True
)

# ==============================
# LLM + Embeddings
# ==============================
llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ==============================
# Vectorstore builder
# ==============================
def build_vectorstore_from_files(files):
    """Read and embed all uploaded files into FAISS vectorstore."""
    docs = []
    for file in files:
        file_path = file.name
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path)
        else:
            continue

        docs.extend(loader.load())

    if not docs:
        raise ValueError("No valid documents found. Please upload PDFs or text files.")

    return FAISS.from_documents(docs, embeddings)

# ==============================
# QA Chain builder
# ==============================
def get_qa_chain():
    global VECTORSTORE
    if VECTORSTORE is None:
        raise ValueError("No documents loaded. Please upload files first.")

    retriever = VECTORSTORE.as_retriever(search_kwargs={"k": 3})
    template = """You are a helpful AI assistant. Use the provided context to answer the question.
If you don't know, say you don't know.

Context:
{context}

Question:
{question}
"""
    prompt = PromptTemplate.from_template(template)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=MEMORY,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False
    )

# ==============================
# Core agent logic
# ==============================
def chat_with_agent(message, chat_history, files):
    """Handles user message and file uploads."""
    global VECTORSTORE

    # Build vectorstore on first upload
    if files and VECTORSTORE is None:
        VECTORSTORE = build_vectorstore_from_files(files)
        return "📄 Files uploaded and processed successfully. Now you can ask questions."

    if VECTORSTORE is None:
        return "Please upload at least one file to start."

    qa_chain = get_qa_chain()
    result = qa_chain.invoke({"question": message})
    return result["answer"]

# ==============================
# Gradio interface logic
# ==============================
def respond(message, chat_history, files):
    """Wraps chat interaction for Gradio."""
    try:
        reply = chat_with_agent(message, chat_history, files)
    except Exception as e:
        reply = f"⚠️ Agent error: {e}"

    # Return in new Gradio message format
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply}
    ]
    return chat_history

# ==============================
# Launch the app
# ==============================
if __name__ == "__main__":
    print("🚀 Launching Ollama Multi-File Chat Agent at http://127.0.0.1:7861")

    gr.ChatInterface(
        fn=respond,
        type="messages",
        title="🧠 Local Multi-File Chat Agent (Llama3 + FAISS)",
        description="Upload multiple PDFs or text files and chat with your local Ollama model.",
        additional_inputs=[
            gr.File(label="Upload files", file_count="multiple", type="filepath")
        ],
    ).launch(share=False, server_port=7861)
