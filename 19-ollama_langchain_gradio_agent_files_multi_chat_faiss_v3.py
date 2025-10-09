# pip install -U langchain langchain-community langchain-huggingface langchain-ollama 
# pip install faiss-cpu gradio duckduckgo-search wikipedia
# pip install langgraph pypdf
# pip install torchvision torchaudio
# pip install torch==2.3.0 transformers==4.42.4 sentence-transformers==3.0.1 protobuf==3.20.3


# python ollama_langchain_gradio_agent_files_multi_chat_faiss_v3.py
# Summarize the main arguments from all documents.

import os
import gradio as gr
import shutil
from datetime import datetime

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.chains import ConversationalRetrievalChain

# ============================================
# ⚙️ Config
# ============================================
MODEL = "llama3"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "vectorstore"
VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")
DB_PATH = "chat_memory.sqlite"
SESSION_ID = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

SYSTEM_PROMPT = (
    "You are a concise and factual AI assistant. "
    "Answer based only on the uploaded documents, and cite filenames when possible."
)

# ============================================
# 🧩 Build / Load FAISS Vector Store
# ============================================
def list_indexed_files():
    """List files used in the FAISS index."""
    meta_path = os.path.join(VECTORSTORE_DIR, "sources.txt")
    if not os.path.exists(meta_path):
        return "No indexed files yet."
    with open(meta_path, "r", encoding="utf-8") as f:
        return f.read()

def build_or_load_vectorstore(files=None):
    """Load or create FAISS store."""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # If existing index and no new files → load
    if (not files or len(files) == 0) and os.path.exists(VECTORSTORE_PATH):
        print("✅ Loading existing FAISS vectorstore...")
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

    # Otherwise, build from scratch
    documents = []
    indexed_files = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for fpath in files or []:
        if fpath.endswith(".pdf"):
            loader = PyPDFLoader(fpath)
        elif fpath.endswith((".txt", ".md")):
            loader = TextLoader(fpath)
        else:
            continue
        docs = loader.load()
        documents.extend(splitter.split_documents(docs))
        indexed_files.append(os.path.basename(fpath))

    if len(documents) == 0:
        raise ValueError("No valid documents found to index.")

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    # Save list of indexed files
    with open(os.path.join(VECTORSTORE_DIR, "sources.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(indexed_files))

    print("💾 Saved FAISS vectorstore.")
    return vectorstore


def rebuild_vectorstore(files, chat_history):
    """Force rebuild FAISS from uploaded files."""
    if not files:
        chat_history.append({"role": "assistant", "content": "⚠️ Please upload at least one file first."})
        return chat_history, "No files uploaded."
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
    build_or_load_vectorstore(files)
    chat_history.append({"role": "assistant", "content": "✅ Index rebuilt successfully!"})
    return chat_history, list_indexed_files()

# ============================================
# 🧠 Persistent Memory
# ============================================
def reset_memory():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    return [], "🧹 Memory cleared successfully."

history = SQLChatMessageHistory(
    session_id=SESSION_ID,
    table_name="chat_history",
    connection_string=f"sqlite:///{DB_PATH}"
)

MEMORY = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=history,
    return_messages=True
)

# ============================================
# 🤖 Ollama LLM
# ============================================
llm = Ollama(model=MODEL)

# ============================================
# 💬 Chat Logic
# ============================================
def chat_with_agent(message, chat_history, files):
    """Chat handler using FAISS retriever + persistent memory."""
    try:
        vectorstore = build_or_load_vectorstore(files or [])
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"⚠️ Error: {str(e)}"})
        return chat_history, chat_history

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=MEMORY,
        verbose=False
    )

    prompt = f"{SYSTEM_PROMPT}\n\nUser: {message}"
    result = qa_chain.invoke({"question": prompt})
    response = result["answer"]

    # Optional: append file sources
    if "source_documents" in result and result["source_documents"]:
        files_used = {os.path.basename(d.metadata.get("source", "")) for d in result["source_documents"]}
        response += "\n\n📚 Sources:\n" + "\n".join(f"- {f}" for f in files_used)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    return chat_history, chat_history

# ============================================
# 🧰 Gradio UI
# ============================================
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🦙 Ollama Multi-File Chat Agent — v3.1 Stable")

    with gr.Row():
        chatbot = gr.Chatbot(label="AI Chat", type="messages", height=400)
        with gr.Column():
            file_input = gr.File(label="📁 Upload Files", file_count="multiple", type="filepath")
            index_display = gr.Textbox(label="📂 Indexed Files", interactive=False)
            rebuild = gr.Button("🔄 Rebuild Index")
            clear_memory = gr.Button("🧹 Clear Memory")

    msg = gr.Textbox(label="💬 Ask something about your files")
    submit = gr.Button("Send 🚀")

    submit.click(fn=chat_with_agent, inputs=[msg, chatbot, file_input], outputs=[chatbot, chatbot])
    rebuild.click(fn=rebuild_vectorstore, inputs=[file_input, chatbot], outputs=[chatbot, index_display])
    clear_memory.click(fn=reset_memory, outputs=[chatbot, index_display])

print("🚀 Running Ollama FAISS RAG v3.1 → http://127.0.0.1:7860")
demo.launch()
