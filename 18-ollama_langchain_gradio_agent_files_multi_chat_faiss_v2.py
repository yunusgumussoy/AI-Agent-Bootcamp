# pip install -U langchain langchain-community langchain-huggingface langchain-ollama 
# pip install faiss-cpu gradio duckduckgo-search wikipedia
# pip install langgraph pypdf
# pip install torchvision torchaudio
# pip install torch==2.3.0 transformers==4.42.4 sentence-transformers==3.0.1 protobuf==3.20.3


# python ollama_langchain_gradio_agent_files_multi_chat_faiss_v2.py
# Summarize the main arguments from all documents.

import os
import gradio as gr
import shutil
from datetime import datetime

# LangChain & community imports
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
    "You are a helpful and concise research assistant. "
    "You answer based on the uploaded documents and clearly list your sources."
)

# ============================================
# 🧩 Build / Load FAISS Vector Store
# ============================================
def build_or_load_vectorstore(files):
    """Load existing FAISS store or build new one from uploaded files."""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if not files and os.path.exists(VECTORSTORE_PATH):
        print("✅ Loading existing FAISS vectorstore...")
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    if files:
        for file in files:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file)
            elif file.endswith((".txt", ".md")):
                loader = TextLoader(file)
            else:
                continue
            docs = loader.load()
            documents.extend(splitter.split_documents(docs))

    if not documents and os.path.exists(VECTORSTORE_PATH):
        print("✅ No new documents, loading existing FAISS vectorstore...")
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

    print(f"⚙️ Building FAISS store from {len(documents)} chunks...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print("💾 Saved new FAISS vectorstore.")
    return vectorstore


def rebuild_vectorstore(files, chat_history):
    """Force rebuild FAISS store from uploaded files and reload retriever."""
    if files:
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
        build_or_load_vectorstore(files)
        chat_history.append({"role": "assistant", "content": "✅ Index rebuilt successfully and reloaded."})
    else:
        chat_history.append({"role": "assistant", "content": "⚠️ Please upload files first."})
    return chat_history


# ============================================
# 🧠 Persistent Memory
# ============================================
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
    """Chat handler integrating FAISS retriever and persistent memory."""
    vectorstore = build_or_load_vectorstore(files or [])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=MEMORY,
        verbose=False
    )

    # Prepend system role
    prompt = f"{SYSTEM_PROMPT}\n\nUser: {message}"
    result = qa_chain.invoke({"question": prompt})
    response = result["answer"]

    # Add source citation if available
    sources = result.get("source_documents", [])
    if sources:
        cited = "\n\n📚 **Sources:**\n" + "\n".join(
            f"- {os.path.basename(doc.metadata.get('source', 'Unknown'))}" for doc in sources
        )
        response += cited

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})

    return chat_history, chat_history


# ============================================
# 🧰 Gradio UI
# ============================================
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🦙 Ollama Multi-File RAG Chat (FAISS + Persistent Memory)")

    chatbot = gr.Chatbot(label="AI Chat", type="messages")
    msg = gr.Textbox(label="Your Message")
    file_input = gr.File(label="Upload files", file_count="multiple", type="filepath")
    submit = gr.Button("Send 🚀")
    rebuild = gr.Button("🔄 Rebuild Index")

    submit.click(fn=chat_with_agent, inputs=[msg, chatbot, file_input], outputs=[chatbot, chatbot])
    rebuild.click(fn=rebuild_vectorstore, inputs=[file_input, chatbot], outputs=chatbot)

print("🚀 Launching Ollama Multi-File Chat Agent (FAISS v2) at http://127.0.0.1:7860")
demo.launch()
