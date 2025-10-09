# pip install -U langchain langchain-community langchain-huggingface langchain-ollama 
# pip install faiss-cpu gradio duckduckgo-search wikipedia
# pip install langgraph pypdf
# pip install torchvision torchaudio
# pip install torch==2.3.0 transformers==4.42.4 sentence-transformers==3.0.1 protobuf==3.20.3


# python ollama_langchain_gradio_agent_files_multi_chat_faiss2.py
# Summarize the main arguments from all documents.


import os
import gradio as gr
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import SQLChatMessageHistory
import shutil

# ============================================
# ⚙️ Config
# ============================================
MODEL = "llama3"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "vectorstore"
VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")
DB_PATH = "chat_memory.sqlite"
SESSION_ID = "ollama_multi_file_session"

# ============================================
# 🧠 Persistent FAISS Vector Store
# ============================================
def build_or_load_vectorstore(files):
    """Load existing FAISS store or build new one from uploaded files"""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    if os.path.exists(VECTORSTORE_PATH):
        print("✅ Loading existing FAISS vectorstore...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("⚙️ Building new FAISS vectorstore...")
        documents = []
        for file in files:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file.name)
            elif file.name.endswith((".txt", ".md")):
                loader = TextLoader(file.name)
            else:
                continue
            documents.extend(loader.load())

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        print("💾 Saved new FAISS vectorstore.")

    return vectorstore


def rebuild_vectorstore(files):
    """Force rebuild FAISS store from uploaded files"""
    if files:
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
        build_or_load_vectorstore(files)
        return "✅ Index rebuilt successfully!"
    return "⚠️ Please upload files first."

def rebuild_vectorstore(files, chat_history):
    """Force rebuild FAISS store from uploaded files"""
    if files:
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
        build_or_load_vectorstore(files)
        chat_history.append({"role": "assistant", "content": "✅ Index rebuilt successfully!"})
    else:
        chat_history.append({"role": "assistant", "content": "⚠️ Please upload files first."})
    return chat_history


# ============================================
# 🧩 Chat Memory (SQL persistent)
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
    vectorstore = build_or_load_vectorstore(files or [])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=MEMORY,
        verbose=False
    )

    result = qa_chain({"question": message})
    response = result["answer"]

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})

    return chat_history, chat_history


# ============================================
# 🧰 Gradio UI
# ============================================
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🦙 Ollama Multi-File Chat Agent with Persistent Memory & FAISS RAG")

    chatbot = gr.Chatbot(label="AI Chat", type="messages")
    msg = gr.Textbox(label="Your Message")
    file_input = gr.File(label="Upload files", file_count="multiple", type="filepath")
    submit = gr.Button("Send 🚀")
    refresh_button = gr.Button("🔄 Rebuild Index")

    submit.click(fn=chat_with_agent, inputs=[msg, chatbot, file_input], outputs=[chatbot, chatbot])
    refresh_button.click(fn=rebuild_vectorstore, inputs=[file_input, chatbot], outputs=chatbot)

print("🚀 Launching Ollama Multi-File Chat Agent with FAISS at http://127.0.0.1:7860")
demo.launch()
