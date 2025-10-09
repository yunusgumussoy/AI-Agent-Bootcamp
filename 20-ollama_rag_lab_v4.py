# pip install -U langchain langchain-community langchain-huggingface langchain-ollama 
# pip install faiss-cpu gradio duckduckgo-search wikipedia
# pip install langgraph pypdf
# pip install torchvision torchaudio
# pip install torch==2.3.0 transformers==4.42.4 sentence-transformers==3.0.1 protobuf==3.20.3

# Save as ollama_rag_lab_v4.py and run: python ollama_rag_lab_v4.py
# Summarize the main arguments from all documents.

import os
import shutil
import json
from datetime import datetime
import gradio as gr

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.chains import ConversationalRetrievalChain

# -------------------------
# Configuration / Globals
# -------------------------
MODEL = "llama3"  # Ollama local model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # compact local embedder
UPLOADED_DIR = "uploaded_files"
VECTORSTORE_DIR = "vectorstore"
VECTORSTORE_INDEX = os.path.join(VECTORSTORE_DIR, "faiss_index")
SOURCES_META = os.path.join(VECTORSTORE_DIR, "sources.json")
DB_PATH = "chat_memory.sqlite"

# Defaults for chunking and retrieval
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4

# Initialize directories
os.makedirs(UPLOADED_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Persistent memory & LLM
SESSION_ID = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
history = SQLChatMessageHistory(session_id=SESSION_ID, table_name="chat_history",
                                connection_string=f"sqlite:///{DB_PATH}")
MEMORY = ConversationBufferMemory(memory_key="chat_history", chat_memory=history, return_messages=True)

llm = Ollama(model=MODEL)

# globals for in-memory retriever / vectorstore
_VECTORSTORE = None
_CURRENT_SETTINGS = {
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "top_k": DEFAULT_TOP_K,
    "embed_model": EMBED_MODEL
}

# -------------------------
# Utilities
# -------------------------
def copy_uploaded_files(filepaths):
    """Copy uploaded files into UPLOADED_DIR if not already there."""
    saved = []
    for path in filepaths:
        if not path:
            continue
        basename = os.path.basename(path)
        dest = os.path.join(UPLOADED_DIR, basename)
        # If it's already present, skip copy; else copy
        if os.path.exists(dest):
            saved.append(dest)
            continue
        shutil.copy(path, dest)
        saved.append(dest)
    return saved

def list_uploaded_files():
    files = sorted([f for f in os.listdir(UPLOADED_DIR) if os.path.isfile(os.path.join(UPLOADED_DIR, f))])
    if not files:
        return "No files uploaded yet."
    return "\n".join(files)

def save_sources_meta(file_list):
    with open(SOURCES_META, "w", encoding="utf-8") as f:
        json.dump(list(file_list), f)

def load_sources_meta():
    if os.path.exists(SOURCES_META):
        with open(SOURCES_META, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []

# -------------------------
# Vectorstore build/load
# -------------------------
def build_vectorstore_from_uploaded(chunk_size, chunk_overlap, embed_model):
    """
    Build and save FAISS vectorstore from files in UPLOADED_DIR,
    using given chunk_size and chunk_overlap.
    """
    global _VECTORSTORE
    # Collect files
    files = sorted([os.path.join(UPLOADED_DIR, f) for f in os.listdir(UPLOADED_DIR)
                    if os.path.isfile(os.path.join(UPLOADED_DIR, f))])
    if not files:
        raise ValueError("No uploaded files found. Upload files first.")

    # Load docs
    docs = []
    indexed_files = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for fpath in files:
        try:
            if fpath.lower().endswith(".pdf"):
                loader = PyPDFLoader(fpath)
            else:
                loader = TextLoader(fpath)
            loaded = loader.load()
            split = splitter.split_documents(loaded)
            docs.extend(split)
            indexed_files.append(os.path.basename(fpath))
        except Exception as e:
            print(f"Failed to load {fpath}: {e}")

    if not docs:
        raise ValueError("No readable text extracted from uploaded files.")

    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_INDEX)
    save_sources_meta(indexed_files)

    _VECTORSTORE = vectorstore
    return vectorstore, indexed_files

def load_vectorstore_if_exists(embed_model):
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    if os.path.exists(VECTORSTORE_INDEX):
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        _VECTORSTORE = FAISS.load_local(VECTORSTORE_INDEX, embeddings, allow_dangerous_deserialization=True)
        return _VECTORSTORE
    return None

def ensure_vectorstore(chunk_size=None, chunk_overlap=None, embed_model=None):
    """
    Ensure that a vectorstore exists: load if present, else raise.
    chunk_size etc are used only when building new index.
    """
    vs = load_vectorstore_if_exists(embed_model or _CURRENT_SETTINGS["embed_model"])
    if vs is None:
        raise ValueError("No existing index found. Upload files and rebuild the index.")
    return vs

# -------------------------
# High-level actions (used by UI)
# -------------------------
def action_upload_and_rebuild(uploaded_files, chunk_size, chunk_overlap, top_k, embed_model):
    """
    Copy uploaded files to storage and rebuild index using the chunk params.
    Returns (message, indexed_files_text, chat_history_message)
    """
    global _CURRENT_SETTINGS, _VECTORSTORE
    # copy files
    paths = [f for f in uploaded_files or []]
    copied = copy_uploaded_files(paths)
    if not copied:
        return "No files to upload.", list_uploaded_files(), [{"role": "assistant", "content": "No files uploaded."}]
    # update settings
    _CURRENT_SETTINGS.update({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "top_k": top_k, "embed_model": embed_model})
    # rebuild
    try:
        vectorstore, indexed_files = build_vectorstore_from_uploaded(chunk_size, chunk_overlap, embed_model)
    except Exception as e:
        return f"Error building index: {e}", list_uploaded_files(), [{"role": "assistant", "content": f"Error building index: {e}"}]
    # success
    return f"✅ Indexed {len(indexed_files)} files.", "\n".join(indexed_files), [{"role": "assistant", "content": f"✅ Index rebuilt for {len(indexed_files)} files."}]

def action_rebuild_from_existing(chunk_size, chunk_overlap, top_k, embed_model):
    """Rebuild using files already in uploaded_files (useful after deleting files)."""
    global _CURRENT_SETTINGS
    _CURRENT_SETTINGS.update({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "top_k": top_k, "embed_model": embed_model})
    files = [os.path.join(UPLOADED_DIR, f) for f in os.listdir(UPLOADED_DIR) if os.path.isfile(os.path.join(UPLOADED_DIR, f))]
    if not files:
        return "No files available in storage to rebuild.", list_uploaded_files(), [{"role": "assistant", "content": "No files to rebuild."}]
    try:
        vectorstore, indexed_files = build_vectorstore_from_uploaded(chunk_size, chunk_overlap, embed_model)
    except Exception as e:
        return f"Error rebuilding: {e}", list_uploaded_files(), [{"role": "assistant", "content": f"Error rebuilding: {e}"}]
    return f"✅ Rebuilt index for {len(indexed_files)} files.", "\n".join(indexed_files), [{"role": "assistant", "content": "✅ Rebuilt index successfully."}]

def action_delete_files(selected_files, chunk_size, chunk_overlap, top_k, embed_model):
    """Delete selected files from uploaded_files and rebuild index (if any files remain)."""
    if not selected_files:
        return "No file selected.", list_uploaded_files(), [{"role": "assistant", "content": "No file selected to delete."}]
    # delete selected
    for name in selected_files:
        path = os.path.join(UPLOADED_DIR, name)
        if os.path.exists(path):
            os.remove(path)
    remaining = sorted([f for f in os.listdir(UPLOADED_DIR) if os.path.isfile(os.path.join(UPLOADED_DIR, f))])
    # if no remaining files, remove vectorstore
    if not remaining:
        if os.path.exists(VECTORSTORE_INDEX):
            shutil.rmtree(VECTORSTORE_INDEX)
        if os.path.exists(SOURCES_META):
            os.remove(SOURCES_META)
        global _VECTORSTORE
        _VECTORSTORE = None
        return "Deleted selected files. No files remain; index removed.", list_uploaded_files(), [{"role": "assistant", "content": "Deleted files and cleared index."}]
    # else rebuild index from remaining
    return action_rebuild_from_existing(chunk_size, chunk_overlap, top_k, embed_model)

def action_semantic_search(query, top_k):
    """Return top_k chunks + filenames for a semantic search query."""
    if not query or query.strip() == "":
        return "Please enter a query to search."
    try:
        vs = ensure_vectorstore()
    except Exception as e:
        return f"Index missing: {e}"
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No results."
    # Format results: show excerpt and source filename
    out = []
    for i, d in enumerate(docs[:top_k], start=1):
        src = os.path.basename(d.metadata.get("source", "Unknown"))
        text = d.page_content.strip().replace("\n", " ")
        snippet = text[:600] + ("..." if len(text) > 600 else "")
        out.append(f"{i}. ({src}) {snippet}")
    return "\n\n".join(out)

# -------------------------
# Chat / RAG functions
# -------------------------
def chat_handler(message, chat_history, top_k):
    """
    Chat using conversational retrieval chain, top_k controls retriever k.
    Expects chat_history as list of role/content dicts (Gradio messages).
    Returns updated chat_history (list of role/content dicts).
    """
    # Add user message to memory and chat_history in expected format
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})

    try:
        vs = ensure_vectorstore()
    except Exception as e:
        assistant_msg = f"⚠️ No index found. Upload files and rebuild first. ({e})"
        chat_history.append({"role": "assistant", "content": assistant_msg})
        # persist to memory
        MEMORY.chat_memory.add_user_message(message)
        MEMORY.chat_memory.add_ai_message(assistant_msg)
        return chat_history

    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=MEMORY, verbose=False)

    # build prompt via the chain: Directly call .invoke for compatibility
    prompt = message  # the chain will use retriever + memory
    result = qa_chain.invoke({"question": prompt})
    answer = result.get("answer", "No answer returned.")
    # gather sources
    sources = result.get("source_documents", [])
    if sources:
        files_used = {os.path.basename(d.metadata.get("source", "")) for d in sources if d.metadata.get("source")}
        if files_used:
            answer += "\n\n📚 Sources:\n" + "\n".join(f"- {s}" for s in sorted(files_used))

    chat_history.append({"role": "assistant", "content": answer})
    # Persist to memory
    MEMORY.chat_memory.add_user_message(message)
    MEMORY.chat_memory.add_ai_message(answer)
    return chat_history

# -------------------------
# Gradio UI layout
# -------------------------
with gr.Blocks(title="Local RAG Lab (v4)", css="""
    .leftcol {min-width:480px}
    .rightcol {min-width:320px}
""") as demo:
    gr.Markdown("# 🧪 Local RAG Lab — v4 (offline)")
    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.Chatbot(label="Chat with your docs", type="messages", height=420)
            with gr.Row():
                msg = gr.Textbox(label="Type your question or instruction", placeholder="Ask about the uploaded documents...")
                send = gr.Button("Send")
        with gr.Column(scale=1):
            gr.Markdown("### Upload & Index")
            files_in = gr.File(label="Upload files (PDF / TXT / MD)", file_count="multiple", type="filepath")
            chunk_size_in = gr.Number(label="Chunk size", value=DEFAULT_CHUNK_SIZE, precision=0)
            chunk_overlap_in = gr.Number(label="Chunk overlap", value=DEFAULT_CHUNK_OVERLAP, precision=0)
            top_k_in = gr.Slider(label="Retriever top_k (affects chat & search)", minimum=1, maximum=12, value=DEFAULT_TOP_K, step=1)
            embed_model_in = gr.Textbox(label="Embedding model (HF)", value=EMBED_MODEL)
            upload_rebuild = gr.Button("Upload & Rebuild Index")
            rebuild_existing = gr.Button("Rebuild from stored files")
            gr.Markdown("### Indexed files")
            indexed_list = gr.CheckboxGroup(choices=[], label="Indexed files (select to delete)", interactive=True)
            delete_selected = gr.Button("Delete selected files (and rebuild)")

            gr.Markdown("### Quick tools")
            search_q = gr.Textbox(label="Semantic search (find para about X)")
            search_k = gr.Slider(label="Search top_k", minimum=1, maximum=8, value=3, step=1)
            search_btn = gr.Button("Search")

            gr.Markdown("### Controls")
            clear_memory_btn = gr.Button("Clear Memory (SQLite)")
            status_box = gr.Textbox(label="Status / Output", interactive=False)

    # Hookups
    def handle_upload_rebuild(files, chunk_size, chunk_overlap, top_k, embed_model):
        msg, indexed, bot_msg = action_upload_and_rebuild(files, int(chunk_size), int(chunk_overlap), int(top_k), embed_model)
        # update indexed_list choices
        choices = list(os.listdir(UPLOADED_DIR)) if os.path.exists(UPLOADED_DIR) else []
        return bot_msg, choices, indexed, msg

    upload_rebuild.click(fn=handle_upload_rebuild, inputs=[files_in, chunk_size_in, chunk_overlap_in, top_k_in, embed_model_in],
                         outputs=[chat, indexed_list, status_box, status_box])

    def handle_rebuild_existing(chunk_size, chunk_overlap, top_k, embed_model):
        msg, indexed, bot_msg = action_rebuild_from_existing(int(chunk_size), int(chunk_overlap), int(top_k), embed_model)
        choices = list(os.listdir(UPLOADED_DIR)) if os.path.exists(UPLOADED_DIR) else []
        return bot_msg, choices, indexed, msg

    rebuild_existing.click(fn=handle_rebuild_existing, inputs=[chunk_size_in, chunk_overlap_in, top_k_in, embed_model_in],
                           outputs=[chat, indexed_list, status_box, status_box])

    def handle_delete(selected, chunk_size, chunk_overlap, top_k, embed_model):
        msg, indexed, bot_msg = action_delete_files(selected, int(chunk_size), int(chunk_overlap), int(top_k), embed_model)
        choices = list(os.listdir(UPLOADED_DIR)) if os.path.exists(UPLOADED_DIR) else []
        return bot_msg, choices, indexed, msg

    delete_selected.click(fn=handle_delete, inputs=[indexed_list, chunk_size_in, chunk_overlap_in, top_k_in, embed_model_in],
                          outputs=[chat, indexed_list, status_box, status_box])

    def handle_search(q, k):
        return action_semantic_search(q, int(k))

    search_btn.click(fn=handle_search, inputs=[search_q, search_k], outputs=[status_box])

    def handle_clear_memory():
        restored, msg = reset_memory()
        return restored, msg

    clear_memory_btn.click(fn=handle_clear_memory, inputs=None, outputs=[chat, status_box])

    # Chat send
    def ui_send(msg_text, chat_history, top_k):
        return chat_handler(msg_text, chat_history, int(top_k))

    send.click(fn=ui_send, inputs=[msg, chat, top_k_in], outputs=[chat])

    # initialize UI display
    if os.path.exists(UPLOADED_DIR):
        initial_choices = list(os.listdir(UPLOADED_DIR))
    else:
        initial_choices = []
    indexed_list.value = initial_choices
    status_box.value = "Ready."

# Launch
demo.launch(server_name="127.0.0.1", server_port=7860)
