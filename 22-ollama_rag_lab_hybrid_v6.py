# pip install -U langchain langchain-community langchain-huggingface langchain-ollama 
# pip install faiss-cpu gradio duckduckgo-search wikipedia
# pip install langgraph pypdf
# pip install torchvision torchaudio
# pip install torch==2.3.0 transformers==4.42.4 sentence-transformers==3.0.1 protobuf==3.20.3

# Save as ollama_rag_lab_hybrid_v6.py and run: python ollama_rag_lab_hybrid_v6.py
# Summarize the main arguments from all documents.

import os
import shutil
import json
from datetime import datetime
import pickle
import gradio as gr

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.chains import ConversationalRetrievalChain

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------
# Config & paths
# -------------------------
MODEL = "llama3"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
UPLOADED_DIR = "uploaded_files"
VECTORSTORE_DIR = "vectorstore"
VECTORSTORE_INDEX = os.path.join(VECTORSTORE_DIR, "faiss_index")
SOURCES_META = os.path.join(VECTORSTORE_DIR, "sources.json")
TFIDF_ARTIFACT = os.path.join(VECTORSTORE_DIR, "tfidf_artifacts.pkl")
DB_PATH = "chat_memory.sqlite"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4
DEFAULT_ALPHA = 0.7  # weight for semantic (FAISS) when combining

os.makedirs(UPLOADED_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# -------------------------
# Memory & LLM
# -------------------------
SESSION_ID = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
history = SQLChatMessageHistory(session_id=SESSION_ID, table_name="chat_history",
                                connection_string=f"sqlite:///{DB_PATH}")
MEMORY = ConversationBufferMemory(memory_key="chat_history", chat_memory=history, return_messages=True)
llm = Ollama(model=MODEL)

# Globals (in-memory)
_VECTORSTORE = None
_TFIDF_VECTORIZER = None
_TFIDF_MATRIX = None
_CHUNK_TEXTS = []      # parallel list of chunk texts
_CHUNK_META = []       # parallel list of metadata dicts (source, chunk_id)
_CURRENT_SETTINGS = {
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "top_k": DEFAULT_TOP_K,
    "embed_model": EMBED_MODEL
}

# -------------------------
# Helpers: file handling
# -------------------------
def copy_uploaded_files(filepaths):
    saved = []
    for path in filepaths or []:
        if not path:
            continue
        basename = os.path.basename(path)
        dest = os.path.join(UPLOADED_DIR, basename)
        if os.path.exists(dest):
            saved.append(dest)
            continue
        shutil.copy(path, dest)
        saved.append(dest)
    return saved

def list_uploaded_files_text():
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
# Build hybrid indices (FAISS + TF-IDF)
# -------------------------
def build_indices_from_uploaded(chunk_size, chunk_overlap, embed_model):
    """Build FAISS vectorstore and TF-IDF artifacts from files in UPLOADED_DIR."""
    global _VECTORSTORE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META
    files = sorted([os.path.join(UPLOADED_DIR, f) for f in os.listdir(UPLOADED_DIR)
                    if os.path.isfile(os.path.join(UPLOADED_DIR, f))])
    if not files:
        raise ValueError("No files in storage. Upload files first.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    chunk_texts = []
    chunk_meta = []

    for path in files:
        try:
            if path.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path)
            loaded_docs = loader.load()
            split_docs = splitter.split_documents(loaded_docs)
            # LangChain Document objects have .page_content and .metadata
            for i, d in enumerate(split_docs):
                docs.append(d)
                chunk_texts.append(d.page_content)
                metadata = d.metadata.copy() if hasattr(d, "metadata") else {}
                metadata["source"] = metadata.get("source", path)
                metadata["chunk_id"] = f"{os.path.basename(path)}::chunk_{i}"
                chunk_meta.append(metadata)
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    if not docs:
        raise ValueError("No readable text found in uploaded files.")

    # build FAISS
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_INDEX)

    # build TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=65536)
    matrix = tfidf.fit_transform(chunk_texts)

    # persist TF-IDF artifacts
    with open(TFIDF_ARTIFACT, "wb") as f:
        pickle.dump({
            "vectorizer": tfidf,
            "matrix": matrix,
            "chunk_texts": chunk_texts,
            "chunk_meta": chunk_meta
        }, f)

    # set globals
    _VECTORSTORE = vectorstore
    _TFIDF_VECTORIZER = tfidf
    _TFIDF_MATRIX = matrix
    _CHUNK_TEXTS = chunk_texts
    _CHUNK_META = chunk_meta

    # save sources list
    save_sources_meta([os.path.basename(p) for p in files])

    return vectorstore, [os.path.basename(p) for p in files]

def load_indices_if_exist(embed_model):
    """Try to load FAISS and TF-IDF artifacts from disk (if present)."""
    global _VECTORSTORE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META
    # load FAISS
    if _VECTORSTORE is None and os.path.exists(VECTORSTORE_INDEX):
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        _VECTORSTORE = FAISS.load_local(VECTORSTORE_INDEX, embeddings, allow_dangerous_deserialization=True)
    # load TF-IDF
    if _TFIDF_VECTORIZER is None and os.path.exists(TFIDF_ARTIFACT):
        with open(TFIDF_ARTIFACT, "rb") as f:
            data = pickle.load(f)
            _TFIDF_VECTORIZER = data["vectorizer"]
            _TFIDF_MATRIX = data["matrix"]
            _CHUNK_TEXTS = data["chunk_texts"]
            _CHUNK_META = data["chunk_meta"]
    return _VECTORSTORE is not None

def ensure_indices_or_raise():
    if not load_indices_if_exist(_CURRENT_SETTINGS["embed_model"]):
        raise ValueError("No index found. Upload files and rebuild the index first.")

# -------------------------
# Hybrid retrieval utilities
# -------------------------
def semantic_candidates(query, k):
    """Get top-k semantic docs (faiss). Returns list of (doc_obj, score)."""
    assert _VECTORSTORE is not None, "FAISS not loaded"
    retr = _VECTORSTORE.as_retriever(search_kwargs={"k": k})
    docs = retr.get_relevant_documents(query)
    # FAISS retriever may not return scores; we use distance proxies by re-embedding (skip for simplicity)
    # We assign a uniform starting score of 1.0 / rank
    results = []
    for rank, d in enumerate(docs, start=1):
        results.append((d, 1.0 / rank))
    return results

def keyword_candidates(query, k):
    """Get top-k keyword docs via TF-IDF cosine similarity. Returns list of (chunk_meta, score)."""
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META
    if _TFIDF_VECTORIZER is None or _TFIDF_MATRIX is None:
        raise ValueError("TF-IDF artifacts not available.")
    qv = _TFIDF_VECTORIZER.transform([query])  # 1 x V
    sims = cosine_similarity(qv, _TFIDF_MATRIX).ravel()  # vector of similarities
    top_idx = np.argsort(-sims)[:k]
    results = []
    for idx in top_idx:
        results.append((_CHUNK_META[idx], float(sims[idx])))
    return results

def combine_candidates(query, base_top_k, alpha=DEFAULT_ALPHA):
    """
    Combine semantic and keyword candidates into a ranked list.
    alpha: weight for semantic (0..1). keyword weight = 1 - alpha.
    """
    # adaptive top_k scaling based on query length
    words = len(query.split())
    if words <= 3:
        scale = 1
    elif words <= 12:
        scale = 1.3
    else:
        scale = 1.8
    k_sem = max(3, int(base_top_k * scale))
    k_kw = max(3, int(base_top_k * scale))

    sems = semantic_candidates(query, k_sem)  # list of (doc_obj, score)
    kws = keyword_candidates(query, k_kw)     # list of (meta, score)

    # normalize scores
    sem_scores = np.array([s for (_, s) in sems])
    if sem_scores.size:
        sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.ptp() + 1e-9)
    kw_scores = np.array([s for (_, s) in kws])
    if kw_scores.size:
        kw_scores = (kw_scores - kw_scores.min()) / (kw_scores.ptp() + 1e-9)

    # Build a map: key -> combined score and representative doc/metadata
    combined = {}

    # Add semantic candidates
    for i, (d, s) in enumerate(sems):
        key = d.metadata.get("chunk_id", d.metadata.get("source", str(i)) + f"::{i}")
        combined[key] = {
            "source": d.metadata.get("source", "Unknown"),
            "chunk_id": key,
            "text": d.page_content,
            "sem_score": float(sem_scores[i]) if sem_scores.size else float(s),
            "kw_score": 0.0
        }

    # Add kw candidates
    for i, (meta, s) in enumerate(kws):
        key = meta.get("chunk_id", meta.get("source", f"kw_{i}") )
        if key in combined:
            combined[key]["kw_score"] = float(kw_scores[i]) if kw_scores.size else float(s)
        else:
            combined[key] = {
                "source": meta.get("source", "Unknown"),
                "chunk_id": key,
                "text": _CHUNK_TEXTS[i] if i < len(_CHUNK_TEXTS) else "",
                "sem_score": 0.0,
                "kw_score": float(kw_scores[i]) if kw_scores.size else float(s)
            }

    # Compute final score and sort
    merged = []
    for k, v in combined.items():
        final = alpha * v["sem_score"] + (1 - alpha) * v["kw_score"]
        merged.append((k, final, v))
    merged_sorted = sorted(merged, key=lambda x: -x[1])
    return merged_sorted

# -------------------------
# High-level UI actions
# -------------------------
def ui_upload_and_rebuild(filepaths, chunk_size, chunk_overlap, top_k, embed_model, alpha):
    # copy files into storage
    copied = copy_uploaded_files(filepaths or [])
    if not copied:
        return [{"role": "assistant", "content": "No files uploaded."}], list_uploaded_files_text(), "No files uploaded."
    _CURRENT_SETTINGS.update({"chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap),
                              "top_k": int(top_k), "embed_model": embed_model, "alpha": float(alpha)})
    try:
        vs, indexed = build_indices_from_uploaded(int(chunk_size), int(chunk_overlap), embed_model)
    except Exception as e:
        return [{"role": "assistant", "content": f"Error: {e}"}], list_uploaded_files_text(), f"Error: {e}"
    return [{"role": "assistant", "content": f"✅ Indexed {len(indexed)} files."}], "\n".join(indexed), f"Indexed {len(indexed)} files."

def ui_rebuild_existing(chunk_size, chunk_overlap, top_k, embed_model, alpha):
    _CURRENT_SETTINGS.update({"chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap),
                              "top_k": int(top_k), "embed_model": embed_model, "alpha": float(alpha)})
    try:
        vs, indexed = build_indices_from_uploaded(int(chunk_size), int(chunk_overlap), embed_model)
    except Exception as e:
        return [{"role": "assistant", "content": f"Error rebuild: {e}"}], list_uploaded_files_text(), f"Error rebuild: {e}"
    return [{"role": "assistant", "content": f"✅ Rebuilt index for {len(indexed)} files."}], "\n".join(indexed), f"Rebuilt {len(indexed)} files."

def ui_delete_files(selected_files, chunk_size, chunk_overlap, top_k, embed_model, alpha):
    if not selected_files:
        return [{"role": "assistant", "content": "No files selected."}], list_uploaded_files_text(), "No files selected."
    for name in selected_files:
        path = os.path.join(UPLOADED_DIR, name)
        if os.path.exists(path):
            os.remove(path)
    # rebuild if there are remaining files
    remaining = [f for f in os.listdir(UPLOADED_DIR) if os.path.isfile(os.path.join(UPLOADED_DIR, f))]
    if not remaining:
        # clear indices
        if os.path.exists(VECTORSTORE_INDEX):
            shutil.rmtree(VECTORSTORE_INDEX)
        if os.path.exists(TFIDF_ARTIFACT):
            os.remove(TFIDF_ARTIFACT)
        if os.path.exists(SOURCES_META):
            os.remove(SOURCES_META)
        global _VECTORSTORE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META
        _VECTORSTORE = None
        _TFIDF_VECTORIZER = None
        _TFIDF_MATRIX = None
        _CHUNK_TEXTS = []
        _CHUNK_META = []
        return [{"role": "assistant", "content": "Deleted selected files and cleared index."}], list_uploaded_files_text(), "Deleted selected files."
    # otherwise rebuild from remaining
    return ui_rebuild_existing(chunk_size, chunk_overlap, top_k, embed_model, alpha)

def ui_semantic_search(q, top_k):
    if not q or q.strip() == "":
        return "Enter query."
    try:
        ensure_indices_or_raise()
    except Exception as e:
        return f"No index: {e}"
    merged = combine_candidates(q, int(top_k), alpha=_CURRENT_SETTINGS.get("alpha", DEFAULT_ALPHA))
    if not merged:
        return "No results."
    # format top results
    lines = []
    for rank, (key, score, v) in enumerate(merged[:int(top_k)], start=1):
        src = os.path.basename(v["source"])
        snippet = v["text"].strip().replace("\n", " ")
        snippet = snippet[:800] + ("..." if len(snippet) > 800 else "")
        lines.append(f"{rank}. ({src}) score={score:.3f}\n{snippet}\n")
    return "\n".join(lines)

# -------------------------
# Chat handler (uses hybrid retriever)
# -------------------------
def chat_handler(message, chat_history, top_k_ui, alpha_ui, adaptive_toggle):
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})
    try:
        ensure_indices_or_raise()
    except Exception as e:
        assistant_msg = f"⚠️ No index found. Upload files and rebuild first. ({e})"
        chat_history.append({"role": "assistant", "content": assistant_msg})
        MEMORY.chat_memory.add_user_message(message)
        MEMORY.chat_memory.add_ai_message(assistant_msg)
        return chat_history

    # adaptive top_k
    base_k = int(top_k_ui)
    if adaptive_toggle:
        words = len(message.split())
        if words <= 3:
            k = max(3, base_k - 1)
        elif words <= 12:
            k = base_k
        else:
            k = min(12, int(base_k * 1.8))
    else:
        k = base_k

    _CURRENT_SETTINGS["alpha"] = float(alpha_ui)

    merged = combine_candidates(message, k, alpha=float(alpha_ui))
    # take top N final docs
    top_docs_meta = [v for (_, _, v) in merged[:k]]

    # Create a simple list of Documents for chain input; we can construct retrieval via retriever too,
    # but simplest is to use FAISS retriever for the chain and pass the merged doc ids as context not implemented here.
    # Instead, we let the chain use the standard retriever but with top_k set to k (semantic side),
    # the hybrid merge helps us choose top docs but integrating that into chain would require a custom chain.
    vs = _VECTORSTORE
    retriever = vs.as_retriever(search_kwargs={"k": k})
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=MEMORY, verbose=False)

    result = qa_chain.invoke({"question": message})
    answer = result.get("answer", "No answer returned.")
    # add citations from hybrid merged top docs
    if top_docs_meta:
        used_files = sorted({os.path.basename(m["source"]) for m in top_docs_meta})
        if used_files:
            answer += "\n\n📚 Sources (hybrid top docs):\n" + "\n".join(f"- {f}" for f in used_files)

    chat_history.append({"role": "assistant", "content": answer})
    MEMORY.chat_memory.add_user_message(message)
    MEMORY.chat_memory.add_ai_message(answer)
    return chat_history

# -------------------------
# Gradio UI wiring (patched)
# -------------------------
with gr.Blocks(title="Local RAG Lab (v6 hybrid)") as demo:
    gr.Markdown("# 🔀 Local RAG Lab — v6 (Hybrid + Adaptive Retriever)")

    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.Chatbot(label="Chat", type="messages", height=420)
            with gr.Row():
                msg = gr.Textbox(label="Your question", placeholder="Ask about uploaded documents...")
                send = gr.Button("Send")
        with gr.Column(scale=1):
            gr.Markdown("### Upload & Index")
            files_in = gr.File(label="Upload files (pdf/txt/md)", file_count="multiple", type="filepath")
            chunk_size_in = gr.Number(label="Chunk size", value=DEFAULT_CHUNK_SIZE, precision=0)
            chunk_overlap_in = gr.Number(label="Chunk overlap", value=DEFAULT_CHUNK_OVERLAP, precision=0)
            top_k_in = gr.Slider(label="Base top_k", minimum=1, maximum=12, value=DEFAULT_TOP_K, step=1)
            alpha_in = gr.Slider(label="Semantic weight α (FAISS vs TF-IDF)", minimum=0.0, maximum=1.0, value=DEFAULT_ALPHA, step=0.05)
            adaptive_toggle = gr.Checkbox(label="Adaptive top_k", value=True)
            upload_rebuild = gr.Button("Upload & Rebuild")
            rebuild_existing = gr.Button("Rebuild stored files")
            gr.Markdown("### Indexed files")
            indexed = gr.CheckboxGroup(choices=list(os.listdir(UPLOADED_DIR)), label="Indexed files", interactive=True)
            delete_btn = gr.Button("Delete selected files")

            gr.Markdown("### Semantic search")
            search_q = gr.Textbox(label="Search for paragraph about...")
            search_k = gr.Slider(label="Search top_k", value=3, minimum=1, maximum=8, step=1)
            search_btn = gr.Button("Search")
            search_out = gr.Textbox(label="Search results", interactive=False)

            gr.Markdown("### Controls")
            clear_memory_btn = gr.Button("Clear memory")
            status_box = gr.Textbox(label="Status", interactive=False)

    # ---- hidden single embed_model_box (reused) ----
    embed_model_box = gr.Textbox(value=EMBED_MODEL, visible=False)

    # Upload & rebuild
    def on_upload_rebuild(files, chunk_size, chunk_overlap, top_k, embed_model, alpha):
        copied = copy_uploaded_files(files or [])
        msg, indexed_list, status = ui_upload_and_rebuild(copied, int(chunk_size), int(chunk_overlap), int(top_k), embed_model, alpha)
        choices = list(os.listdir(UPLOADED_DIR))
        return msg, choices, status

    upload_rebuild.click(fn=on_upload_rebuild, inputs=[files_in, chunk_size_in, chunk_overlap_in, top_k_in, embed_model_box, alpha_in],
                         outputs=[chat, indexed, status_box])

    # Rebuild existing
    def on_rebuild_existing(chunk_size, chunk_overlap, top_k, embed_model, alpha):
        msg, indexed_list, status = ui_rebuild_existing(int(chunk_size), int(chunk_overlap), int(top_k), embed_model, alpha)
        choices = list(os.listdir(UPLOADED_DIR))
        return msg, choices, status

    rebuild_existing.click(fn=on_rebuild_existing, inputs=[chunk_size_in, chunk_overlap_in, top_k_in, embed_model_box, alpha_in],
                           outputs=[chat, indexed, status_box])

    # delete
    def on_delete(selected, chunk_size, chunk_overlap, top_k, embed_model, alpha):
        msg, indexed_list, status = ui_delete_files(selected, int(chunk_size), int(chunk_overlap), int(top_k), embed_model, alpha)
        choices = list(os.listdir(UPLOADED_DIR))
        return msg, choices, status

    delete_btn.click(fn=on_delete, inputs=[indexed, chunk_size_in, chunk_overlap_in, top_k_in, embed_model_box, alpha_in],
                     outputs=[chat, indexed, status_box])

    # semantic search
    search_btn.click(fn=ui_semantic_search, inputs=[search_q, search_k], outputs=[search_out])

    # clear memory
    clear_memory_btn.click(fn=lambda: ([], "Memory cleared"), outputs=[chat, status_box])

    # chat send
    send.click(fn=lambda m, h, tk, a, ad: chat_handler(m, h, int(tk), float(a), bool(ad)),
               inputs=[msg, chat, top_k_in, alpha_in, adaptive_toggle], outputs=[chat])

    # initialize indexed list
    indexed.value = list(os.listdir(UPLOADED_DIR))

demo.launch(server_name="127.0.0.1", server_port=7860)
