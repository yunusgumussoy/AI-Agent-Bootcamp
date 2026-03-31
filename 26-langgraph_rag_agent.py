# =============================================================================
# 🤖 LangGraph RAG Agent — v1
# =============================================================================
#
# Bu proje v8 RAG sistemini bir "araç" olarak kullanan bir LangGraph agent'ı kurar.
#
# LangGraph'ın temel kavramları:
#   • StateGraph : Düğüm (node) ve kenarlardan (edge) oluşan yönlendirilmiş graf.
#   • State      : Tüm düğümlerin paylaştığı ortak veri yapısı (TypedDict).
#                  Her düğüm state'i okur, günceller ve bir sonraki düğüme bırakır.
#   • Node       : Bir Python fonksiyonu. State alır, güncel state döner.
#   • Edge       : Düğümler arası geçiş. Sabit (A→B) veya koşullu (A→? ) olabilir.
#   • Conditional Edge: Fonksiyon çalışır, dönen değere göre hangi düğüme
#                       gidileceğine karar verilir. Ajanın "düşünmesi" burada olur.
#
# Agent'ın düşünme döngüsü (ReAct pattern):
#   1. agent_node  : LLM soruyu okur. "RAG'a mı sorayım, yoksa biliyorum mu?" diye karar verir.
#   2. should_continue : LLM araç çağırdıysa → tool_node; cevap verdiyse → END
#   3. tool_node   : RAG aracını çalıştırır, sonucu state'e ekler.
#   4. agent_node  : RAG sonucunu görerek final cevabı üretir.
#
# Kurulum:
#   pip install langgraph langchain langchain-community langchain-huggingface
#   pip install langchain-ollama faiss-cpu sentence-transformers scikit-learn
#   pip install gradio pypdf numpy
#
# Çalıştırma:
#   python 26-langgraph_rag_agent.py
# =============================================================================

'''
ollama pull llama3.1
'''

import os
import json
import shutil
import pickle
from datetime import datetime
from typing import Annotated

import numpy as np
import gradio as gr

# --- LangGraph ---
from langgraph.graph import StateGraph, END          # Graf yapısı
from langgraph.graph.message import add_messages     # State'e mesaj ekleme yardımcısı

# --- LangChain ---
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool               # @tool dekoratörü
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.documents import Document

# --- Sklearn / Reranker ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

# Python'un TypedDict'i — AgentState için
from typing import TypedDict


# =============================================================================
# ⚙️  Konfigürasyon
# =============================================================================

MODEL        = "llama3.1"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

UPLOADED_DIR      = "uploaded_files"
VECTORSTORE_DIR   = "vectorstore"
VECTORSTORE_INDEX = os.path.join(VECTORSTORE_DIR, "faiss_index")
TFIDF_ARTIFACT    = os.path.join(VECTORSTORE_DIR, "tfidf_artifacts.pkl")

DEFAULT_TOP_K  = 4
DEFAULT_ALPHA  = 0.7
DEFAULT_RERANK = True

os.makedirs(UPLOADED_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# =============================================================================
# 🧠  Global Durum — RAG bileşenleri
# =============================================================================

_VECTORSTORE      = None
_TFIDF_VECTORIZER = None
_TFIDF_MATRIX     = None
_CHUNK_TEXTS      = []
_CHUNK_META       = []

# LLM — araç çağırma (tool calling) için bind_tools ile bağlı olacak
_BASE_LLM = ChatOllama(model=MODEL, temperature=0)

# Reranker
_RERANKER = CrossEncoder(RERANK_MODEL)


# =============================================================================
# 📁  RAG: Index Oluşturma ve Yükleme (v8 ile aynı)
# =============================================================================

def build_indices(chunk_size: int = 1000, chunk_overlap: int = 200):
    """Yüklenen dosyalardan FAISS + TF-IDF index oluşturur."""
    global _VECTORSTORE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META

    files = [
        os.path.join(UPLOADED_DIR, f)
        for f in os.listdir(UPLOADED_DIR)
        if os.path.isfile(os.path.join(UPLOADED_DIR, f))
    ]
    if not files:
        raise ValueError("Dosya yok. Önce PDF/TXT yükleyin.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_docs, chunk_texts, chunk_meta = [], [], []

    for path in sorted(files):
        try:
            loader = PyPDFLoader(path) if path.lower().endswith(".pdf") else TextLoader(path, encoding="utf-8")
            for i, doc in enumerate(splitter.split_documents(loader.load())):
                chunk_id = f"{os.path.basename(path)}::chunk_{i}"
                doc.metadata["chunk_id"] = chunk_id
                all_docs.append(doc)
                chunk_texts.append(doc.page_content)
                chunk_meta.append({
                    "source": doc.metadata.get("source", path),
                    "chunk_id": chunk_id,
                    "list_idx": len(chunk_texts) - 1,
                })
        except Exception as e:
            print(f"⚠️ {path}: {e}")

    if not all_docs:
        raise ValueError("Okunabilir metin bulunamadı.")

    embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_INDEX)

    tfidf  = TfidfVectorizer(ngram_range=(1, 2), max_features=65536)
    matrix = tfidf.fit_transform(chunk_texts)

    with open(TFIDF_ARTIFACT, "wb") as f:
        pickle.dump({
            "vectorizer": tfidf, "matrix": matrix,
            "chunk_texts": chunk_texts, "chunk_meta": chunk_meta,
        }, f)

    _VECTORSTORE      = vectorstore
    _TFIDF_VECTORIZER = tfidf
    _TFIDF_MATRIX     = matrix
    _CHUNK_TEXTS      = chunk_texts
    _CHUNK_META       = chunk_meta
    print(f"✅ Index hazır: {len(all_docs)} chunk.")
    return [os.path.basename(p) for p in files]


def load_indices_if_needed():
    """Uygulama başlarken varsa indexi diskten yükler."""
    global _VECTORSTORE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META
    if _VECTORSTORE is None and os.path.exists(VECTORSTORE_INDEX):
        emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _VECTORSTORE = FAISS.load_local(VECTORSTORE_INDEX, emb, allow_dangerous_deserialization=True)
    if _TFIDF_VECTORIZER is None and os.path.exists(TFIDF_ARTIFACT):
        with open(TFIDF_ARTIFACT, "rb") as f:
            data = pickle.load(f)
            _TFIDF_VECTORIZER = data["vectorizer"]
            _TFIDF_MATRIX     = data["matrix"]
            _CHUNK_TEXTS      = data["chunk_texts"]
            _CHUNK_META       = data["chunk_meta"]


# =============================================================================
# 🔍  RAG: Hybrid Retrieval (v8 ile aynı)
# =============================================================================

def _normalize(scores: np.ndarray) -> np.ndarray:
    rng = scores.max() - scores.min()
    return np.zeros_like(scores) if rng < 1e-9 else (scores - scores.min()) / rng


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K, alpha: float = DEFAULT_ALPHA) -> list[Document]:
    """
    Hybrid retrieval: FAISS (semantic) + TF-IDF (keyword) birleşimi.
    CrossEncoder ile reranking yapılır.
    Sonuç: LangChain Document listesi.
    """
    load_indices_if_needed()
    if _VECTORSTORE is None:
        return []

    k_fetch = max(top_k * 2, 6)

    # Semantic (FAISS)
    sem_results = _VECTORSTORE.similarity_search_with_score(query, k=k_fetch)
    sem_dists   = np.array([d for (_, d) in sem_results])
    sem_norms   = _normalize(-sem_dists)

    # Keyword (TF-IDF)
    qv    = _TFIDF_VECTORIZER.transform([query])
    sims  = cosine_similarity(qv, _TFIDF_MATRIX).ravel()
    top_i = np.argsort(-sims)[:k_fetch]
    kw_results = [(_CHUNK_META[i], float(sims[i]), int(i)) for i in top_i]
    kw_norms   = _normalize(np.array([s for (_, s, _) in kw_results]))

    # Birleştir
    combined = {}
    for i, (doc, _) in enumerate(sem_results):
        key = doc.metadata.get("chunk_id", f"s{i}")
        combined[key] = {"doc": doc, "sem": float(sem_norms[i]), "kw": 0.0}
    for i, (meta, _, idx) in enumerate(kw_results):
        key = meta.get("chunk_id", f"k{i}")
        if key in combined:
            combined[key]["kw"] = float(kw_norms[i])
        else:
            doc = Document(
                page_content=_CHUNK_TEXTS[idx] if idx < len(_CHUNK_TEXTS) else "",
                metadata={"source": meta.get("source", "?"), "chunk_id": key}
            )
            combined[key] = {"doc": doc, "sem": 0.0, "kw": float(kw_norms[i])}

    # Sırala ve rerank
    ranked = sorted(combined.values(), key=lambda v: -(alpha * v["sem"] + (1 - alpha) * v["kw"]))
    top_docs = [v["doc"] for v in ranked[:k_fetch]]

    # CrossEncoder rerank
    if top_docs:
        pairs  = [(query, d.page_content) for d in top_docs]
        scores = _RERANKER.predict(pairs)
        top_docs = [top_docs[i] for i in np.argsort(-scores)[:top_k]]

    return top_docs


# =============================================================================
# 🛠️  Araç Tanımı: @tool dekoratörü
# =============================================================================
# LangChain'in @tool dekoratörü fonksiyonu bir "tool" nesnesine dönüştürür.
# LLM bu aracı "isim + docstring" açıklamasını okuyarak ne zaman kullanacağına karar verir.
# Bu yüzden docstring çok önemli — LLM'in "araç seçim rehberi"dir.

@tool
def rag_search(query: str) -> str:
    """
    Yüklenen PDF ve metin belgelerinde hybrid arama yapar.
    Kullanıcının sorusu yüklenen belgelerle ilgiliyse bu aracı kullan.
    Genel bilgi sorularında veya belgelerle alakasız konularda kullanma.

    Args:
        query: Belgelerde aranacak soru veya anahtar kelimeler.

    Returns:
        İlgili belge parçalarını (chunk) içeren metin.
    """
    load_indices_if_needed()
    if _VECTORSTORE is None:
        return "Index bulunamadı. Önce belge yükleyip index oluşturun."

    docs = hybrid_search(query, top_k=DEFAULT_TOP_K, alpha=DEFAULT_ALPHA)
    if not docs:
        return "Belgelerde ilgili içerik bulunamadı."

    # Chunk'ları numaralı ve kaynaklı olarak formatla
    parts = []
    for i, doc in enumerate(docs, 1):
        src = os.path.basename(doc.metadata.get("source", "?"))
        parts.append(f"[Kaynak {i}: {src}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


# Gelecekte kolayca eklenebilecek başka araçlar:
# @tool
# def web_search(query: str) -> str:
#     """İnternette arama yapar. Belgede olmayan güncel bilgiler için kullan."""
#     ...
#
# @tool
# def summarize_topic(topic: str) -> str:
#     """Belgelerdeki belirli bir konuyu özetler."""
#     ...

# Aktif araç listesi — agent bu listedeki araçları kullanabilir
TOOLS = [rag_search]

# LLM'e araçları bağla — bind_tools, LLM'e "bu araçları çağırabilirsin" der
LLM_WITH_TOOLS = _BASE_LLM.bind_tools(TOOLS)


# =============================================================================
# 📋  Agent State — Tüm graf boyunca taşınan veri
# =============================================================================

class AgentState(TypedDict):
    """
    LangGraph'ın tüm düğümleri bu state'i paylaşır.

    'messages' alanındaki Annotated[..., add_messages] sözdizimi önemli:
    → add_messages: Yeni mesajları listenin başına eklemek yerine sona ekler.
       Bu sayede her node'un döndürdüğü mesajlar birikir, üzerine yazılmaz.

    Örnek akış:
      Başlangıç : [HumanMessage("PDF'de ne diyor?")]
      agent_node : + [AIMessage(tool_calls=[rag_search(query="...")])]
      tool_node  : + [ToolMessage(content="...chunk içeriği...")]
      agent_node : + [AIMessage(content="Belgeye göre şöyle: ...")]
    """
    messages: Annotated[list, add_messages]


# =============================================================================
# 🔵  Node 1: agent_node — LLM karar verici
# =============================================================================

# Sistem mesajı: Agent'ın kişiliğini ve ne zaman hangi aracı kullanacağını tanımlar
SYSTEM_PROMPT = SystemMessage(content="""Sen yüklenen belgeler hakkında soru yanıtlayan bir yapay zeka asistanısın.

Elinde 'rag_search' adında bir araç var. Bu araç yüklenen PDF ve metin dosyalarında arama yapar.

Karar kuralları:
- Kullanıcının sorusu yüklenen belgelerle ilgiliyse → rag_search aracını çağır.
- Genel bilgi sorusuysa (tarih, matematik, genel kültür vb.) → araç çağırmadan cevapla.
- Belgeleri aradıktan sonra kullanıcıya net ve kaynak belirterek cevap ver.
- Belgede yoksa bunu açıkça söyle.

Türkçe yanıt ver.""")


def agent_node(state: AgentState) -> AgentState:
    """
    Agent'ın ana düşünme düğümü.

    Ne yapar:
      1. Tüm mesaj geçmişini (+ sistem promptu) LLM'e gönderir.
      2. LLM ya bir araç çağırır (tool_call) ya da direkt cevap verir.
      3. LLM'in cevabını (AIMessage) state'e ekler.

    Bu düğüm defalarca çalışabilir:
      - İlk çalışmada: "RAG kullanmalı mıyım?" kararı verilir.
      - Tool çalıştıktan sonra: RAG sonucunu görür, final cevabı üretir.
    """
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = LLM_WITH_TOOLS.invoke(messages)
    # State'e yeni AIMessage'ı ekle (add_messages otomatik biriktiriyor)
    return {"messages": [response]}


# =============================================================================
# 🔀  Conditional Edge: should_continue — Yön karar verici
# =============================================================================

def should_continue(state: AgentState) -> str:
    """
    Bu fonksiyon LangGraph'a "bir sonraki düğüm hangisi?" diye sorar.

    Nasıl çalışır:
      - state'deki son mesajı (LLM'in son cevabını) inceler.
      - Eğer LLM bir araç çağırdıysa (tool_calls dolu) → "tools" düğümüne git.
      - Eğer araç çağırmadıysa → END (konuşma bitti).

    LangGraph bu fonksiyonun dönüş değerini edge haritasında arar:
      "tools" → tool_node
      END     → grafın sonu
    """
    last_message = state["messages"][-1]

    # AIMessage'da tool_calls varsa araç çağrısı yapılmış demektir
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"   # → tool_node'a git
    else:
        return END       # → grafı bitir


# =============================================================================
# 🟢  Node 2: tool_node — Araç çalıştırıcı
# =============================================================================

# Araçları isimlerine göre sözlüğe al (çağırma için)
TOOL_MAP = {t.name: t for t in TOOLS}


def tool_node(state: AgentState) -> AgentState:
    """
    LLM'in istediği aracı çalıştıran düğüm.

    Ne yapar:
      1. Son AIMessage'daki tool_calls listesini okur.
      2. Her araç çağrısı için ilgili fonksiyonu çalıştırır.
      3. Sonuçları ToolMessage olarak state'e ekler.

    ToolMessage nedir?
      → LangChain'de araç çıktısını taşıyan özel mesaj tipi.
         LLM bu mesajı görür ve cevabını buna göre üretir.
         tool_call_id, hangi tool_call'a yanıt olduğunu belirtir.
    """
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id   = tool_call["id"]

        if tool_name in TOOL_MAP:
            try:
                # Aracı çağır (örn: rag_search(query="..."))
                result = TOOL_MAP[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Araç hatası ({tool_name}): {e}"
        else:
            result = f"Bilinmeyen araç: {tool_name}"

        # Sonucu ToolMessage olarak paketle
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
                name=tool_name,
            )
        )

    return {"messages": tool_messages}


# =============================================================================
# 🏗️  Graf İnşası — StateGraph ile düğüm ve kenarları bağla
# =============================================================================

def build_graph():
    """
    LangGraph StateGraph'ı oluşturur ve derler.

    Graf yapısı:
      START → agent_node → (koşullu) → tool_node → agent_node (döngü)
                        ↘ END (araç gerekmiyorsa)

    compile() çağrısı grafı çalıştırılabilir hale getirir.
    Opsiyonel olarak checkpointer eklenirse konuşma state'i otomatik saklanır.
    """
    # Boş graf — AgentState şemasıyla başlat
    graph = StateGraph(AgentState)

    # Düğümleri ekle (isim → fonksiyon)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Başlangıç noktası: graf her zaman agent'dan başlar
    graph.set_entry_point("agent")

    # Koşullu kenar: agent çalıştıktan sonra should_continue'ya sor
    graph.add_conditional_edges(
        "agent",                          # kaynak düğüm
        should_continue,                  # hangi yönde gideceğini belirler
        {                                 # dönüş değeri → hedef düğüm haritası
            "tools": "tools",             # "tools" → tool düğümüne git
            END:     END,                 # END → grafı bitir
        }
    )

    # Tool çalıştıktan sonra her zaman agent'a geri dön (döngü)
    graph.add_edge("tools", "agent")

    # Grafı derle ve döndür
    return graph.compile()


# Global agent — uygulama başlarken bir kez oluştur
AGENT = build_graph()


# =============================================================================
# 💬  Agent Çalıştırıcı — Adım adım iz takibi ile
# =============================================================================

def run_agent(question: str, chat_history: list) -> tuple[list, str]:
    """
    Agent'ı çalıştırır ve her adımı kaydeder.

    LangGraph'ın .stream() metodu:
      Her düğüm çalıştığında bir event üretir.
      Böylece "agent düşünüyor → araç çağırıyor → cevap üretiyor"
      adımlarını gerçek zamanlı takip edebiliriz.

    Returns:
        chat_history : Güncellenmiş sohbet geçmişi
        trace_log    : Her adımın okunabilir özeti (Gradio'da göstermek için)
    """
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": question})

    # İlk state: sadece kullanıcı sorusu
    initial_state = {"messages": [HumanMessage(content=question)]}

    trace_lines = ["### 🔍 Agent Adım İzleme\n"]
    final_answer = ""
    step_count  = 0

    try:
        # .stream(): Her düğüm çalıştığında bir event dict döner
        # event = {"düğüm_adı": {"messages": [...]}}
        for event in AGENT.stream(initial_state):
            step_count += 1

            for node_name, node_output in event.items():
                messages = node_output.get("messages", [])
                for msg in messages:

                    if node_name == "agent":
                        # ── Agent düğümü çalıştı ──────────────────────────
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            # LLM araç çağırdı
                            for tc in msg.tool_calls:
                                q = tc["args"].get("query", str(tc["args"]))
                                trace_lines.append(
                                    f"**Adım {step_count} — Araç Çağrısı**\n"
                                    f"- Araç: `{tc['name']}`\n"
                                    f"- Sorgu: _{q}_\n"
                                )
                        elif hasattr(msg, "content") and msg.content:
                            # LLM final cevabı verdi
                            final_answer = msg.content
                            trace_lines.append(
                                f"**Adım {step_count} — Final Cevap üretildi**\n"
                                f"_{final_answer[:100]}..._\n" if len(final_answer) > 100
                                else f"**Adım {step_count} — Final Cevap üretildi**\n"
                            )

                    elif node_name == "tools":
                        # ── Tool düğümü çalıştı ───────────────────────────
                        snippet = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                        trace_lines.append(
                            f"**Adım {step_count} — Araç Sonucu** (`{msg.name}`)\n"
                            f"```\n{snippet}\n```\n"
                        )

    except Exception as e:
        final_answer = f"❌ Agent hatası: {e}"
        trace_lines.append(f"\n❌ Hata: {e}")

    if not final_answer:
        final_answer = "Agent bir cevap üretemedi."

    chat_history.append({"role": "assistant", "content": final_answer})
    trace_log = "\n".join(trace_lines)
    return chat_history, trace_log


# =============================================================================
# 🗂️  Dosya ve Index Yönetimi (Gradio için)
# =============================================================================

def ui_upload_and_build(files, chunk_size, chunk_overlap):
    """Dosyaları kopyalar ve index oluşturur."""
    if not files:
        return [{"role": "assistant", "content": "Dosya seçilmedi."}], [], "Dosya yok."
    for path in files:
        basename = os.path.basename(path)
        dest = os.path.join(UPLOADED_DIR, basename)
        if not os.path.exists(dest):
            shutil.copy(path, dest)
    try:
        indexed = build_indices(int(chunk_size), int(chunk_overlap))
    except Exception as e:
        return [{"role": "assistant", "content": f"❌ {e}"}], [], str(e)
    choices = [f for f in os.listdir(UPLOADED_DIR) if os.path.isfile(os.path.join(UPLOADED_DIR, f))]
    msg = f"✅ {len(indexed)} dosya indexlendi."
    return [{"role": "assistant", "content": msg}], choices, msg


def list_files():
    return sorted([f for f in os.listdir(UPLOADED_DIR) if os.path.isfile(os.path.join(UPLOADED_DIR, f))])


# =============================================================================
# 🖥️  Gradio Arayüzü
# =============================================================================

with gr.Blocks(title="LangGraph RAG Agent", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🤖 LangGraph RAG Agent")
    gr.Markdown(
        "Agent yüklenen belgeler hakkında soru-cevap yapar. "
        "Belgeyle ilgili sorularda RAG aracını çağırır; "
        "genel sorularda kendi bilgisiyle cevaplar."
    )

    with gr.Row():

        # ── Sol: Sohbet alanı ──────────────────────────────────────────────
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Sohbet", type="messages", height=440)
            with gr.Row():
                msg_box  = gr.Textbox(
                    label="Sorunuz",
                    placeholder="Belgeyle ilgili veya genel bir şey sorun...",
                    scale=4
                )
                send_btn = gr.Button("Gönder 🚀", scale=1, variant="primary")

            # Agent'ın adım izleme logu
            gr.Markdown("#### 🔍 Agent Adım İzleme")
            trace_box = gr.Markdown(
                value="*Bir soru sorduğunda agent'ın hangi adımları attığı burada görünecek.*"
            )

        # ── Sağ: Ayarlar ──────────────────────────────────────────────────
        with gr.Column(scale=1):

            with gr.Accordion("📁 Belge Yükleme", open=True):
                files_input   = gr.File(label="PDF / TXT yükle", file_count="multiple", type="filepath")
                chunk_size_in = gr.Number(label="Chunk boyutu", value=1000, precision=0)
                chunk_over_in = gr.Number(label="Chunk örtüşmesi", value=200, precision=0)
                upload_btn    = gr.Button("Yükle & Index Oluştur 🔨")

            with gr.Accordion("📋 Yüklü Dosyalar", open=True):
                file_list = gr.CheckboxGroup(
                    choices=list_files(), label="Indexlenmiş dosyalar", interactive=False
                )

            with gr.Accordion("ℹ️ Agent Hakkında", open=False):
                gr.Markdown("""
**Graf yapısı:**
```
START → agent → (araç gerekli mi?)
                   ↓ evet          ↓ hayır
                tools node        END
                   ↓
                agent (cevapla)
                   ↓
                END
```
**Araçlar:**
- `rag_search` — yüklenen belgeler
- *(eklenebilir: web_search, summarize)*

**State:** Her mesaj birikir;
agent önceki araç sonuçlarını görür.
                """)

            clear_btn  = gr.Button("🧹 Sohbeti Temizle")
            status_box = gr.Textbox(label="Durum", interactive=False)

    # ── Event Bağlantıları ─────────────────────────────────────────────────

    def on_upload(files, cs, co):
        msgs, choices, status = ui_upload_and_build(files, cs, co)
        return msgs, gr.CheckboxGroup(choices=choices, value=[]), status

    upload_btn.click(
        fn=on_upload,
        inputs=[files_input, chunk_size_in, chunk_over_in],
        outputs=[chatbot, file_list, status_box]
    )

    def on_send(message, history):
        if not message.strip():
            return history, "*Mesaj boş.*"
        return run_agent(message, history)

    send_btn.click(
        fn=on_send,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, trace_box]
    )
    msg_box.submit(
        fn=on_send,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, trace_box]
    )

    clear_btn.click(
        fn=lambda: ([], "*Temizlendi.*"),
        outputs=[chatbot, trace_box]
    )

    # Sayfa açılınca mevcut dosyaları göster
    demo.load(fn=list_files, outputs=[file_list])


demo.launch(server_name="127.0.0.1", server_port=7861)
# Not: v8 ile aynı anda çalıştırabilmek için port 7861 kullandık
