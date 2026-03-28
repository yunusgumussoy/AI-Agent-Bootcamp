# =============================================================================
# 🔀 Local RAG Lab — v7 (Hybrid + Reranker + LCEL)
# =============================================================================
#
# v6'dan v7'ye yapılan değişiklikler:
#   ✅ BUG #1 — Çift bellek yazımı düzeltildi
#   ✅ BUG #2 — HybridRetriever oluşturuldu; chain artık gerçekten hybrid sonuçları kullanıyor
#   ✅ BUG #3 — keyword_candidates'taki metin index hatası düzeltildi
#   ✅ BUG #4 — Clear Memory artık SQLite'ı da temizliyor
#   ✅ FAISS'te similarity_search_with_score kullanıldı (reciprocal rank yerine gerçek skor)
#   ✅ Deprecated API'ler güncellendi (ChatOllama, .invoke(), LCEL zinciri)
#   ✅ np.ptp() yerine .max()-.min() kullanıldı
#   ✅ Cross-encoder reranker eklendi (hybrid merge sonrası)
#
# Kurulum:
#   pip install langchain langchain-community langchain-huggingface langchain-ollama
#   pip install faiss-cpu gradio sentence-transformers
#   pip install langchain-core pypdf
#   pip install torch transformers sentence-transformers
#   pip install scikit-learn numpy
#
# Çalıştırma:
#   python 24-ollama_rag_lab_hybrid_v7.py
# =============================================================================

import os
import shutil
import json
import pickle
from datetime import datetime

import numpy as np
import gradio as gr

# --- LangChain bileşenleri ---
from langchain_ollama import ChatOllama                          # ✅ Güncellendi: langchain_community yerine
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.retrievers import BaseRetriever             # ✅ Yeni: Custom retriever için
from langchain_core.documents import Document                   # ✅ Yeni: Document objesi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter          # LCEL'de dict'ten tek alan çekmek için
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Sklearn / Reranker ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder                  # ✅ Yeni: Reranker modeli


# =============================================================================
# ⚙️  Konfigürasyon & Sabitler
# =============================================================================

MODEL        = "llama3"                                   # Ollama'da yüklü model adı
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding modeli
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"    # Reranker modeli (ilk çalışmada indirilir)

UPLOADED_DIR      = "uploaded_files"                      # Yüklenen dosyaların saklandığı klasör
VECTORSTORE_DIR   = "vectorstore"                         # Index dosyalarının klasörü
VECTORSTORE_INDEX = os.path.join(VECTORSTORE_DIR, "faiss_index")
SOURCES_META      = os.path.join(VECTORSTORE_DIR, "sources.json")
TFIDF_ARTIFACT    = os.path.join(VECTORSTORE_DIR, "tfidf_artifacts.pkl")
DB_PATH           = "chat_memory.sqlite"                  # Konuşma geçmişi veritabanı

# Varsayılan parametreler (UI'dan değiştirilebilir)
DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K         = 4
DEFAULT_ALPHA         = 0.7   # Semantic (FAISS) ağırlığı; 1-alpha = TF-IDF ağırlığı
DEFAULT_RERANK        = True  # Reranker varsayılan açık mı?

# Klasörlerin varlığını garantile
os.makedirs(UPLOADED_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# =============================================================================
# 🧠  Global Durum (in-memory; uygulama yeniden başlarsa diskten yüklenir)
# =============================================================================

_VECTORSTORE      = None   # FAISS vektör deposu
_TFIDF_VECTORIZER = None   # TF-IDF dönüştürücü
_TFIDF_MATRIX     = None   # Chunk'ların TF-IDF matrisi
_CHUNK_TEXTS      = []     # Her chunk'ın ham metni (TF-IDF ile paralel)
_CHUNK_META       = []     # Her chunk'ın metadata'sı (source, chunk_id)

# Aktif ayarlar sözlüğü; UI slider/input değerleri buraya yansıtılır
_CURRENT_SETTINGS = {
    "chunk_size":    DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "top_k":         DEFAULT_TOP_K,
    "embed_model":   EMBED_MODEL,
    "alpha":         DEFAULT_ALPHA,
    "rerank":        DEFAULT_RERANK,
}

# LLM — ChatOllama: streaming destekli, modern API
llm = ChatOllama(model=MODEL, temperature=0)

# Reranker modeli (cross-encoder): ilk kullanımda HuggingFace'den indirilir
_RERANKER = CrossEncoder(RERANK_MODEL)

# Konuşma geçmişi: her oturum için benzersiz session_id
SESSION_ID = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
_CHAT_HISTORY_STORE = SQLChatMessageHistory(
    session_id=SESSION_ID,
    table_name="chat_history",
    connection_string=f"sqlite:///{DB_PATH}"
)


# =============================================================================
# 📁  Dosya Yönetimi
# =============================================================================

def copy_uploaded_files(filepaths: list) -> list:
    """
    Gradio'nun geçici konumundaki dosyaları kalıcı UPLOADED_DIR'e kopyalar.
    Aynı isimli dosya zaten varsa tekrar kopyalamaz.
    """
    saved = []
    for path in filepaths or []:
        if not path:
            continue
        basename = os.path.basename(path)
        dest = os.path.join(UPLOADED_DIR, basename)
        if not os.path.exists(dest):
            shutil.copy(path, dest)
        saved.append(dest)
    return saved


def list_uploaded_files() -> list:
    """UPLOADED_DIR içindeki dosya adlarını liste olarak döner."""
    return sorted([
        f for f in os.listdir(UPLOADED_DIR)
        if os.path.isfile(os.path.join(UPLOADED_DIR, f))
    ])


def save_sources_meta(file_list: list):
    """İndekslenen dosya listesini JSON olarak diske kaydeder."""
    with open(SOURCES_META, "w", encoding="utf-8") as f:
        json.dump(list(file_list), f)


# =============================================================================
# 🗂️  Index Oluşturma: FAISS + TF-IDF
# =============================================================================

def build_indices(chunk_size: int, chunk_overlap: int, embed_model: str):
    """
    UPLOADED_DIR'deki tüm dosyaları okur, chunk'lara böler ve
    iki ayrı index oluşturur:
      1. FAISS   — anlamsal (semantic) arama için
      2. TF-IDF  — anahtar kelime (keyword) araması için

    Her iki index de diske kaydedilir; bir sonraki başlatmada tekrar build'e gerek kalmaz.
    """
    global _VECTORSTORE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META

    files = [
        os.path.join(UPLOADED_DIR, f)
        for f in os.listdir(UPLOADED_DIR)
        if os.path.isfile(os.path.join(UPLOADED_DIR, f))
    ]
    if not files:
        raise ValueError("Klasörde dosya yok. Önce dosya yükleyin.")

    # Metin bölme stratejisi: örtüşmeli pencere yöntemi
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_docs     = []   # LangChain Document listesi → FAISS için
    chunk_texts  = []   # Ham metin listesi → TF-IDF için
    chunk_meta   = []   # Metadata listesi → kaynak atfetme için

    for path in sorted(files):
        try:
            # Dosya tipine göre loader seç
            if path.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding="utf-8")

            loaded_docs = loader.load()
            split_docs  = splitter.split_documents(loaded_docs)

            for chunk_idx, doc in enumerate(split_docs):
                # Her chunk'a benzersiz bir kimlik ver
                chunk_id = f"{os.path.basename(path)}::chunk_{chunk_idx}"
                doc.metadata["chunk_id"] = chunk_id

                all_docs.append(doc)
                chunk_texts.append(doc.page_content)
                chunk_meta.append({
                    "source":   doc.metadata.get("source", path),
                    "chunk_id": chunk_id,
                    # ✅ BUG #3 FIX: chunk'ın gerçek listIndex'ini sakla;
                    #    combine_candidates'ta metin index karışıklığını önler
                    "list_idx": len(chunk_texts) - 1
                })

        except Exception as e:
            print(f"⚠️ Dosya okunamadı ({path}): {e}")

    if not all_docs:
        raise ValueError("Yüklenebilir metin bulunamadı.")

    # ── 1. FAISS index ──────────────────────────────────────────────────────
    print("🔨 FAISS index oluşturuluyor...")
    embeddings  = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_INDEX)

    # ── 2. TF-IDF index ─────────────────────────────────────────────────────
    # ngram_range=(1,2): tekil kelimeler + iki kelimelik ifadeler birlikte kullanılır
    print("🔨 TF-IDF index oluşturuluyor...")
    tfidf  = TfidfVectorizer(ngram_range=(1, 2), max_features=65536)
    matrix = tfidf.fit_transform(chunk_texts)

    # TF-IDF yapıtlarını diske kaydet
    with open(TFIDF_ARTIFACT, "wb") as f:
        pickle.dump({
            "vectorizer":  tfidf,
            "matrix":      matrix,
            "chunk_texts": chunk_texts,
            "chunk_meta":  chunk_meta,
        }, f)

    # Global değişkenlere ata (uygulama yeniden başlayana kadar bellekte kalır)
    _VECTORSTORE      = vectorstore
    _TFIDF_VECTORIZER = tfidf
    _TFIDF_MATRIX     = matrix
    _CHUNK_TEXTS      = chunk_texts
    _CHUNK_META       = chunk_meta

    save_sources_meta([os.path.basename(p) for p in files])
    print(f"✅ Index hazır: {len(all_docs)} chunk, {len(files)} dosya.")
    return vectorstore, [os.path.basename(p) for p in files]


def load_indices_if_exist(embed_model: str) -> bool:
    """
    Daha önce oluşturulmuş FAISS ve TF-IDF indexlerini diskten yükler.
    Eğer index yoksa False döner (build etmeden devam edilemez).
    """
    global _VECTORSTORE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META

    if _VECTORSTORE is None and os.path.exists(VECTORSTORE_INDEX):
        print("📂 FAISS index diskten yükleniyor...")
        embeddings  = HuggingFaceEmbeddings(model_name=embed_model)
        _VECTORSTORE = FAISS.load_local(
            VECTORSTORE_INDEX, embeddings,
            allow_dangerous_deserialization=True  # Güvenilir yerel dosya
        )

    if _TFIDF_VECTORIZER is None and os.path.exists(TFIDF_ARTIFACT):
        print("📂 TF-IDF index diskten yükleniyor...")
        with open(TFIDF_ARTIFACT, "rb") as f:
            data = pickle.load(f)
            _TFIDF_VECTORIZER = data["vectorizer"]
            _TFIDF_MATRIX     = data["matrix"]
            _CHUNK_TEXTS      = data["chunk_texts"]
            _CHUNK_META       = data["chunk_meta"]

    return _VECTORSTORE is not None


def ensure_indices_or_raise():
    """Index yoksa kullanıcıya anlamlı hata mesajı fırlatır."""
    if not load_indices_if_exist(_CURRENT_SETTINGS["embed_model"]):
        raise ValueError("Index bulunamadı. Dosya yükleyip 'Rebuild' yapın.")


# =============================================================================
# 🔍  Hybrid Retrieval: Semantic + Keyword
# =============================================================================

def semantic_candidates(query: str, k: int) -> list:
    """
    FAISS vektör deposundan en yakın k chunk'ı döner.
    ✅ v6'dan fark: similarity_search_with_score kullanır →
       reciprocal rank (1/rank) yerine gerçek L2 uzaklık skoru kullanılır.

    Returns: [(Document, score), ...]  — score küçüldükçe daha benzer (L2 uzaklık)
    """
    assert _VECTORSTORE is not None, "FAISS yüklenmemiş"
    # similarity_search_with_score: (doc, distance) çiftleri döner
    results = _VECTORSTORE.similarity_search_with_score(query, k=k)
    return results  # [(Document, float), ...]


def keyword_candidates(query: str, k: int) -> list:
    """
    TF-IDF cosine similarity ile en ilgili k chunk'ı döner.
    ✅ v6'dan fark: chunk'ın gerçek list_idx'ini de döner →
       _CHUNK_TEXTS'ten doğru metni almayı garantiler.

    Returns: [(chunk_meta, score, list_idx), ...]
    """
    if _TFIDF_VECTORIZER is None or _TFIDF_MATRIX is None:
        raise ValueError("TF-IDF index yüklenmemiş.")

    # Sorguyu TF-IDF vektörüne çevir
    qv   = _TFIDF_VECTORIZER.transform([query])          # shape: (1, vocab_size)
    sims = cosine_similarity(qv, _TFIDF_MATRIX).ravel()  # her chunk için cosine benzerliği

    # En yüksek k benzerliği sırala
    top_idx = np.argsort(-sims)[:k]

    results = []
    for idx in top_idx:
        results.append((
            _CHUNK_META[idx],    # metadata
            float(sims[idx]),    # cosine similarity skoru
            int(idx)             # ✅ BUG #3 FIX: gerçek list indeksi
        ))
    return results


def normalize(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalizasyon: skorları [0, 1] aralığına çeker.
    ✅ v6'dan fark: np.ptp() deprecated olduğu için .max()-.min() kullanılıyor.
    """
    rng = scores.max() - scores.min()
    if rng < 1e-9:
        return np.zeros_like(scores)
    return (scores - scores.min()) / rng


def combine_candidates(query: str, base_top_k: int, alpha: float) -> list:
    """
    Semantic (FAISS) ve keyword (TF-IDF) adaylarını birleştirir ve
    ağırlıklı skor hesaplar:
        final_score = alpha * sem_score + (1 - alpha) * kw_score

    alpha = 1.0 → tamamen semantic
    alpha = 0.0 → tamamen keyword
    alpha = 0.7 → önerilen denge (varsayılan)

    Önemli: FAISS L2 uzaklığı döner (düşük = iyi), bu yüzden ters çevrilir.

    Returns: [(chunk_key, final_score, value_dict), ...] — skora göre azalan sırada
    """
    # Sorgu uzunluğuna göre daha fazla aday al; en iyi k'yı sonradan seç
    words = len(query.split())
    scale = 1.0 if words <= 3 else (1.3 if words <= 12 else 1.8)
    k_fetch = max(3, int(base_top_k * scale))

    # ── Adayları getir ──────────────────────────────────────────────────────
    sems = semantic_candidates(query, k_fetch)  # [(Doc, l2_distance), ...]
    kws  = keyword_candidates(query, k_fetch)   # [(meta, cosine, list_idx), ...]

    # ── Semantic skoru normalize et (L2 uzaklığını benzerliğe çevir) ────────
    sem_dists = np.array([dist for (_, dist) in sems])
    # Uzaklık arttıkça benzerlik azalır → negatif al, sonra normalize et
    sem_scores_norm = normalize(-sem_dists)

    # ── Keyword skoru normalize et ──────────────────────────────────────────
    kw_sims = np.array([s for (_, s, _) in kws])
    kw_scores_norm = normalize(kw_sims)

    # ── Birleştirme: chunk_id → ortak sözlük ────────────────────────────────
    combined = {}

    # Semantic adayları ekle
    for i, (doc, _) in enumerate(sems):
        key = doc.metadata.get("chunk_id", f"sem_{i}")
        combined[key] = {
            "source":    doc.metadata.get("source", "Bilinmiyor"),
            "chunk_id":  key,
            "text":      doc.page_content,
            "sem_score": float(sem_scores_norm[i]),
            "kw_score":  0.0,   # TF-IDF'de yoksa 0
        }

    # Keyword adaylarını ekle veya mevcut kayda kw_score ekle
    for i, (meta, _, list_idx) in enumerate(kws):
        key = meta.get("chunk_id", f"kw_{i}")
        if key in combined:
            # Zaten semantic'ten geldi; sadece kw_score'u güncelle
            combined[key]["kw_score"] = float(kw_scores_norm[i])
        else:
            # Sadece keyword'de bulundu; semantic skoru 0
            combined[key] = {
                "source":    meta.get("source", "Bilinmiyor"),
                "chunk_id":  key,
                # ✅ BUG #3 FIX: list_idx kullanarak doğru metni al
                "text":      _CHUNK_TEXTS[list_idx] if list_idx < len(_CHUNK_TEXTS) else "",
                "sem_score": 0.0,
                "kw_score":  float(kw_scores_norm[i]),
            }

    # ── Final skor hesapla ve sırala ─────────────────────────────────────────
    merged = []
    for key, v in combined.items():
        final_score = alpha * v["sem_score"] + (1 - alpha) * v["kw_score"]
        merged.append((key, final_score, v))

    merged_sorted = sorted(merged, key=lambda x: -x[1])
    return merged_sorted


# =============================================================================
# 🏆  Reranker: Cross-Encoder ile son sıralama
# =============================================================================

def rerank(query: str, candidates: list, top_k: int) -> list:
    """
    CrossEncoder modeli ile hybrid sonuçlarını yeniden sıralar.

    CrossEncoder, sorgu-belge çiftini beraber değerlendirir (bi-encoder'dan daha doğru).
    Daha yavaş ama kesinlik oranı daha yüksek.

    Args:
        query:      Kullanıcı sorusu
        candidates: combine_candidates() çıktısı [(key, score, value), ...]
        top_k:      Kaç sonuç döneceğiz

    Returns:
        [(key, rerank_score, value), ...] — rerank skoruna göre azalan sırada
    """
    if not candidates:
        return []

    # Cross-encoder'a (sorgu, belge) çiftleri hazırla
    pairs = [(query, v["text"]) for (_, _, v) in candidates]

    # CrossEncoder puanları tahmin et
    scores = _RERANKER.predict(pairs)  # numpy array, her çift için bir skor

    # Skor + orijinal veri → yeniden sırala
    reranked = [
        (candidates[i][0], float(scores[i]), candidates[i][2])
        for i in range(len(candidates))
    ]
    reranked_sorted = sorted(reranked, key=lambda x: -x[1])
    return reranked_sorted[:top_k]


# =============================================================================
# 🔗  HybridRetriever: LangChain'e entegre custom retriever
# =============================================================================

class HybridRetriever(BaseRetriever):
    """
    ✅ BUG #2 FIX: v6'da chain sadece FAISS kullanıyordu;
       hybrid merge hesaplanıyordu ama LLM'e iletilmiyordu.

    Bu sınıf LangChain'in BaseRetriever'ından türer ve
    _get_relevant_documents() metodunu override eder.
    Böylece chain (LCEL) bu retriever'ı çağırdığında
    gerçek hybrid + rerank sonuçlarını alır.
    """

    # Pydantic field'ları (LangChain BaseRetriever'ın gerektirdiği)
    top_k:  int   = DEFAULT_TOP_K
    alpha:  float = DEFAULT_ALPHA
    rerank: bool  = DEFAULT_RERANK

    def _get_relevant_documents(self, query: str) -> list:
        """
        LCEL zinciri bu metodu çağırır.
        1. Hybrid merge (FAISS + TF-IDF)
        2. Opsiyonel cross-encoder reranking
        3. LangChain Document listesi döner
        """
        # Adımlar için daha fazla aday getir, sonra top_k'ya kes
        fetch_k     = min(self.top_k * 3, 20)
        candidates  = combine_candidates(query, fetch_k, alpha=self.alpha)

        if self.rerank and candidates:
            # Reranker varsa daha fazla aday üzerinden çalıştır
            top_candidates = candidates[:min(len(candidates), self.top_k * 2)]
            final = rerank(query, top_candidates, top_k=self.top_k)
        else:
            final = candidates[:self.top_k]

        # LangChain'in beklediği Document formatına dönüştür
        docs = []
        for (_, score, v) in final:
            docs.append(Document(
                page_content=v["text"],
                metadata={
                    "source":   v["source"],
                    "chunk_id": v["chunk_id"],
                    "score":    round(score, 4),
                }
            ))
        return docs

    async def _aget_relevant_documents(self, query: str) -> list:
        """Async versiyon (şimdilik sync'i çağırıyor; gerekirse genişletilebilir)."""
        return self._get_relevant_documents(query)


# =============================================================================
# ⛓️  LCEL Zinciri (Conversational RAG)
# =============================================================================

# RAG için prompt şablonu
# {context}  → retriever'dan gelen belgeler
# {history}  → konuşma geçmişi (HumanMessage / AIMessage listesi)
# {question} → kullanıcının sorusu
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Sen yardımcı bir asistansın. Aşağıdaki bağlam belgelerini kullanarak soruyu yanıtla.\n"
     "Eğer cevap belgelerde yoksa bunu açıkça söyle.\n\n"
     "Bağlam:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


def format_docs(docs: list) -> str:
    """Retriever'dan gelen Document listesini tek bir metin bloğuna çevirir."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        src = os.path.basename(doc.metadata.get("source", "?"))
        parts.append(f"[Belge {i} — {src}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain(retriever: HybridRetriever):
    """
    LCEL (LangChain Expression Language) zinciri kurar.
    v6'daki ConversationalRetrievalChain yerine modern LCEL pipe syntax kullanılıyor.

    Akış:
      1. retriever → ilgili chunk'ları getir
      2. format_docs → tek metin bloğu yap
      3. RAG_PROMPT → prompt'u doldur
      4. llm → cevap üret
      5. StrOutputParser → string çıktı al

    ⚠️ LCEL dikkat noktası:
       chain.invoke({"question": ..., "history": ...}) çağrıldığında
       her dal tüm dict'i alır. RunnablePassthrough() burada tüm dict'i
       geçirirdi — retriever da dict'i string sanıp .split() çağırır, patlardı.
       itemgetter("question") ile sadece ilgili alan çekilir.
    """
    chain = (
        {
            # itemgetter("question"): dict'ten sadece "question" değerini çeker
            # → retriever artık string alır, dict değil
            "context":  itemgetter("question") | retriever | format_docs,

            # itemgetter("history"): sadece konuşma geçmişini prompt'a ilet
            "history":  itemgetter("history"),

            # itemgetter("question"): soruyu prompt'a ilet
            "question": itemgetter("question"),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# =============================================================================
# 💬  Chat Handler
# =============================================================================

def chat_handler(message: str, chat_history: list, top_k: int, alpha: float,
                 adaptive: bool, use_rerank: bool):
    """
    Gradio chatbot'undan gelen mesajı işler ve cevap döner.

    ✅ BUG #1 FIX: v6'da mesajlar belleğe iki kez yazılıyordu
       (chain + manuel add_user_message/add_ai_message).
       Şimdi sadece bu fonksiyon SQLite'a yazar.

    ✅ BUG #2 FIX: Chain artık HybridRetriever kullanıyor.
    """
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})

    try:
        ensure_indices_or_raise()
    except Exception as e:
        err = f"⚠️ {e}"
        chat_history.append({"role": "assistant", "content": err})
        return chat_history

    # ── Adaptive top_k: sorgu uzunluğuna göre k ayarla ──────────────────────
    base_k = int(top_k)
    if adaptive:
        words = len(message.split())
        if words <= 3:
            k = max(3, base_k - 1)
        elif words <= 12:
            k = base_k
        else:
            k = min(12, int(base_k * 1.8))
    else:
        k = base_k

    # Global ayarları güncelle
    _CURRENT_SETTINGS.update({
        "top_k":  k,
        "alpha":  float(alpha),
        "rerank": bool(use_rerank),
    })

    # ── HybridRetriever oluştur ──────────────────────────────────────────────
    retriever = HybridRetriever(
        top_k=k,
        alpha=float(alpha),
        rerank=bool(use_rerank),
    )

    # ── Konuşma geçmişini LangChain mesaj formatına çevir ───────────────────
    history_messages = []
    for msg in _CHAT_HISTORY_STORE.messages:
        if isinstance(msg, HumanMessage):
            history_messages.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            history_messages.append(AIMessage(content=msg.content))

    # ── LCEL zincirini çalıştır ──────────────────────────────────────────────
    chain = build_chain(retriever)
    try:
        answer = chain.invoke({
            "question": message,
            "history":  history_messages,
        })
    except Exception as e:
        answer = f"❌ LLM hatası: {e}"

    # ── Kaynak atfetme: hangi belgelerden yararlanıldı? ──────────────────────
    try:
        final_docs = retriever._get_relevant_documents(message)
        used_files = sorted({os.path.basename(d.metadata.get("source", "?")) for d in final_docs})
        if used_files:
            answer += "\n\n📚 **Kaynaklar:**\n" + "\n".join(f"- {f}" for f in used_files)
    except Exception:
        pass  # Kaynak atfetme başarısız olursa cevabı etkileme

    # ── Konuşma geçmişini SQLite'a kaydet ───────────────────────────────────
    # ✅ BUG #1 FIX: Sadece burada yazıyoruz; chain ayrıca yazmıyor
    _CHAT_HISTORY_STORE.add_user_message(message)
    _CHAT_HISTORY_STORE.add_ai_message(answer)

    chat_history.append({"role": "assistant", "content": answer})
    return chat_history


# =============================================================================
# 🗑️  Bellek & Dosya Yönetimi (UI Aksiyonları)
# =============================================================================

def clear_memory():
    """
    ✅ BUG #4 FIX: v6'da sadece Gradio UI temizleniyordu;
       SQLite'daki gerçek konuşma geçmişi silinmiyordu.
       Şimdi hem UI hem veritabanı temizleniyor.
    """
    _CHAT_HISTORY_STORE.clear()  # SQLite'ı temizle
    return [], "🧹 Bellek temizlendi (UI + SQLite)."


def ui_upload_and_rebuild(files, chunk_size, chunk_overlap, top_k, embed_model, alpha):
    """Dosyaları kopyalar ve indexi yeniden oluşturur."""
    copied = copy_uploaded_files(files or [])
    if not copied:
        return [{"role": "assistant", "content": "Dosya seçilmedi."}], [], "Dosya yok."
    _CURRENT_SETTINGS.update({
        "chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap),
        "top_k": int(top_k), "embed_model": embed_model,
    })
    try:
        _, indexed = build_indices(int(chunk_size), int(chunk_overlap), embed_model)
    except Exception as e:
        return [{"role": "assistant", "content": f"❌ Hata: {e}"}], list_uploaded_files(), str(e)
    msg = f"✅ {len(indexed)} dosya indexlendi."
    return [{"role": "assistant", "content": msg}], list_uploaded_files(), msg


def ui_rebuild_existing(chunk_size, chunk_overlap, top_k, embed_model, alpha):
    """Yeni dosya yüklemeden mevcut dosyalardan indexi yeniden oluşturur."""
    _CURRENT_SETTINGS.update({
        "chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap),
        "top_k": int(top_k), "embed_model": embed_model,
    })
    try:
        _, indexed = build_indices(int(chunk_size), int(chunk_overlap), embed_model)
    except Exception as e:
        return [{"role": "assistant", "content": f"❌ Hata: {e}"}], list_uploaded_files(), str(e)
    msg = f"✅ {len(indexed)} dosya yeniden indexlendi."
    return [{"role": "assistant", "content": msg}], list_uploaded_files(), msg


def ui_delete_files(selected_files, chunk_size, chunk_overlap, top_k, embed_model, alpha):
    """Seçili dosyaları siler ve kalan dosyalardan indexi yeniden oluşturur."""
    global _VECTORSTORE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META

    if not selected_files:
        return [{"role": "assistant", "content": "Dosya seçilmedi."}], list_uploaded_files(), "Seçim yok."

    for name in selected_files:
        path = os.path.join(UPLOADED_DIR, name)
        if os.path.exists(path):
            os.remove(path)

    remaining = list_uploaded_files()
    if not remaining:
        # Tüm dosyalar silindi → indexleri temizle
        for artifact in [VECTORSTORE_INDEX, TFIDF_ARTIFACT, SOURCES_META]:
            if os.path.exists(artifact):
                (shutil.rmtree if os.path.isdir(artifact) else os.remove)(artifact)
        _VECTORSTORE = _TFIDF_VECTORIZER = _TFIDF_MATRIX = None
        _CHUNK_TEXTS = []
        _CHUNK_META  = []
        msg = "🗑️ Tüm dosyalar ve index silindi."
        return [{"role": "assistant", "content": msg}], [], msg

    # Kalan dosyalardan indexi yeniden oluştur
    return ui_rebuild_existing(chunk_size, chunk_overlap, top_k, embed_model, alpha)


def ui_semantic_search(query: str, top_k: int) -> str:
    """
    Ayrı bir 'Semantic Search' aracı: chat dışında doğrudan chunk arama.
    Hangi chunk'ların bulunduğunu görsel olarak incelemeye yarar.
    """
    if not query.strip():
        return "Sorgu girin."
    try:
        ensure_indices_or_raise()
    except Exception as e:
        return f"Index yok: {e}"

    merged = combine_candidates(query, int(top_k), alpha=_CURRENT_SETTINGS.get("alpha", DEFAULT_ALPHA))
    if not merged:
        return "Sonuç bulunamadı."

    lines = []
    for rank, (key, score, v) in enumerate(merged[:int(top_k)], start=1):
        src     = os.path.basename(v["source"])
        snippet = v["text"].strip().replace("\n", " ")[:800]
        snippet += "..." if len(v["text"]) > 800 else ""
        lines.append(f"**{rank}.** `{src}` — skor: {score:.3f}\n{snippet}\n")
    return "\n".join(lines)


# =============================================================================
# 🖥️  Gradio Arayüzü
# =============================================================================

with gr.Blocks(title="Local RAG Lab v7", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🔀 Local RAG Lab — v7 (Hybrid + Reranker + LCEL)")
    gr.Markdown(
        "**Yenilikler:** HybridRetriever, CrossEncoder reranker, LCEL zinciri, "
        "tüm v6 bug'ları düzeltildi."
    )

    with gr.Row():

        # ── Sol panel: Chat ─────────────────────────────────────────────────
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Sohbet", type="messages", height=480)
            with gr.Row():
                msg_box = gr.Textbox(
                    label="Sorunuz",
                    placeholder="Yüklenen belgeler hakkında sorun...",
                    scale=4
                )
                send_btn = gr.Button("Gönder 📤", scale=1)

        # ── Sağ panel: Ayarlar ve Araçlar ──────────────────────────────────
        with gr.Column(scale=1):

            with gr.Accordion("📁 Dosya Yükleme & Index", open=True):
                files_input      = gr.File(label="PDF / TXT / MD yükle", file_count="multiple", type="filepath")
                chunk_size_input = gr.Number(label="Chunk boyutu", value=DEFAULT_CHUNK_SIZE, precision=0)
                chunk_overlap_in = gr.Number(label="Chunk örtüşmesi", value=DEFAULT_CHUNK_OVERLAP, precision=0)
                upload_btn       = gr.Button("Yükle & Index Oluştur 🔨")
                rebuild_btn      = gr.Button("Mevcut dosyaları yeniden indexle 🔄")

            with gr.Accordion("🎛️ Retrieval Ayarları", open=True):
                top_k_slider   = gr.Slider(label="Base top_k", minimum=1, maximum=12, value=DEFAULT_TOP_K, step=1)
                alpha_slider   = gr.Slider(label="Semantic ağırlığı α (1=saf FAISS, 0=saf TF-IDF)",
                                           minimum=0.0, maximum=1.0, value=DEFAULT_ALPHA, step=0.05)
                adaptive_check = gr.Checkbox(label="Adaptive top_k (sorgu uzunluğuna göre ayarla)", value=True)
                rerank_check   = gr.Checkbox(label="CrossEncoder Reranker kullan", value=DEFAULT_RERANK)

            with gr.Accordion("📋 Indexlenmiş Dosyalar", open=True):
                indexed_list = gr.CheckboxGroup(
                    choices=list_uploaded_files(),
                    label="Dosyalar (silmek için seç)",
                    interactive=True
                )
                delete_btn = gr.Button("Seçilileri Sil 🗑️")

            with gr.Accordion("🔍 Doğrudan Chunk Arama", open=False):
                search_query = gr.Textbox(label="Arama sorgusu")
                search_top_k = gr.Slider(label="Kaç sonuç?", value=3, minimum=1, maximum=8, step=1)
                search_btn   = gr.Button("Ara 🔎")
                search_out   = gr.Markdown(label="Sonuçlar")

            with gr.Row():
                clear_mem_btn = gr.Button("🧹 Belleği Temizle")
                status_box    = gr.Textbox(label="Durum", interactive=False, scale=2)

    # Gizli: embed model değeri (UI'dan değiştirilemiyor ama fonksiyonlara aktarılıyor)
    embed_model_hidden = gr.Textbox(value=EMBED_MODEL, visible=False)

    # ── Event bağlantıları ──────────────────────────────────────────────────

    def on_upload(files, cs, co, tk, em, alpha):
        chat_msg, choices, status = ui_upload_and_rebuild(files, cs, co, tk, em, alpha)
        return chat_msg, gr.CheckboxGroup(choices=choices, value=[]), status

    upload_btn.click(
        fn=on_upload,
        inputs=[files_input, chunk_size_input, chunk_overlap_in, top_k_slider, embed_model_hidden, alpha_slider],
        outputs=[chatbot, indexed_list, status_box]
    )

    def on_rebuild(cs, co, tk, em, alpha):
        chat_msg, choices, status = ui_rebuild_existing(cs, co, tk, em, alpha)
        return chat_msg, gr.CheckboxGroup(choices=choices, value=[]), status

    rebuild_btn.click(
        fn=on_rebuild,
        inputs=[chunk_size_input, chunk_overlap_in, top_k_slider, embed_model_hidden, alpha_slider],
        outputs=[chatbot, indexed_list, status_box]
    )

    def on_delete(selected, cs, co, tk, em, alpha):
        chat_msg, choices, status = ui_delete_files(selected, cs, co, tk, em, alpha)
        return chat_msg, gr.CheckboxGroup(choices=choices, value=[]), status

    delete_btn.click(
        fn=on_delete,
        inputs=[indexed_list, chunk_size_input, chunk_overlap_in, top_k_slider, embed_model_hidden, alpha_slider],
        outputs=[chatbot, indexed_list, status_box]
    )

    search_btn.click(
        fn=ui_semantic_search,
        inputs=[search_query, search_top_k],
        outputs=[search_out]
    )

    clear_mem_btn.click(
        fn=clear_memory,
        outputs=[chatbot, status_box]
    )

    send_btn.click(
        fn=chat_handler,
        inputs=[msg_box, chatbot, top_k_slider, alpha_slider, adaptive_check, rerank_check],
        outputs=[chatbot]
    )

    # Enter tuşuyla da gönderilebilsin
    msg_box.submit(
        fn=chat_handler,
        inputs=[msg_box, chatbot, top_k_slider, alpha_slider, adaptive_check, rerank_check],
        outputs=[chatbot]
    )

# Uygulamayı başlat
demo.launch(server_name="127.0.0.1", server_port=7860)
