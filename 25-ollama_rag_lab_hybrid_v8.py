# =============================================================================
# 🔀 Local RAG Lab — v8 (Hybrid + Reranker + LCEL + Evaluation Pipeline)
# =============================================================================
#
# v7'den v8'e yapılan değişiklikler:
#   ✅ Evaluation Pipeline eklendi — 3 referanssız metrik:
#        • Faithfulness    : Cevap, context'e dayanıyor mu?
#        • Answer Relevancy: Cevap, soruyu karşılıyor mu?
#        • Context Relevancy: Getirilen chunk'lar soruyla ilgili mi?
#   ✅ LLM-as-a-Judge yaklaşımı — dış API gerekmez, tamamen Ollama ile çalışır
#   ✅ Evaluation log JSON'a kaydedilir (eval_log.json)
#   ✅ Gradio'ya "📊 Değerlendirme" sekmesi eklendi
#   ✅ Her chat yanıtı sonrası opsiyonel otomatik değerlendirme
#   ✅ Geçmiş skorların tablosu ve çalışan ortalamaları
#
# Kurulum (v7'ye ek):
#   pip install langchain langchain-community langchain-huggingface langchain-ollama
#   pip install faiss-cpu gradio sentence-transformers
#   pip install langchain-core pypdf scikit-learn numpy
#
# Çalıştırma:
#   python 25-ollama_rag_lab_hybrid_v8.py
# =============================================================================

import os
import re
import shutil
import json
import pickle
from datetime import datetime
from operator import itemgetter

import numpy as np
import gradio as gr

# --- LangChain bileşenleri ---
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Sklearn / Reranker ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder, SentenceTransformer


# =============================================================================
# ⚙️  Konfigürasyon & Sabitler
# =============================================================================

MODEL        = "llama3"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

UPLOADED_DIR      = "uploaded_files"
VECTORSTORE_DIR   = "vectorstore"
VECTORSTORE_INDEX = os.path.join(VECTORSTORE_DIR, "faiss_index")
SOURCES_META      = os.path.join(VECTORSTORE_DIR, "sources.json")
TFIDF_ARTIFACT    = os.path.join(VECTORSTORE_DIR, "tfidf_artifacts.pkl")
DB_PATH           = "chat_memory.sqlite"

# ✅ YENİ: Evaluation loglarının kaydedileceği dosya
EVAL_LOG_PATH = "eval_log.json"

DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K         = 4
DEFAULT_ALPHA         = 0.7
DEFAULT_RERANK        = True

os.makedirs(UPLOADED_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# =============================================================================
# 🧠  Global Durum
# =============================================================================

_VECTORSTORE      = None
_TFIDF_VECTORIZER = None
_TFIDF_MATRIX     = None
_CHUNK_TEXTS      = []
_CHUNK_META       = []

_CURRENT_SETTINGS = {
    "chunk_size":    DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "top_k":         DEFAULT_TOP_K,
    "embed_model":   EMBED_MODEL,
    "alpha":         DEFAULT_ALPHA,
    "rerank":        DEFAULT_RERANK,
}

# Ana LLM (sohbet için)
llm = ChatOllama(model=MODEL, temperature=0)

# Judge LLM (evaluation için) — aynı model ama ayrı instance; temperature=0 tutarlılık için
# Not: Daha hızlı evaluation istersen buraya daha küçük bir model yazabilirsin
# Örn: judge_llm = ChatOllama(model="phi3", temperature=0)
judge_llm = ChatOllama(model=MODEL, temperature=0)

# Reranker modeli
_RERANKER = CrossEncoder(RERANK_MODEL)

# ✅ YENİ: Answer Relevancy için embedding modeli
# Cevaptan üretilen soruları orijinal soruyla karşılaştırmak için kullanılır
_EMBED_MODEL = SentenceTransformer(EMBED_MODEL)

# Konuşma geçmişi
SESSION_ID = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
_CHAT_HISTORY_STORE = SQLChatMessageHistory(
    session_id=SESSION_ID,
    table_name="chat_history",
    connection_string=f"sqlite:///{DB_PATH}"
)


# =============================================================================
# 📁  Dosya Yönetimi (v7 ile aynı)
# =============================================================================

def copy_uploaded_files(filepaths: list) -> list:
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
    return sorted([
        f for f in os.listdir(UPLOADED_DIR)
        if os.path.isfile(os.path.join(UPLOADED_DIR, f))
    ])


def save_sources_meta(file_list: list):
    with open(SOURCES_META, "w", encoding="utf-8") as f:
        json.dump(list(file_list), f)


# =============================================================================
# 🗂️  Index Oluşturma (v7 ile aynı)
# =============================================================================

def build_indices(chunk_size: int, chunk_overlap: int, embed_model: str):
    global _VECTORSTORE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META

    files = [
        os.path.join(UPLOADED_DIR, f)
        for f in os.listdir(UPLOADED_DIR)
        if os.path.isfile(os.path.join(UPLOADED_DIR, f))
    ]
    if not files:
        raise ValueError("Klasörde dosya yok. Önce dosya yükleyin.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_docs, chunk_texts, chunk_meta = [], [], []

    for path in sorted(files):
        try:
            loader = PyPDFLoader(path) if path.lower().endswith(".pdf") else TextLoader(path, encoding="utf-8")
            for chunk_idx, doc in enumerate(splitter.split_documents(loader.load())):
                chunk_id = f"{os.path.basename(path)}::chunk_{chunk_idx}"
                doc.metadata["chunk_id"] = chunk_id
                all_docs.append(doc)
                chunk_texts.append(doc.page_content)
                chunk_meta.append({
                    "source":   doc.metadata.get("source", path),
                    "chunk_id": chunk_id,
                    "list_idx": len(chunk_texts) - 1
                })
        except Exception as e:
            print(f"⚠️ Dosya okunamadı ({path}): {e}")

    if not all_docs:
        raise ValueError("Yüklenebilir metin bulunamadı.")

    print("🔨 FAISS index oluşturuluyor...")
    embeddings  = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_INDEX)

    print("🔨 TF-IDF index oluşturuluyor...")
    tfidf  = TfidfVectorizer(ngram_range=(1, 2), max_features=65536)
    matrix = tfidf.fit_transform(chunk_texts)

    with open(TFIDF_ARTIFACT, "wb") as f:
        pickle.dump({
            "vectorizer": tfidf, "matrix": matrix,
            "chunk_texts": chunk_texts, "chunk_meta": chunk_meta,
        }, f)

    _VECTORSTORE = vectorstore
    _TFIDF_VECTORIZER, _TFIDF_MATRIX = tfidf, matrix
    _CHUNK_TEXTS, _CHUNK_META = chunk_texts, chunk_meta

    save_sources_meta([os.path.basename(p) for p in files])
    print(f"✅ Index hazır: {len(all_docs)} chunk, {len(files)} dosya.")
    return vectorstore, [os.path.basename(p) for p in files]


def load_indices_if_exist(embed_model: str) -> bool:
    global _VECTORSTORE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META

    if _VECTORSTORE is None and os.path.exists(VECTORSTORE_INDEX):
        embeddings   = HuggingFaceEmbeddings(model_name=embed_model)
        _VECTORSTORE = FAISS.load_local(
            VECTORSTORE_INDEX, embeddings, allow_dangerous_deserialization=True
        )

    if _TFIDF_VECTORIZER is None and os.path.exists(TFIDF_ARTIFACT):
        with open(TFIDF_ARTIFACT, "rb") as f:
            data = pickle.load(f)
            _TFIDF_VECTORIZER = data["vectorizer"]
            _TFIDF_MATRIX     = data["matrix"]
            _CHUNK_TEXTS      = data["chunk_texts"]
            _CHUNK_META       = data["chunk_meta"]

    return _VECTORSTORE is not None


def ensure_indices_or_raise():
    if not load_indices_if_exist(_CURRENT_SETTINGS["embed_model"]):
        raise ValueError("Index bulunamadı. Dosya yükleyip 'Rebuild' yapın.")


# =============================================================================
# 🔍  Hybrid Retrieval (v7 ile aynı)
# =============================================================================

def semantic_candidates(query: str, k: int) -> list:
    assert _VECTORSTORE is not None, "FAISS yüklenmemiş"
    return _VECTORSTORE.similarity_search_with_score(query, k=k)


def keyword_candidates(query: str, k: int) -> list:
    if _TFIDF_VECTORIZER is None or _TFIDF_MATRIX is None:
        raise ValueError("TF-IDF index yüklenmemiş.")
    qv      = _TFIDF_VECTORIZER.transform([query])
    sims    = cosine_similarity(qv, _TFIDF_MATRIX).ravel()
    top_idx = np.argsort(-sims)[:k]
    return [(_CHUNK_META[idx], float(sims[idx]), int(idx)) for idx in top_idx]


def normalize(scores: np.ndarray) -> np.ndarray:
    rng = scores.max() - scores.min()
    if rng < 1e-9:
        return np.zeros_like(scores)
    return (scores - scores.min()) / rng


def combine_candidates(query: str, base_top_k: int, alpha: float) -> list:
    words   = len(query.split())
    scale   = 1.0 if words <= 3 else (1.3 if words <= 12 else 1.8)
    k_fetch = max(3, int(base_top_k * scale))

    sems = semantic_candidates(query, k_fetch)
    kws  = keyword_candidates(query, k_fetch)

    sem_dists       = np.array([dist for (_, dist) in sems])
    sem_scores_norm = normalize(-sem_dists)
    kw_sims         = np.array([s for (_, s, _) in kws])
    kw_scores_norm  = normalize(kw_sims)

    combined = {}

    for i, (doc, _) in enumerate(sems):
        key = doc.metadata.get("chunk_id", f"sem_{i}")
        combined[key] = {
            "source":    doc.metadata.get("source", "Bilinmiyor"),
            "chunk_id":  key,
            "text":      doc.page_content,
            "sem_score": float(sem_scores_norm[i]),
            "kw_score":  0.0,
        }

    for i, (meta, _, list_idx) in enumerate(kws):
        key = meta.get("chunk_id", f"kw_{i}")
        if key in combined:
            combined[key]["kw_score"] = float(kw_scores_norm[i])
        else:
            combined[key] = {
                "source":    meta.get("source", "Bilinmiyor"),
                "chunk_id":  key,
                "text":      _CHUNK_TEXTS[list_idx] if list_idx < len(_CHUNK_TEXTS) else "",
                "sem_score": 0.0,
                "kw_score":  float(kw_scores_norm[i]),
            }

    merged = [
        (k, alpha * v["sem_score"] + (1 - alpha) * v["kw_score"], v)
        for k, v in combined.items()
    ]
    return sorted(merged, key=lambda x: -x[1])


def rerank(query: str, candidates: list, top_k: int) -> list:
    if not candidates:
        return []
    pairs  = [(query, v["text"]) for (_, _, v) in candidates]
    scores = _RERANKER.predict(pairs)
    reranked = [
        (candidates[i][0], float(scores[i]), candidates[i][2])
        for i in range(len(candidates))
    ]
    return sorted(reranked, key=lambda x: -x[1])[:top_k]


class HybridRetriever(BaseRetriever):
    """v7 ile aynı — LCEL zincirine entegre hybrid retriever."""
    top_k:  int   = DEFAULT_TOP_K
    alpha:  float = DEFAULT_ALPHA
    rerank: bool  = DEFAULT_RERANK

    def _get_relevant_documents(self, query: str) -> list:
        fetch_k    = min(self.top_k * 3, 20)
        candidates = combine_candidates(query, fetch_k, alpha=self.alpha)

        if self.rerank and candidates:
            top_cands = candidates[:min(len(candidates), self.top_k * 2)]
            final     = rerank(query, top_cands, top_k=self.top_k)
        else:
            final = candidates[:self.top_k]

        return [
            Document(
                page_content=v["text"],
                metadata={"source": v["source"], "chunk_id": v["chunk_id"], "score": round(score, 4)}
            )
            for (_, score, v) in final
        ]

    async def _aget_relevant_documents(self, query: str) -> list:
        return self._get_relevant_documents(query)


# =============================================================================
# ⛓️  LCEL Zinciri (v7 ile aynı)
# =============================================================================

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Sen yardımcı bir asistansın. Aşağıdaki bağlam belgelerini kullanarak soruyu yanıtla.\n"
     "Eğer cevap belgelerde yoksa bunu açıkça söyle.\n\n"
     "Bağlam:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


def format_docs(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        src = os.path.basename(doc.metadata.get("source", "?"))
        parts.append(f"[Belge {i} — {src}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain(retriever: HybridRetriever):
    return (
        {
            "context":  itemgetter("question") | retriever | format_docs,
            "history":  itemgetter("history"),
            "question": itemgetter("question"),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


# =============================================================================
# 📊  Evaluation Pipeline — LLM-as-a-Judge
# =============================================================================
#
# RAG sistemlerini değerlendirmek için 3 referanssız (ground-truth gerektirmeyen) metrik:
#
#   1. Faithfulness (Sadakat)
#      "Cevap, sadece context'teki bilgilere mi dayanıyor?"
#      → Cevaptaki her iddia context'te desteklenebiliyorsa yüksek skor alır.
#      → LLM judge, cevabı cümlelere böler ve her cümleyi context'te arar.
#
#   2. Answer Relevancy (Cevap Alaka Düzeyi)
#      "Cevap, gerçekten soruyu yanıtlıyor mu?"
#      → LLM judge, cevaptan N adet soru üretir.
#      → Bu soruların orijinal soruyla cosine similarity'si hesaplanır.
#      → Yüksek benzerlik → cevap soruya odaklanmış demektir.
#
#   3. Context Relevancy (Context Alaka Düzeyi)
#      "Getirilen chunk'lar soruyla ne kadar ilgili?"
#      → LLM judge, context içindeki soruyla alakalı cümleleri sayar.
#      → Alakalı cümle oranı skoru verir.
#
# Not: Bu 3 metrik RAGAS kütüphanesindeki kavramların Ollama ile
#      yerel olarak çalışan implementasyonudur.
# =============================================================================

def _ask_judge(prompt: str) -> str:
    """
    Judge LLM'e tek bir soru sorar ve string cevap alır.
    Tüm evaluation fonksiyonları bu yardımcıyı kullanır.
    """
    response = judge_llm.invoke(prompt)
    # ChatOllama AIMessage döner; .content ile string alırız
    return response.content.strip()


def evaluate_faithfulness(question: str, answer: str, context: str) -> dict:
    """
    Faithfulness (Sadakat) Metriği
    ─────────────────────────────
    Adımlar:
      1. Judge LLM, cevabı ayrı iddialara (claim) böler.
      2. Her iddia için "bu iddia context'te destekleniyor mu?" diye sorar.
      3. Desteklenen iddia sayısı / toplam iddia sayısı = skor

    Skor = 1.0 → cevabın tamamı context'e dayanıyor (hallucination yok)
    Skor = 0.0 → cevap tamamen context dışı bilgiden oluşuyor
    """
    # Adım 1: Cevabı iddialara böl
    decompose_prompt = f"""Aşağıdaki cevabı bağımsız ve atomik iddialara böl.
Her iddiayı yeni bir satıra yaz. Sadece iddiaları listele, başka bir şey yazma.

Cevap: {answer}

İddialar (her satıra bir tane):"""

    raw_claims = _ask_judge(decompose_prompt)
    # Her satırı ayrı bir iddia olarak al; boş satırları ve madde işaretlerini temizle
    claims = [
        re.sub(r"^[\-\*\d\.\)]+\s*", "", line).strip()
        for line in raw_claims.split("\n")
        if line.strip() and len(line.strip()) > 10
    ]

    if not claims:
        return {"score": 0.0, "supported": 0, "total": 0, "detail": "İddia bulunamadı."}

    # Adım 2: Her iddiayı context'e karşı kontrol et
    supported = 0
    claim_results = []

    for claim in claims:
        verify_prompt = f"""Aşağıdaki iddia, verilen bağlam metninde destekleniyor mu?
Sadece 'EVET' veya 'HAYIR' yaz.

Bağlam: {context[:2000]}

İddia: {claim}

Cevap:"""
        verdict = _ask_judge(verify_prompt).upper()
        is_supported = "EVET" in verdict or "YES" in verdict
        if is_supported:
            supported += 1
        claim_results.append(f"{'✅' if is_supported else '❌'} {claim}")

    score = supported / len(claims)
    detail = f"**{supported}/{len(claims)} iddia context'te desteklendi**\n\n" + "\n".join(claim_results)

    return {"score": round(score, 3), "supported": supported, "total": len(claims), "detail": detail}


def evaluate_answer_relevancy(question: str, answer: str, n_questions: int = 3) -> dict:
    """
    Answer Relevancy (Cevap Alaka Düzeyi) Metriği
    ──────────────────────────────────────────────
    Adımlar:
      1. Judge LLM, cevabı görerek N adet olası soru üretir.
         (Yani: "Bu cevap hangi soruya verilmiş olabilir?")
      2. Üretilen soruların her biri, orijinal soruyla embedding cosine similarity ile karşılaştırılır.
      3. Ortalama benzerlik = skor

    Fikir: Cevap soruyu iyi karşılıyorsa, cevaptan üretilen sorular orijinal soruya benzer olmalı.
    Cevap konu dışına çıktıysa, üretilen sorular farklı yönlere işaret eder → düşük skor.

    Skor = 1.0 → cevap tam olarak soruya odaklanmış
    Skor = 0.0 → cevap konuyla alakasız
    """
    gen_prompt = f"""Aşağıdaki cevabı gören biri, bu cevabın hangi soruya verildiğini merak ediyor.
Bu cevap için {n_questions} farklı olası soru üret.
Her soruyu yeni bir satıra yaz. Başka hiçbir şey yazma.

Cevap: {answer}

Olası sorular:"""

    raw_questions = _ask_judge(gen_prompt)
    generated_qs  = [
        re.sub(r"^[\-\*\d\.\)]+\s*", "", line).strip()
        for line in raw_questions.split("\n")
        if line.strip() and "?" in line
    ][:n_questions]

    if not generated_qs:
        return {"score": 0.0, "generated_questions": [], "detail": "Soru üretilemedi."}

    # Embedding cosine similarity hesapla
    original_emb  = _EMBED_MODEL.encode([question])          # (1, dim)
    generated_emb = _EMBED_MODEL.encode(generated_qs)        # (N, dim)
    sims          = cosine_similarity(original_emb, generated_emb).ravel()
    score         = float(sims.mean())

    detail_lines = [f"Orijinal soru: **{question}**\n\nÜretilen sorular ve benzerlik:"]
    for q, s in zip(generated_qs, sims):
        bar = "█" * int(s * 10) + "░" * (10 - int(s * 10))
        detail_lines.append(f"- {q}\n  [{bar}] {s:.3f}")

    return {
        "score":              round(score, 3),
        "generated_questions": generated_qs,
        "similarities":       [round(float(s), 3) for s in sims],
        "detail":             "\n".join(detail_lines)
    }


def evaluate_context_relevancy(question: str, context: str) -> dict:
    """
    Context Relevancy (Context Alaka Düzeyi) Metriği
    ─────────────────────────────────────────────────
    Adımlar:
      1. Context, cümlelere bölünür.
      2. Judge LLM, hangi cümlelerin soruyu cevaplamak için gerekli olduğunu belirler.
      3. Gerekli cümle sayısı / toplam cümle sayısı = skor

    Skor = 1.0 → getirilen her chunk soruyla doğrudan ilgili (retrieval çok iyi)
    Skor = 0.0 → getirilen chunk'ların hiçbiri soruyla ilgili değil (retrieval kötü)

    Düşük skor → top_k'yı azalt veya alpha'yı ayarla.
    """
    # Context'i cümlelere böl (basit nokta bazlı bölme)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if len(s.strip()) > 20]

    if not sentences:
        return {"score": 0.0, "relevant": 0, "total": 0, "detail": "Context boş."}

    # LLM'e gönderilecek cümle listesi (çok uzun olmasın diye ilk 30 cümle)
    sample     = sentences[:30]
    numbered   = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sample))

    filter_prompt = f"""Aşağıdaki soruyu cevaplamak için aşağıdaki cümlelerden hangisi gerekli?
Sadece gerekli cümlelerin numaralarını virgülle yaz. Başka bir şey yazma.

Soru: {question}

Cümleler:
{numbered}

Gerekli cümle numaraları:"""

    raw_nums = _ask_judge(filter_prompt)

    # Sayıları parse et
    relevant_indices = set()
    for token in re.findall(r'\d+', raw_nums):
        idx = int(token) - 1  # 1-indexed → 0-indexed
        if 0 <= idx < len(sample):
            relevant_indices.add(idx)

    score = len(relevant_indices) / len(sample) if sample else 0.0

    detail_lines = [f"**{len(relevant_indices)}/{len(sample)} cümle alakalı bulundu**\n"]
    for i, s in enumerate(sample):
        icon = "✅" if i in relevant_indices else "⬜"
        detail_lines.append(f"{icon} {s[:120]}{'...' if len(s) > 120 else ''}")

    return {
        "score":    round(score, 3),
        "relevant": len(relevant_indices),
        "total":    len(sample),
        "detail":   "\n".join(detail_lines)
    }


def run_full_evaluation(question: str, answer: str, context_docs: list) -> dict:
    """
    Tek bir soru-cevap-context üçlüsü için 3 metriği de çalıştırır.

    Args:
        question:     Kullanıcının sorusu
        answer:       LLM'in ürettiği cevap (kaynak satırları olmadan)
        context_docs: HybridRetriever'dan dönen Document listesi

    Returns:
        Tüm metrikleri, skorları ve detayları içeren dict
    """
    # Context'i tek string'e çevir (faithfulness ve context relevancy için)
    context_str = "\n\n".join(doc.page_content for doc in context_docs)

    print("🔍 Faithfulness hesaplanıyor...")
    faith   = evaluate_faithfulness(question, answer, context_str)

    print("🔍 Answer Relevancy hesaplanıyor...")
    rel     = evaluate_answer_relevancy(question, answer)

    print("🔍 Context Relevancy hesaplanıyor...")
    ctx_rel = evaluate_context_relevancy(question, context_str)

    # Kullanılan ayarları da kaydet (hangi parametrelerle bu skoru aldık?)
    settings_snapshot = {
        "top_k":         _CURRENT_SETTINGS.get("top_k", DEFAULT_TOP_K),
        "alpha":         _CURRENT_SETTINGS.get("alpha", DEFAULT_ALPHA),
        "rerank":        _CURRENT_SETTINGS.get("rerank", DEFAULT_RERANK),
        "chunk_size":    _CURRENT_SETTINGS.get("chunk_size", DEFAULT_CHUNK_SIZE),
        "chunk_overlap": _CURRENT_SETTINGS.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
    }

    result = {
        "timestamp":          datetime.now().isoformat(),
        "question":           question,
        "answer_preview":     answer[:200] + "..." if len(answer) > 200 else answer,
        "faithfulness":       faith,
        "answer_relevancy":   rel,
        "context_relevancy":  ctx_rel,
        "settings":           settings_snapshot,
        # Özet skorlar (log tablosunda göstermek için)
        "scores": {
            "faithfulness":       faith["score"],
            "answer_relevancy":   rel["score"],
            "context_relevancy":  ctx_rel["score"],
            "overall":            round((faith["score"] + rel["score"] + ctx_rel["score"]) / 3, 3),
        }
    }

    return result


# =============================================================================
# 💾  Evaluation Log — JSON'a okuma/yazma
# =============================================================================

def load_eval_log() -> list:
    """
    eval_log.json dosyasını okur.
    Dosya yoksa boş liste döner.
    """
    if not os.path.exists(EVAL_LOG_PATH):
        return []
    with open(EVAL_LOG_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []


def save_eval_log(log: list):
    """Evaluation logunu JSON'a yazar."""
    with open(EVAL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def append_eval_result(result: dict):
    """
    Yeni bir evaluation sonucunu mevcut loga ekler ve diske yazar.
    Her çalıştırma geçmişi birikiyor; böylece farklı parametrelerle
    yapılan deneyleri karşılaştırmak mümkün oluyor.
    """
    log = load_eval_log()
    log.append(result)
    save_eval_log(log)


def format_eval_table() -> str:
    """
    Tüm evaluation logunu Gradio'da gösterilecek Markdown tablosuna çevirir.
    En yeni sonuçlar üstte görünür.
    """
    log = load_eval_log()
    if not log:
        return "Henüz değerlendirme yapılmadı. Sohbet et ve 'Değerlendir' butonuna bas."

    # Çalışan ortalamalar
    all_scores = [e["scores"] for e in log]
    avg_faith  = round(sum(s["faithfulness"]      for s in all_scores) / len(all_scores), 3)
    avg_rel    = round(sum(s["answer_relevancy"]   for s in all_scores) / len(all_scores), 3)
    avg_ctx    = round(sum(s["context_relevancy"]  for s in all_scores) / len(all_scores), 3)
    avg_all    = round(sum(s["overall"]            for s in all_scores) / len(all_scores), 3)

    def score_emoji(s: float) -> str:
        """Skora göre görsel gösterge döner."""
        if s >= 0.8: return "🟢"
        if s >= 0.5: return "🟡"
        return "🔴"

    lines = [
        f"## 📊 Evaluation Geçmişi ({len(log)} kayıt)",
        "",
        f"**Çalışan Ortalamalar** — "
        f"Sadakat: {score_emoji(avg_faith)} {avg_faith} | "
        f"Cevap Alakası: {score_emoji(avg_rel)} {avg_rel} | "
        f"Context Alakası: {score_emoji(avg_ctx)} {avg_ctx} | "
        f"Genel: {score_emoji(avg_all)} {avg_all}",
        "",
        "| # | Soru | Sadakat | Cevap Alaka | Context Alaka | Genel | top_k | α | Rerank |",
        "|---|------|---------|-------------|---------------|-------|-------|---|--------|",
    ]

    for i, entry in enumerate(reversed(log), start=1):
        s     = entry["scores"]
        q     = entry["question"][:40] + "..." if len(entry["question"]) > 40 else entry["question"]
        setts = entry.get("settings", {})
        lines.append(
            f"| {i} | {q} | "
            f"{score_emoji(s['faithfulness'])} {s['faithfulness']} | "
            f"{score_emoji(s['answer_relevancy'])} {s['answer_relevancy']} | "
            f"{score_emoji(s['context_relevancy'])} {s['context_relevancy']} | "
            f"{score_emoji(s['overall'])} **{s['overall']}** | "
            f"{setts.get('top_k','?')} | "
            f"{setts.get('alpha','?')} | "
            f"{'✅' if setts.get('rerank') else '❌'} |"
        )

    return "\n".join(lines)


def format_last_eval_detail() -> str:
    """
    En son evaluation sonucunun ayrıntılı açıklamasını döner.
    Her metriğin "neden bu skoru aldı?" sorusunu cevaplar.
    """
    log = load_eval_log()
    if not log:
        return "Henüz değerlendirme yok."

    entry = log[-1]   # En son sonuç
    s     = entry["scores"]

    def score_bar(score: float, width: int = 20) -> str:
        filled = int(score * width)
        return "█" * filled + "░" * (width - filled) + f" {score:.3f}"

    lines = [
        f"## 🔬 Son Değerlendirme Detayı",
        f"**Soru:** {entry['question']}",
        f"**Cevap:** {entry['answer_preview']}",
        f"**Zaman:** {entry['timestamp']}",
        "",
        "---",
        "",
        f"### 1️⃣ Faithfulness (Sadakat)",
        f"`{score_bar(s['faithfulness'])}`",
        f"Cevap, sadece context'teki bilgilere dayanıyor mu?",
        f"{entry['faithfulness']['supported']}/{entry['faithfulness']['total']} iddia desteklendi.",
        "",
        entry["faithfulness"].get("detail", ""),
        "",
        "---",
        "",
        f"### 2️⃣ Answer Relevancy (Cevap Alaka Düzeyi)",
        f"`{score_bar(s['answer_relevancy'])}`",
        f"Cevap, soruyu gerçekten yanıtlıyor mu?",
        "",
        entry["answer_relevancy"].get("detail", ""),
        "",
        "---",
        "",
        f"### 3️⃣ Context Relevancy (Context Alaka Düzeyi)",
        f"`{score_bar(s['context_relevancy'])}`",
        f"Getirilen chunk'lar soruyla ne kadar ilgili?",
        f"{entry['context_relevancy']['relevant']}/{entry['context_relevancy']['total']} cümle alakalı.",
        "",
        entry["context_relevancy"].get("detail", ""),
    ]

    return "\n".join(lines)


# =============================================================================
# 💬  Chat Handler (v8 — evaluation entegreli)
# =============================================================================

# Son yanıtın evaluation için gerekli verilerini tutan global
# (Gradio'nun state mekanizmasını kullanmak yerine basit global tercih ettik)
_LAST_RESPONSE = {
    "question":     "",
    "answer":       "",   # kaynak satırları olmadan saf cevap
    "context_docs": [],   # HybridRetriever'dan dönen Document listesi
}


def chat_handler(message: str, chat_history: list, top_k: int, alpha: float,
                 adaptive: bool, use_rerank: bool, auto_eval: bool):
    """
    v8 değişikliği:
      - Cevap ve context_docs, _LAST_RESPONSE global'ine kaydediliyor.
      - auto_eval True ise her yanıt sonrası otomatik evaluation çalışıyor.
    """
    global _LAST_RESPONSE

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})

    try:
        ensure_indices_or_raise()
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"⚠️ {e}"})
        return chat_history

    # Adaptive top_k
    base_k = int(top_k)
    if adaptive:
        words = len(message.split())
        k = max(3, base_k - 1) if words <= 3 else (base_k if words <= 12 else min(12, int(base_k * 1.8)))
    else:
        k = base_k

    _CURRENT_SETTINGS.update({"top_k": k, "alpha": float(alpha), "rerank": bool(use_rerank)})

    retriever = HybridRetriever(top_k=k, alpha=float(alpha), rerank=bool(use_rerank))

    history_messages = [
        HumanMessage(content=m.content) if isinstance(m, HumanMessage) else AIMessage(content=m.content)
        for m in _CHAT_HISTORY_STORE.messages
    ]

    chain = build_chain(retriever)
    try:
        answer = chain.invoke({"question": message, "history": history_messages})
    except Exception as e:
        answer = f"❌ LLM hatası: {e}"

    # Context dokümanlarını al (evaluation + kaynak atfetme için)
    try:
        context_docs = retriever._get_relevant_documents(message)
    except Exception:
        context_docs = []

    # ✅ YENİ: Saf cevabı (kaynak satırları olmadan) evaluation için sakla
    _LAST_RESPONSE = {
        "question":     message,
        "answer":       answer,   # kaynak eklenmeden önce sakla
        "context_docs": context_docs,
    }

    # Kaynak atfetme
    used_files = sorted({os.path.basename(d.metadata.get("source", "?")) for d in context_docs})
    if used_files:
        answer += "\n\n📚 **Kaynaklar:**\n" + "\n".join(f"- {f}" for f in used_files)

    _CHAT_HISTORY_STORE.add_user_message(message)
    _CHAT_HISTORY_STORE.add_ai_message(answer)
    chat_history.append({"role": "assistant", "content": answer})

    # ✅ YENİ: Otomatik evaluation
    if auto_eval and context_docs:
        try:
            print("🔄 Otomatik evaluation başlıyor...")
            result = run_full_evaluation(message, _LAST_RESPONSE["answer"], context_docs)
            append_eval_result(result)
            s = result["scores"]
            chat_history.append({
                "role":    "assistant",
                "content": (
                    f"📊 **Otomatik Değerlendirme:**  "
                    f"Sadakat {s['faithfulness']} | "
                    f"Cevap Alakası {s['answer_relevancy']} | "
                    f"Context Alakası {s['context_relevancy']} | "
                    f"Genel **{s['overall']}**"
                )
            })
        except Exception as e:
            print(f"⚠️ Otomatik eval hatası: {e}")

    return chat_history


def manual_evaluate():
    """
    Kullanıcı 'Değerlendir' butonuna basınca son yanıtı değerlendirir.
    Otomatik eval kapalıyken veya belirli bir yanıtı tekrar değerlendirmek isteyince kullanılır.
    """
    if not _LAST_RESPONSE["question"]:
        return "Önce bir soru sor.", format_eval_table()

    if not _LAST_RESPONSE["context_docs"]:
        return "Context bulunamadı. Index yüklü olduğundan emin ol.", format_eval_table()

    try:
        print("🔄 Manuel evaluation başlıyor...")
        result = run_full_evaluation(
            _LAST_RESPONSE["question"],
            _LAST_RESPONSE["answer"],
            _LAST_RESPONSE["context_docs"],
        )
        append_eval_result(result)
        return format_last_eval_detail(), format_eval_table()
    except Exception as e:
        return f"❌ Evaluation hatası: {e}", format_eval_table()


def clear_eval_log():
    """Tüm evaluation geçmişini temizler."""
    if os.path.exists(EVAL_LOG_PATH):
        os.remove(EVAL_LOG_PATH)
    return "🧹 Evaluation geçmişi temizlendi.", ""


# =============================================================================
# 🗑️  Dosya & Bellek Yönetimi (v7 ile aynı)
# =============================================================================

def clear_memory():
    _CHAT_HISTORY_STORE.clear()
    return [], "🧹 Bellek temizlendi."


def ui_upload_and_rebuild(files, chunk_size, chunk_overlap, top_k, embed_model, alpha):
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
    global _VECTORSTORE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _CHUNK_TEXTS, _CHUNK_META
    if not selected_files:
        return [{"role": "assistant", "content": "Dosya seçilmedi."}], list_uploaded_files(), "Seçim yok."
    for name in selected_files:
        path = os.path.join(UPLOADED_DIR, name)
        if os.path.exists(path):
            os.remove(path)
    remaining = list_uploaded_files()
    if not remaining:
        for artifact in [VECTORSTORE_INDEX, TFIDF_ARTIFACT, SOURCES_META]:
            if os.path.exists(artifact):
                (shutil.rmtree if os.path.isdir(artifact) else os.remove)(artifact)
        _VECTORSTORE = _TFIDF_VECTORIZER = _TFIDF_MATRIX = None
        _CHUNK_TEXTS = []; _CHUNK_META = []
        msg = "🗑️ Tüm dosyalar ve index silindi."
        return [{"role": "assistant", "content": msg}], [], msg
    return ui_rebuild_existing(chunk_size, chunk_overlap, top_k, embed_model, alpha)


def ui_semantic_search(query: str, top_k: int) -> str:
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
# 🖥️  Gradio Arayüzü — v8 (Evaluation sekmesi eklendi)
# =============================================================================

with gr.Blocks(title="Local RAG Lab v8", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🔀 Local RAG Lab — v8 (Hybrid + Reranker + LCEL + Evaluation)")
    gr.Markdown(
        "**Yenilikler:** Faithfulness · Answer Relevancy · Context Relevancy metrikleri, "
        "LLM-as-a-Judge yaklaşımı, evaluation geçmişi ve karşılaştırma tablosu."
    )

    # ── Ana sekmeler ──────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── SEKME 1: Sohbet ──────────────────────────────────────────────────
        with gr.Tab("💬 Sohbet"):
            with gr.Row():
                # Sol: chatbot
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Sohbet", type="messages", height=480)
                    with gr.Row():
                        msg_box  = gr.Textbox(
                            label="Sorunuz",
                            placeholder="Yüklenen belgeler hakkında sorun...",
                            scale=4
                        )
                        send_btn = gr.Button("Gönder 📤", scale=1)

                # Sağ: ayarlar
                with gr.Column(scale=1):

                    with gr.Accordion("📁 Dosya Yükleme & Index", open=True):
                        files_input      = gr.File(label="PDF / TXT / MD yükle", file_count="multiple", type="filepath")
                        chunk_size_input = gr.Number(label="Chunk boyutu", value=DEFAULT_CHUNK_SIZE, precision=0)
                        chunk_overlap_in = gr.Number(label="Chunk örtüşmesi", value=DEFAULT_CHUNK_OVERLAP, precision=0)
                        upload_btn       = gr.Button("Yükle & Index Oluştur 🔨")
                        rebuild_btn      = gr.Button("Mevcut dosyaları yeniden indexle 🔄")

                    with gr.Accordion("🎛️ Retrieval Ayarları", open=True):
                        top_k_slider   = gr.Slider(label="Base top_k", minimum=1, maximum=12, value=DEFAULT_TOP_K, step=1)
                        alpha_slider   = gr.Slider(label="Semantic ağırlığı α", minimum=0.0, maximum=1.0, value=DEFAULT_ALPHA, step=0.05)
                        adaptive_check = gr.Checkbox(label="Adaptive top_k", value=True)
                        rerank_check   = gr.Checkbox(label="CrossEncoder Reranker kullan", value=DEFAULT_RERANK)

                    with gr.Accordion("📊 Evaluation Ayarları", open=True):
                        # ✅ YENİ: Otomatik evaluation toggle
                        auto_eval_check = gr.Checkbox(
                            label="Her yanıt sonrası otomatik değerlendir",
                            value=False,   # Varsayılan kapalı; evaluation yavaş olabilir
                            info="Açıkken her cevap için ek LLM çağrıları yapılır (~30-60 sn)"
                        )
                        eval_btn = gr.Button("🔬 Son yanıtı şimdi değerlendir", variant="primary")

                    with gr.Accordion("📋 Indexlenmiş Dosyalar", open=False):
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
                        search_out   = gr.Markdown()

                    with gr.Row():
                        clear_mem_btn = gr.Button("🧹 Belleği Temizle")
                        status_box    = gr.Textbox(label="Durum", interactive=False, scale=2)

            embed_model_hidden = gr.Textbox(value=EMBED_MODEL, visible=False)

        # ── SEKME 2: Evaluation Dashboard ────────────────────────────────────
        with gr.Tab("📊 Değerlendirme"):
            gr.Markdown(
                "### RAG Evaluation Dashboard\n"
                "Her sorgu için 3 metrik hesaplanır ve aşağıda birikir.\n"
                "Farklı ayarları (top_k, alpha, rerank) deneyip sonuçları karşılaştırabilirsin."
            )

            with gr.Row():
                refresh_table_btn = gr.Button("🔄 Tabloyu Yenile")
                clear_log_btn     = gr.Button("🗑️ Geçmişi Temizle")

            eval_status = gr.Textbox(label="Durum", interactive=False)

            # Geçmiş tablosu
            eval_table_md = gr.Markdown(value=format_eval_table())

            gr.Markdown("---")

            # Son değerlendirme detayı
            gr.Markdown("### 🔬 Son Değerlendirme Detayı")
            eval_detail_md = gr.Markdown(value="*Değerlendirme yapıldığında burada görünecek.*")

    # =========================================================================
    # 🔌  Event Bağlantıları
    # =========================================================================

    # Upload & rebuild
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

    search_btn.click(fn=ui_semantic_search, inputs=[search_query, search_top_k], outputs=[search_out])

    clear_mem_btn.click(fn=clear_memory, outputs=[chatbot, status_box])

    # Chat — auto_eval_check artık input olarak eklendi
    send_btn.click(
        fn=chat_handler,
        inputs=[msg_box, chatbot, top_k_slider, alpha_slider, adaptive_check, rerank_check, auto_eval_check],
        outputs=[chatbot]
    )
    msg_box.submit(
        fn=chat_handler,
        inputs=[msg_box, chatbot, top_k_slider, alpha_slider, adaptive_check, rerank_check, auto_eval_check],
        outputs=[chatbot]
    )

    # ✅ YENİ: Manuel evaluation butonu
    eval_btn.click(
        fn=manual_evaluate,
        outputs=[eval_detail_md, eval_table_md]
    )

    # ✅ YENİ: Tablo yenile
    refresh_table_btn.click(
        fn=format_eval_table,
        outputs=[eval_table_md]
    )

    # ✅ YENİ: Geçmişi temizle
    clear_log_btn.click(
        fn=clear_eval_log,
        outputs=[eval_status, eval_detail_md]
    )
    clear_log_btn.click(fn=format_eval_table, outputs=[eval_table_md])


demo.launch(server_name="127.0.0.1", server_port=7860)
