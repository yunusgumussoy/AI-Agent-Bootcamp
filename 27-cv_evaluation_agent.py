# =============================================================================
# 📄 CV Değerlendirme Agent — LangGraph ile
# =============================================================================
#
# RAG agent'ından ne değişti, ne aynı kaldı:
#
#   AYNI KALAN (LangGraph altyapısı):
#     ✅ StateGraph yapısı (agent_node → should_continue → tool_node döngüsü)
#     ✅ AgentState (TypedDict + add_messages)
#     ✅ Manuel ReAct parsing (_parse_react_output)
#     ✅ should_continue ve tool_node mantığı
#
#   DEĞİŞEN (domain'e özgü):
#     🔄 AgentState → cv_text, job_text, report alanları eklendi
#     🔄 Araçlar → extract_cv_info, match_skills, gap_analysis, generate_report
#     🔄 Sistem promptu → CV değerlendirme uzmanı rolü
#     🔄 Gradio UI → CV + iş ilanı yükleme, rapor paneli
#     🔄 Otomatik değerlendirme akışı → tek butonla tüm araçlar sırayla çalışır
#
# Kurulum (yeni bağımlılık yok, RAG agent ile aynı):
#   pip install langgraph langchain langchain-ollama gradio pypdf
#
# Çalıştırma:
#   python 27-cv_evaluation_agent.py
#   → http://127.0.0.1:7862  (RAG agent 7861, bu 7862)
# =============================================================================

import os
import re
import json
import shutil
from datetime import datetime
from typing import Annotated, TypedDict

import gradio as gr

# --- LangGraph ---
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# --- LangChain ---
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# =============================================================================
# ⚙️  Konfigürasyon
# =============================================================================

MODEL = "llama3"   # llama3.1 varsa onu kullan: daha iyi sonuç verir

CV_DIR      = "cv_files"       # Yüklenen CV'lerin saklandığı klasör
REPORT_DIR  = "cv_reports"     # Üretilen raporların kaydedileceği klasör

os.makedirs(CV_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# LLM — tüm araçlar ve karar mekanizması bu modeli kullanır
_LLM = ChatOllama(model=MODEL, temperature=0)


# =============================================================================
# 📋  Agent State — CV agent'ına özgü alanlar
# =============================================================================
#
# RAG agent'ından farkı:
#   Sadece "messages" değil; cv_text, job_text, report gibi
#   domain'e özgü alanlar da state'e eklendi.
#
#   Bu sayede her araç sonucu hem messages'a (LLM geçmişi için)
#   hem de ilgili state alanına (UI'da göstermek için) yazılır.
#
# LangGraph'ta state alanları:
#   - Annotated[list, add_messages] → her güncelleme EKLER (üzerine yazmaz)
#   - Düz str/dict               → her güncelleme ÜZERİNE YAZAR

class CVAgentState(TypedDict):
    messages:      Annotated[list, add_messages]  # Konuşma geçmişi (birikimli)
    cv_text:       str    # Ham CV metni (PDF'den çıkarılan)
    job_text:      str    # İş ilanı metni (kullanıcının girdiği)
    cv_info:       str    # extract_cv_info aracının çıktısı
    match_result:  str    # match_skills aracının çıktısı
    gap_result:    str    # gap_analysis aracının çıktısı
    report:        str    # generate_report aracının final çıktısı


# =============================================================================
# 📁  CV Metin Çıkarma
# =============================================================================

def extract_text_from_file(path: str) -> str:
    """
    PDF veya TXT dosyasından ham metni çıkarır.
    LangChain loader kullanır; tüm sayfaları tek string'e birleştirir.
    """
    try:
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        # Tüm sayfaları/bölümleri birleştir
        return "\n\n".join(doc.page_content for doc in docs).strip()
    except Exception as e:
        return f"Dosya okunamadı: {e}"


# =============================================================================
# 🛠️  Araç Fonksiyonları
# =============================================================================
#
# RAG agent'ında araçlar @tool dekoratörü ile tanımlanmıştı.
# CV agent'ında ise araçları düz Python fonksiyonu olarak tutuyoruz;
# her araç state'i alıp güncellenmiş state döndürüyor.
# Bu sayede araçlar arası veri akışı (cv_info → match_skills → gap_analysis)
# state üzerinden temiz ve izlenebilir şekilde gerçekleşiyor.
#
# Araç çağırma akışı:
#   1. extract_cv_info  → cv_info alanını doldurur
#   2. match_skills     → cv_info + job_text kullanır → match_result doldurur
#   3. gap_analysis     → cv_info + job_text kullanır → gap_result doldurur
#   4. generate_report  → tüm sonuçları birleştirerek rapor üretir

def tool_extract_cv_info(state: CVAgentState) -> dict:
    """
    Araç 1: CV bilgilerini yapılandırılmış formata çıkarır.

    LLM'e CV metnini verir ve şu kategorilerde bilgi çıkarmasını ister:
      - Kişisel bilgiler
      - Eğitim geçmişi
      - İş deneyimi
      - Teknik ve soft beceriler
      - Sertifikalar ve başarılar

    Bu çıktı diğer araçların girdisi olacak; bu yüzden yapılandırılmış
    ve net olması önemli.
    """
    cv_text = state.get("cv_text", "")
    if not cv_text:
        result = "CV metni bulunamadı."
        return {
            "cv_info": result,
            "messages": [HumanMessage(content=f"Araç sonucu (extract_cv_info):\n{result}")]
        }

    prompt = f"""Aşağıdaki CV metnini analiz et ve bilgileri yapılandırılmış olarak çıkar.

CV Metni:
{cv_text[:4000]}

Lütfen şu başlıklar altında bilgileri listele:

## 👤 Kişisel Bilgiler
(İsim, konum, iletişim vb.)

## 🎓 Eğitim
(Okul, bölüm, mezuniyet yılı)

## 💼 İş Deneyimi
(Şirket, pozisyon, süre, sorumluluklar)

## 🔧 Teknik Beceriler
(Programlama dilleri, araçlar, teknolojiler)

## 🤝 Soft Beceriler
(İletişim, liderlik, takım çalışması vb.)

## 🏆 Sertifikalar ve Başarılar
(Varsa)

Türkçe yaz. Bilgi yoksa o başlığı "Belirtilmemiş" olarak işaretle."""

    response  = _LLM.invoke(prompt)
    result    = response.content.strip()

    return {
        "cv_info":  result,
        "messages": [HumanMessage(content=f"Araç sonucu (extract_cv_info):\n{result[:500]}...")]
    }


def tool_match_skills(state: CVAgentState) -> dict:
    """
    Araç 2: CV becerilerini iş ilanı gereksinimleriyle karşılaştırır.

    Hem CV'den çıkarılan bilgileri hem de iş ilanını alarak:
      - Her gereksinim için eşleşme var mı? (✅/❌)
      - Genel uyum skoru (0-100)
      - Güçlü yönler ve zayıf yönler
    üretir.

    Skor hesaplama mantığı:
      LLM'e açık uçlu değerlendirme yaptırmak yerine structured bir format
      istiyoruz; bu sayede raporda tutarlı bir skor gösterilebilir.
    """
    cv_info  = state.get("cv_info", "")
    job_text = state.get("job_text", "")

    if not cv_info or not job_text:
        result = "CV bilgisi veya iş ilanı eksik."
        return {
            "match_result": result,
            "messages": [HumanMessage(content=f"Araç sonucu (match_skills):\n{result}")]
        }

    prompt = f"""Aşağıdaki CV bilgilerini ve iş ilanını karşılaştır.

CV Bilgileri:
{cv_info[:2000]}

İş İlanı:
{job_text[:2000]}

Lütfen şu formatta değerlendir:

## 🎯 Genel Uyum Skoru
X/100 — (açıklama)

## ✅ Güçlü Eşleşmeler
(İş ilanında istenen ve CV'de bulunan beceri/deneyimler)

## ❌ Eksik veya Zayıf Alanlar
(İş ilanında istenen ama CV'de yeterince bulunmayan)

## ⭐ Öne Çıkan Avantajlar
(Adayı diğerlerinden ayıran güçlü yönler)

Türkçe ve nesnel bir dille yaz."""

    response = _LLM.invoke(prompt)
    result   = response.content.strip()

    return {
        "match_result": result,
        "messages": [HumanMessage(content=f"Araç sonucu (match_skills):\n{result[:500]}...")]
    }


def tool_gap_analysis(state: CVAgentState) -> dict:
    """
    Araç 3: Eksik yetkinlikleri bulur ve gelişim yolu önerir.

    match_skills'in "eksik" bulgularını derinleştirir:
      - Hangi beceriler kritik eksik?
      - Bu becerileri kazanmak için somut öneriler
      - Kısa vade (0-3 ay) ve uzun vade (3-12 ay) gelişim planı

    Neden ayrı bir araç?
      match_skills "ne eksik" der; gap_analysis "nasıl kapatılır" der.
      İkisini ayırmak LLM'in her konuya odaklanmasını sağlar.
    """
    cv_info      = state.get("cv_info", "")
    job_text     = state.get("job_text", "")
    match_result = state.get("match_result", "")

    if not match_result:
        result = "Önce beceri eşleştirmesi yapılmalı."
        return {
            "gap_result": result,
            "messages": [HumanMessage(content=f"Araç sonucu (gap_analysis):\n{result}")]
        }

    prompt = f"""Aşağıdaki CV analizi ve beceri eşleştirme sonucuna dayanarak
gelişim yolu öner.

Beceri Eşleştirme Sonucu:
{match_result[:2000]}

İş İlanı:
{job_text[:1000]}

Lütfen şu formatta yaz:

## 🚨 Kritik Eksikler
(Bu pozisyon için mutlaka kazanılması gereken beceriler)

## 📚 Öğrenme Önerileri
Her kritik eksik için:
- Hangi kaynak / kurs / sertifika ile öğrenilebilir?
- Ne kadar sürer?

## 🗓️ Kısa Vade Eylem Planı (0-3 ay)
(Hemen başlanabilecek adımlar)

## 🚀 Uzun Vade Eylem Planı (3-12 ay)
(Kariyer gelişimi için stratejik adımlar)

## 💡 Alternatif Pozisyon Önerileri
(Mevcut profille daha uyumlu olabilecek roller)

Türkçe ve yapıcı bir dille yaz."""

    response = _LLM.invoke(prompt)
    result   = response.content.strip()

    return {
        "gap_result": result,
        "messages": [HumanMessage(content=f"Araç sonucu (gap_analysis):\n{result[:500]}...")]
    }


def tool_generate_report(state: CVAgentState) -> dict:
    """
    Araç 4: Tüm analiz sonuçlarını tek bir yönetici raporuna dönüştürür.

    Önceki 3 aracın çıktılarını alır ve:
      - Yönetici özeti (executive summary)
      - Tüm bulguların derlemesi
      - İşe alım kararı için öneri
    içeren profesyonel bir rapor üretir.

    Bu araç diğerlerine bağımlı — önce 3 araç çalışmış olmalı.
    Agent sistem promptu sayesinde bu sıralamayı kendi belirler.
    """
    cv_info      = state.get("cv_info", "Analiz yapılmadı.")
    match_result = state.get("match_result", "Eşleştirme yapılmadı.")
    gap_result   = state.get("gap_result", "Gap analizi yapılmadı.")
    job_text     = state.get("job_text", "")

    prompt = f"""Aşağıdaki CV analiz sonuçlarını kullanarak profesyonel bir değerlendirme raporu oluştur.

=== CV BİLGİLERİ ===
{cv_info[:1500]}

=== BECERİ EŞLEŞTİRMESİ ===
{match_result[:1500]}

=== GELİŞİM ANALİZİ ===
{gap_result[:1500]}

=== İŞ İLANI ===
{job_text[:500]}

Lütfen şu formatta kapsamlı bir rapor oluştur:

# 📊 CV DEĞERLENDİRME RAPORU
**Tarih:** {datetime.now().strftime("%d.%m.%Y %H:%M")}

## 📌 Yönetici Özeti
(2-3 cümlelik genel değerlendirme)

## 👤 Aday Profili
(Adayın güçlü profil özeti)

## 🎯 Pozisyona Uyum
(Genel skor ve detaylı uyum değerlendirmesi)

## ✅ Güçlü Yönler
(Adayın öne çıkan avantajları)

## ⚠️ Geliştirilmesi Gereken Alanlar
(Eksikler ve öneriler)

## 📋 İşe Alım Önerisi
☐ Güçlü Aday — Görüşmeye çağır
☐ Orta Düzey Aday — Teknik sınav öner
☐ Deneyim Eksik — Gelecek dönem değerlendir
☐ Uygun Değil — Farklı pozisyon öner

**Karar gerekçesi:** (kısa açıklama)

## 🚀 Sonraki Adımlar
(İşe alım süreci için önerilen adımlar)

Türkçe, nesnel ve profesyonel bir dille yaz."""

    response = _LLM.invoke(prompt)
    result   = response.content.strip()

    # Raporu JSON log olarak kaydet
    log_path = os.path.join(
        REPORT_DIR,
        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(result)

    return {
        "report":   result,
        "messages": [HumanMessage(content=f"Araç sonucu (generate_report): Rapor oluşturuldu → {log_path}")]
    }


# Araç haritası — tool_node bu sözlükten fonksiyonu çağırır
TOOL_FUNCTIONS = {
    "extract_cv_info":  tool_extract_cv_info,
    "match_skills":     tool_match_skills,
    "gap_analysis":     tool_gap_analysis,
    "generate_report":  tool_generate_report,
}


# =============================================================================
# 🤖  ReAct Parser (RAG agent ile aynı mantık)
# =============================================================================

def _parse_react_output(text: str) -> dict:
    """
    LLM'in ReAct formatındaki çıktısını parse eder.
    RAG agent ile birebir aynı fonksiyon — yeniden kullanım örneği.

    Beklenen format:
      Araç çağrısı → Action: <araç_adı> / Action Input: <girdi>
      Final cevap  → Final Answer: <cevap>
    """
    action_match = re.search(r"Action\s*:\s*(\w+)", text, re.IGNORECASE)
    input_match  = re.search(
        r"Action Input\s*:\s*(.+?)(?=\n(?:Thought|Final|Action)|$)",
        text, re.IGNORECASE | re.DOTALL
    )

    if action_match:
        tool_name  = action_match.group(1).strip()
        tool_input = input_match.group(1).strip() if input_match else ""
        return {"type": "tool", "tool": tool_name, "input": tool_input}

    final_match = re.search(r"Final Answer\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if final_match:
        return {"type": "answer", "content": final_match.group(1).strip()}

    # Fallback: LLM formatı tutmadıysa cevap olarak kabul et
    return {"type": "answer", "content": text.strip()}


# =============================================================================
# 📝  Sistem Promptu — CV değerlendirme uzmanı rolü
# =============================================================================
#
# RAG agent'ından farkı:
#   - "rag_search" yerine 4 adet CV aracı tanıtılıyor
#   - Agent'a araçları belirli bir SIRADA kullanması söyleniyor
#   - Bu sıra önemli: önce çıkar, sonra karşılaştır, sonra gap bul, son rapor yaz

SYSTEM_PROMPT_TEXT = """Sen deneyimli bir İnsan Kaynakları ve kariyer danışmanısın.
CV'leri analiz edip kapsamlı değerlendirme raporları üretiyorsun.

Elindeki araçlar ve kullanım sırası:

1. extract_cv_info  → CV metninden yapılandırılmış bilgi çıkar (HER ZAMAN İLK ÇALIŞTIR)
2. match_skills     → CV ile iş ilanını karşılaştır (extract_cv_info'dan sonra)
3. gap_analysis     → Eksik yetkinlikleri analiz et (match_skills'den sonra)
4. generate_report  → Kapsamlı rapor oluştur (diğer 3 araçtan sonra)

ÖNEMLİ: Her yanıtında MUTLAKA şu formatlardan birini kullan:

Araç çalıştırmak için:
Thought: [ne yapacağını açıkla]
Action: [araç_adı]
Action Input: başla

Final cevap için (tüm araçlar tamamlandıktan sonra):
Thought: [tüm analizler tamamlandı]
Final Answer: Değerlendirme tamamlandı. Rapor hazır.

Araçları sırayla çalıştır, hiçbirini atlama.
Türkçe yanıt ver."""


# =============================================================================
# 🔵  Node 1: agent_node — LLM karar verici
# =============================================================================

def agent_node(state: CVAgentState) -> dict:
    """
    CV Agent'ının karar verme düğümü.

    RAG agent'ından farkı:
      - Sistem promptu CV değerlendirme uzmanı rolünde
      - LLM'e hangi aracı sıradaki çalıştıracağını messages geçmişinden anlıyor
        (önceki araç sonuçları messages'a eklendiği için LLM sırayı biliyor)

    Döngü tamamlandığında (4 araç çalıştıktan sonra) Final Answer üretir.
    """
    system_msg = SystemMessage(content=SYSTEM_PROMPT_TEXT)
    messages   = [system_msg] + state["messages"]

    response = _LLM.invoke(messages)
    raw_text = response.content
    parsed   = _parse_react_output(raw_text)

    if parsed["type"] == "tool":
        ai_msg = AIMessage(
            content=raw_text,
            additional_kwargs={
                "react_tool":  parsed["tool"],
                "react_input": parsed.get("input", ""),
            }
        )
    else:
        ai_msg = AIMessage(content=parsed.get("content", raw_text))

    return {"messages": [ai_msg]}


# =============================================================================
# 🔀  Conditional Edge: should_continue
# =============================================================================

def should_continue(state: CVAgentState) -> str:
    """
    RAG agent ile aynı mantık:
      - additional_kwargs'da react_tool varsa → tools düğümüne git
      - Yoksa → END
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        if last_msg.additional_kwargs.get("react_tool"):
            return "tools"
    return END


# =============================================================================
# 🟢  Node 2: tool_node — Araç çalıştırıcı
# =============================================================================

def tool_node(state: CVAgentState) -> dict:
    """
    CV araçlarını çalıştıran düğüm.

    RAG agent'ından farkı:
      CV araçları state'in birden fazla alanını günceller
      (cv_info, match_result, gap_result, report).
      Her araç fonksiyonu tüm state'i alır, güncellenmiş alanları döner.
    """
    last_msg  = state["messages"][-1]
    tool_name = last_msg.additional_kwargs.get("react_tool", "")

    print(f"🔧 Araç çalışıyor: {tool_name}")

    if tool_name in TOOL_FUNCTIONS:
        # Araç fonksiyonunu çağır; state'i girdi olarak ver
        updates = TOOL_FUNCTIONS[tool_name](state)
    else:
        updates = {
            "messages": [HumanMessage(content=f"Bilinmeyen araç: {tool_name}")]
        }

    return updates


# =============================================================================
# 🏗️  Graf İnşası
# =============================================================================

def build_cv_graph():
    """
    CV değerlendirme grafını oluşturur.

    RAG agent grafıyla birebir aynı yapı:
      START → agent → (should_continue) → tools → agent → ... → END

    Tek fark: StateGraph'a CVAgentState şeması veriliyor.
    """
    graph = StateGraph(CVAgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )

    graph.add_edge("tools", "agent")

    return graph.compile()


# Global agent
CV_AGENT = build_cv_graph()


# =============================================================================
# 🚀  CV Değerlendirme Çalıştırıcı
# =============================================================================

def run_cv_evaluation(cv_text: str, job_text: str) -> tuple[str, str, str, str, str]:
    """
    CV değerlendirme agent'ını çalıştırır.

    Args:
        cv_text:  PDF'den çıkarılan CV metni
        job_text: Kullanıcının girdiği iş ilanı metni

    Returns:
        (cv_info, match_result, gap_result, report, trace_log)
        Her değer Gradio'da ayrı bir panelde gösterilir.
    """
    if not cv_text.strip():
        return "", "", "", "", "⚠️ CV metni boş. Önce CV yükleyin."
    if not job_text.strip():
        return "", "", "", "", "⚠️ İş ilanı boş. İş ilanını girin."

    # Başlangıç state — cv_text ve job_text dolu, diğerleri boş
    initial_state: CVAgentState = {
        "messages":     [HumanMessage(content="CV değerlendirmesini başlat. Tüm araçları sırayla çalıştır.")],
        "cv_text":      cv_text,
        "job_text":     job_text,
        "cv_info":      "",
        "match_result": "",
        "gap_result":   "",
        "report":       "",
    }

    trace_lines = ["### 🔍 Agent Adım İzleme\n"]
    step = 0

    # Graf çıktısını biriktir
    final_state = initial_state.copy()

    try:
        for event in CV_AGENT.stream(initial_state):
            step += 1
            for node_name, node_output in event.items():

                # State alanlarını güncelle (cv_info, match_result vb.)
                for key in ["cv_info", "match_result", "gap_result", "report"]:
                    if key in node_output and node_output[key]:
                        final_state[key] = node_output[key]

                # Trace log
                messages = node_output.get("messages", [])
                for msg in messages:
                    if node_name == "agent" and isinstance(msg, AIMessage):
                        react_tool = msg.additional_kwargs.get("react_tool")
                        if react_tool:
                            trace_lines.append(
                                f"**Adım {step} — Araç Kararı**\n"
                                f"→ `{react_tool}` çalıştırılıyor...\n"
                            )
                        elif msg.content:
                            trace_lines.append(
                                f"**Adım {step} — Agent Kararı**\n"
                                f"_{msg.content[:120]}..._\n"
                                if len(msg.content) > 120
                                else f"**Adım {step} — Agent Kararı**\n_{msg.content}_\n"
                            )

                    elif node_name == "tools" and isinstance(msg, HumanMessage):
                        # Araç sonucunun ilk 150 karakteri
                        snippet = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                        trace_lines.append(
                            f"**Adım {step} — Araç Tamamlandı**\n"
                            f"```\n{snippet}\n```\n"
                        )

    except Exception as e:
        trace_lines.append(f"\n❌ Hata: {e}")
        return "", "", "", "", "\n".join(trace_lines)

    trace_lines.append(f"\n✅ Değerlendirme tamamlandı — {step} adım")

    return (
        final_state.get("cv_info",      ""),
        final_state.get("match_result", ""),
        final_state.get("gap_result",   ""),
        final_state.get("report",       ""),
        "\n".join(trace_lines),
    )


# =============================================================================
# 📁  CV Dosyası Yükleme
# =============================================================================

def load_cv_file(file_path: str) -> tuple[str, str]:
    """
    CV dosyasını yükler ve metni çıkarır.
    Gradio File bileşeninden gelen geçici yolu alır.

    Returns:
        (cv_text, status_message)
    """
    if not file_path:
        return "", "Dosya seçilmedi."
    try:
        basename = os.path.basename(file_path)
        dest     = os.path.join(CV_DIR, basename)
        shutil.copy(file_path, dest)
        text     = extract_text_from_file(dest)
        if not text or text.startswith("Dosya okunamadı"):
            return "", f"⚠️ {text}"
        word_count = len(text.split())
        return text, f"✅ {basename} yüklendi ({word_count} kelime)"
    except Exception as e:
        return "", f"❌ Hata: {e}"


# =============================================================================
# 🖥️  Gradio Arayüzü
# =============================================================================

with gr.Blocks(title="CV Değerlendirme Agent", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📄 CV Değerlendirme Agent")
    gr.Markdown(
        "CV'yi yükle, iş ilanını yapıştır, **Değerlendirmeyi Başlat** butonuna bas. "
        "Agent 4 araçla otomatik analiz yapar ve rapor üretir."
    )

    # State: CV metni (dosyadan okunur, paneller arası paylaşılır)
    cv_text_state = gr.State("")

    with gr.Tabs():

        # ── Sekme 1: Giriş ───────────────────────────────────────────────────
        with gr.Tab("📥 Giriş"):
            with gr.Row():

                # Sol: CV yükleme
                with gr.Column():
                    gr.Markdown("### CV Dosyası")
                    cv_file    = gr.File(label="CV yükle (PDF veya TXT)", type="filepath")
                    cv_status  = gr.Textbox(label="Durum", interactive=False)
                    cv_preview = gr.Textbox(
                        label="CV Metni Önizleme",
                        lines=12,
                        interactive=False,
                        placeholder="CV yüklendikten sonra metin burada görünür..."
                    )

                # Sağ: İş ilanı
                with gr.Column():
                    gr.Markdown("### İş İlanı")
                    job_input = gr.Textbox(
                        label="İş ilanını buraya yapıştır",
                        lines=15,
                        placeholder="Örn:\nPozisyon: Senior Python Developer\n\nGereksinimler:\n- 5+ yıl Python deneyimi\n- FastAPI, Django\n- PostgreSQL, Redis\n- Docker, Kubernetes\n..."
                    )

            eval_btn    = gr.Button("🚀 Değerlendirmeyi Başlat", variant="primary", size="lg")
            eval_status = gr.Textbox(label="İşlem Durumu", interactive=False)

        # ── Sekme 2: Analiz Sonuçları ─────────────────────────────────────────
        with gr.Tab("📊 Analiz Sonuçları"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 👤 CV Profili")
                    cv_info_out = gr.Markdown(
                        value="*Değerlendirme başlatıldıktan sonra burada görünür.*"
                    )
                with gr.Column():
                    gr.Markdown("### 🎯 Beceri Eşleştirme")
                    match_out = gr.Markdown(
                        value="*Değerlendirme başlatıldıktan sonra burada görünür.*"
                    )

        # ── Sekme 3: Gelişim Planı ────────────────────────────────────────────
        with gr.Tab("📈 Gelişim Planı"):
            gap_out = gr.Markdown(
                value="*Gap analizi burada görünecek.*"
            )

        # ── Sekme 4: Rapor ────────────────────────────────────────────────────
        with gr.Tab("📋 Değerlendirme Raporu"):
            report_out = gr.Markdown(
                value="*Final rapor burada görünecek.*"
            )
            download_btn = gr.Button("💾 Raporu İndir (.txt)")
            download_out = gr.File(label="İndirilen rapor", visible=False)

        # ── Sekme 5: Agent İzleme ─────────────────────────────────────────────
        with gr.Tab("🔍 Agent Adımları"):
            gr.Markdown(
                "Agent'ın hangi araçları hangi sırayla çalıştırdığını buradan izleyebilirsin."
            )
            trace_out = gr.Markdown(
                value="*Değerlendirme başlatıldıktan sonra adımlar burada görünür.*"
            )

    # ── Event Bağlantıları ────────────────────────────────────────────────────

    # CV dosyası yüklendiğinde metni çıkar ve önizle
    def on_cv_upload(file_path):
        text, status = load_cv_file(file_path)
        # İlk 1000 karakter önizleme için yeterli
        preview = text[:1000] + "\n...(devamı var)" if len(text) > 1000 else text
        return text, status, preview

    cv_file.change(
        fn=on_cv_upload,
        inputs=[cv_file],
        outputs=[cv_text_state, cv_status, cv_preview]
    )

    # Değerlendirme başlat
    def on_evaluate(cv_text, job_text):
        if not cv_text.strip():
            return (
                "*CV yüklenmedi.*", "*CV yüklenmedi.*",
                "*CV yüklenmedi.*", "*CV yüklenmedi.*",
                "*CV yüklenmedi.*", "⚠️ Önce CV yükleyin."
            )
        if not job_text.strip():
            return (
                "*İş ilanı girilmedi.*", "*İş ilanı girilmedi.*",
                "*İş ilanı girilmedi.*", "*İş ilanı girilmedi.*",
                "*İş ilanı girilmedi.*", "⚠️ İş ilanını girin."
            )

        status = "⏳ Agent çalışıyor — bu işlem 2-4 dakika sürebilir..."
        cv_info, match, gap, report, trace = run_cv_evaluation(cv_text, job_text)
        done_status = "✅ Değerlendirme tamamlandı."
        return cv_info, match, gap, report, trace, done_status

    eval_btn.click(
        fn=on_evaluate,
        inputs=[cv_text_state, job_input],
        outputs=[cv_info_out, match_out, gap_out, report_out, trace_out, eval_status]
    )

    # Raporu indir
    def on_download(report_text):
        if not report_text or report_text.startswith("*"):
            return gr.File(visible=False)
        path = os.path.join(
            REPORT_DIR,
            f"rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return gr.File(value=path, visible=True)

    download_btn.click(
        fn=on_download,
        inputs=[report_out],
        outputs=[download_out]
    )


demo.launch(server_name="127.0.0.1", server_port=7862)
# Port 7862: RAG agent (7861) ve v8 (7860) ile çakışmaz
