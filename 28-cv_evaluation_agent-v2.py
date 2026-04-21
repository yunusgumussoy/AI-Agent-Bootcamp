# =============================================================================
# 📄 CV Değerlendirme Agent v2
# =============================================================================
#
# v2 hedefleri:
# - v1'deki üretim kalitesindeki Gradio arayüzünü koru
# - Structured çıktı üret (JSON + Markdown rapor)
# - Critic ile kaliteyi değerlendir
# - Gerekirse kontrollü revizyon döngüsü çalıştır
#
# Kurulum:
# pip install langgraph langchain langchain-ollama gradio pypdf langchain-community
#
# Çalıştırma:
# python 28-cv_evaluation_agent-v2.py

import os
import re
import json
import shutil
from datetime import datetime
from typing import Annotated, Optional, TypedDict

import gradio as gr
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.messages import HumanMessage


# =============================================================================
# ⚙️ Konfigürasyon
# =============================================================================

MODEL_MAIN = "llama3"
MODEL_CRITIC = "llama3"

CV_DIR = "cv_files"
REPORT_DIR = "cv_reports_v2"

os.makedirs(CV_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

LLM_MAIN = ChatOllama(model=MODEL_MAIN, temperature=0)
LLM_CRITIC = ChatOllama(model=MODEL_CRITIC, temperature=0)


# =============================================================================
# 🧠 State
# =============================================================================

class CVAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    cv_text: str
    job_text: str
    cv_info: str
    match_result: str
    gap_result: str
    report_markdown: str
    report_json: str
    critic: str
    revision_count: int
    max_revisions: int


# =============================================================================
# 📁 CV metni çıkarma
# =============================================================================

def extract_text_from_file(path: str) -> str:
    try:
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        return "\n\n".join(doc.page_content for doc in docs).strip()
    except Exception as exc:
        return f"Dosya okunamadı: {exc}"


# =============================================================================
# 🧩 Structured çıktı yardımcıları
# =============================================================================

def _extract_tag_block(text: str, tag: str) -> Optional[str]:
    """<TAG>...</TAG> arasındaki içeriği döndürür (iç içe süslü parantez güvenli)."""
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = text.find(open_tag)
    if start == -1:
        return None
    start += len(open_tag)
    end = text.find(close_tag, start)
    if end == -1:
        return None
    return text[start:end].strip()


def split_structured_output(raw_text: str) -> tuple[str, str]:
    """
    Model çıktısından <JSON>...</JSON> ve <REPORT>...</REPORT> bloklarını ayıklar.
    """
    json_raw = _extract_tag_block(raw_text, "JSON")
    report_raw = _extract_tag_block(raw_text, "REPORT")

    json_text = json_raw if json_raw is not None else "{}"
    report_text = report_raw.strip() if report_raw is not None else raw_text.strip()

    try:
        parsed = json.loads(json_text)
        json_text = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        json_text = json.dumps(
            {
                "score": -1,
                "decision": "unknown",
                "strengths": [],
                "weaknesses": [],
                "missing_skills": [],
                "risk_flags": ["JSON parse edilemedi"],
                "summary": "Model structured JSON formatını tam üretemedi.",
            },
            ensure_ascii=False,
            indent=2,
        )

    return json_text, report_text


def get_critic_score(critic_text: str) -> Optional[int]:
    """
    Critic metninden 0-100 skorunu çıkarır.
    Belirsizse None döner (yanlış revizyon tetiklememek için).
    """
    if not critic_text.strip():
        return None

    # Önce "X/100" formatı (en güvenilir)
    for m in re.finditer(r"(\d{1,3})\s*/\s*100", critic_text):
        return max(0, min(100, int(m.group(1))))

    # "## Guven Skoru" veya "Guven Skoru" satırındaki ilk 0-100
    section = re.search(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?Guven\s+Skoru\s*:?\s*(\d{1,3})\s*(?:/100)?",
        critic_text,
        re.IGNORECASE | re.MULTILINE,
    )
    if section:
        return max(0, min(100, int(section.group(1))))

    return None


# =============================================================================
# 🛠️ Analiz nodeları
# =============================================================================

def node_extract_cv_info(state: CVAgentState) -> dict:
    cv_text = state.get("cv_text", "")
    if not cv_text.strip():
        return {
            "cv_info": "CV metni bulunamadı.",
            "messages": [HumanMessage(content="extract_cv_info: CV metni boş")],
        }

    prompt = f"""Aşağıdaki CV metnini analiz et ve yapılandırılmış olarak çıkar.

CV Metni:
{cv_text[:5000]}

Çıktıyı Türkçe üret ve şu başlıkları kullan:
## Kişisel Bilgiler
## Eğitim
## İş Deneyimi
## Teknik Beceriler
## Soft Beceriler
## Sertifikalar ve Başarılar

Bilgi yoksa "Belirtilmemiş" yaz."""

    result = (LLM_MAIN.invoke(prompt).content or "").strip()
    return {
        "cv_info": result,
        "messages": [HumanMessage(content="extract_cv_info tamamlandı")],
    }


def node_match_skills(state: CVAgentState) -> dict:
    cv_info = state.get("cv_info", "")
    job_text = state.get("job_text", "")
    if not cv_info.strip() or not job_text.strip():
        return {
            "match_result": "CV bilgisi veya iş ilanı eksik.",
            "messages": [HumanMessage(content="match_skills: eksik girdi")],
        }

    prompt = f"""Aşağıdaki CV profili ile iş ilanını karşılaştır.

CV Bilgileri:
{cv_info[:3000]}

İş İlanı:
{job_text[:2500]}

Türkçe, nesnel ve ölçülebilir bir çıktı üret:
## Genel Uyum Skoru
X/100 + kısa gerekçe

## Güçlü Eşleşmeler
## Eksik veya Zayıf Alanlar
## Öne Çıkan Avantajlar
## İşe Alım Riski Notları"""

    result = (LLM_MAIN.invoke(prompt).content or "").strip()
    return {
        "match_result": result,
        "messages": [HumanMessage(content="match_skills tamamlandı")],
    }


def node_gap_analysis(state: CVAgentState) -> dict:
    match_result = state.get("match_result", "")
    job_text = state.get("job_text", "")
    if not match_result.strip():
        return {
            "gap_result": "Önce beceri eşleştirme yapılmalı.",
            "messages": [HumanMessage(content="gap_analysis: match sonucu yok")],
        }

    prompt = f"""Aşağıdaki eşleştirme ve iş ilanına göre aday için gelişim planı oluştur.

Eşleştirme Sonucu:
{match_result[:2800]}

İş İlanı:
{job_text[:1800]}

Türkçe cevapla ve şu başlıkları kullan:
## Kritik Eksikler
## Öğrenme Önerileri
## Kısa Vade Planı (0-3 ay)
## Uzun Vade Planı (3-12 ay)
## Alternatif Pozisyon Önerileri"""

    result = (LLM_MAIN.invoke(prompt).content or "").strip()
    return {
        "gap_result": result,
        "messages": [HumanMessage(content="gap_analysis tamamlandı")],
    }


def _report_prompt(
    cv_info: str,
    match_result: str,
    gap_result: str,
    job_text: str,
    revision_hint: str = "",
) -> str:
    return f"""Aşağıdaki analizi kullanarak profesyonel CV değerlendirme raporu üret.

CV BİLGİLERİ:
{cv_info[:2500]}

BECERİ EŞLEŞTİRMESİ:
{match_result[:2500]}

GELİŞİM ANALİZİ:
{gap_result[:2500]}

İŞ İLANI:
{job_text[:1500]}

REVİZYON İPUCU:
{revision_hint if revision_hint else "İlk versiyon rapor üretiliyor."}

MUTLAKA iki blok üret:
1) <JSON> ... </JSON>
2) <REPORT> ... </REPORT>

JSON alanları:
{{
  "score": 0-100,
  "decision": "strong_yes | yes | maybe | no",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "missing_skills": ["..."],
  "risk_flags": ["..."],
  "summary": "2-3 cümle"
}}

REPORT bloğu Türkçe ve şu bölümleri içersin:
# CV DEGERLENDIRME RAPORU
## Yonetici Ozeti
## Aday Profili
## Pozisyona Uyum
## Guclu Yonler
## Gelistirilmesi Gereken Alanlar
## Ise Alim Onerisi
## Sonraki Adimlar

JSON dışında ekstra JSON benzeri blok üretme."""


def node_generate_report(state: CVAgentState) -> dict:
    prompt = _report_prompt(
        cv_info=state.get("cv_info", ""),
        match_result=state.get("match_result", ""),
        gap_result=state.get("gap_result", ""),
        job_text=state.get("job_text", ""),
    )

    raw = (LLM_MAIN.invoke(prompt).content or "").strip()
    report_json, report_markdown = split_structured_output(raw)
    return {
        "report_json": report_json,
        "report_markdown": report_markdown,
        "messages": [HumanMessage(content="generate_report tamamlandı")],
    }


def node_critic_review(state: CVAgentState) -> dict:
    report_json = state.get("report_json", "{}")
    report_markdown = state.get("report_markdown", "")

    prompt = f"""Aşağıdaki rapor çıktısını kalite açısından değerlendir.

JSON:
{report_json[:2000]}

RAPOR:
{report_markdown[:3000]}

Türkçe ve net bir değerlendirme üret. Format:
## Kritik Hatalar
- ...

## Eksikler
- ...

## Guclu Yanitlar
- ...

## Guven Skoru
X/100

Kurallar:
- Skor 0-100 arası olsun
- Major tutarsızlık varsa skoru düşür
- Sadece değerlendirme yaz, raporu tekrar etme"""

    critic = (LLM_CRITIC.invoke(prompt).content or "").strip()
    return {
        "critic": critic,
        "messages": [HumanMessage(content="critic_review tamamlandı")],
    }


def node_revise_report(state: CVAgentState) -> dict:
    revision_hint = f"""Aşağıdaki critic geri bildirimlerine göre raporu iyileştir:
{state.get("critic", "")[:2500]}"""

    prompt = _report_prompt(
        cv_info=state.get("cv_info", ""),
        match_result=state.get("match_result", ""),
        gap_result=state.get("gap_result", ""),
        job_text=state.get("job_text", ""),
        revision_hint=revision_hint,
    )

    raw = (LLM_MAIN.invoke(prompt).content or "").strip()
    report_json, report_markdown = split_structured_output(raw)
    revision_count = state.get("revision_count", 0) + 1

    return {
        "report_json": report_json,
        "report_markdown": report_markdown,
        "revision_count": revision_count,
        "messages": [
            HumanMessage(content=f"revise_report tamamlandı (revizyon: {revision_count})")
        ],
    }


def route_after_critic(state: CVAgentState) -> str:
    score = get_critic_score(state.get("critic", ""))
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 1)

    # Skor çıkarılamadıysa revizyon yapma (yanlış pozitif döngü riski)
    if score is not None and score < 75 and revision_count < max_revisions:
        return "revise"
    return "finalize"


# =============================================================================
# 🏗️ GRAPH
# =============================================================================

def build_cv_graph():
    graph = StateGraph(CVAgentState)

    graph.add_node("extract_cv_info", node_extract_cv_info)
    graph.add_node("match_skills", node_match_skills)
    graph.add_node("gap_analysis", node_gap_analysis)
    graph.add_node("generate_report", node_generate_report)
    graph.add_node("critic_review", node_critic_review)
    graph.add_node("revise_report", node_revise_report)

    graph.set_entry_point("extract_cv_info")
    graph.add_edge("extract_cv_info", "match_skills")
    graph.add_edge("match_skills", "gap_analysis")
    graph.add_edge("gap_analysis", "generate_report")
    graph.add_edge("generate_report", "critic_review")

    graph.add_conditional_edges(
        "critic_review",
        route_after_critic,
        {"revise": "revise_report", "finalize": END},
    )
    graph.add_edge("revise_report", "critic_review")
    return graph.compile()


CV_AGENT = build_cv_graph()


# =============================================================================
# 🚀 Çalıştırıcı
# =============================================================================

def run_cv_evaluation(
    cv_text: str, job_text: str
) -> tuple[str, str, str, str, str, str, str]:
    if not cv_text.strip():
        return "", "", "", "", "", "", "⚠️ CV metni boş. Önce CV yükleyin."
    if not job_text.strip():
        return "", "", "", "", "", "", "⚠️ İş ilanı boş. İş ilanını girin."

    initial_state: CVAgentState = {
        "messages": [HumanMessage(content="CV değerlendirmesini başlat.")],
        "cv_text": cv_text,
        "job_text": job_text,
        "cv_info": "",
        "match_result": "",
        "gap_result": "",
        "report_markdown": "",
        "report_json": "{}",
        "critic": "",
        "revision_count": 0,
        "max_revisions": 1,
    }

    final_state = initial_state.copy()
    trace_lines = ["### Agent Adım İzleme\n"]
    step = 0

    try:
        for event in CV_AGENT.stream(initial_state):
            step += 1
            for node_name, node_output in event.items():
                for key in [
                    "cv_info",
                    "match_result",
                    "gap_result",
                    "report_markdown",
                    "report_json",
                    "critic",
                    "revision_count",
                ]:
                    if key in node_output:
                        final_state[key] = node_output[key]

                msg = node_output.get("messages", [])
                if msg:
                    trace_lines.append(f"**Adım {step} — {node_name}**")
                    trace_lines.append(f"- {msg[-1].content}")

            trace_lines.append("")

    except Exception as exc:
        trace_lines.append(f"❌ Hata: {exc}")
        return "", "", "", "", "", "\n".join(trace_lines), "❌ Agent çalışırken hata oluştu."

    report_text = final_state.get("report_markdown", "")
    if report_text:
        log_path = os.path.join(
            REPORT_DIR, f"report_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(log_path, "w", encoding="utf-8") as fp:
            fp.write(report_text)
        trace_lines.append(f"Rapor kaydedildi: `{log_path}`")

    score = get_critic_score(final_state.get("critic", ""))
    score_part = f"{score}/100" if score is not None else "belirlenemedi"
    status = (
        "✅ Değerlendirme tamamlandı."
        f" Critic skoru: {score_part} | Revizyon: {final_state.get('revision_count', 0)}"
    )

    return (
        final_state.get("cv_info", ""),
        final_state.get("match_result", ""),
        final_state.get("gap_result", ""),
        final_state.get("report_markdown", ""),
        final_state.get("critic", ""),
        "\n".join(trace_lines),
        status,
    )


# =============================================================================
# 📁 CV yükleme
# =============================================================================

def load_cv_file(file_path: str) -> tuple[str, str]:
    if not file_path:
        return "", "Dosya seçilmedi."
    try:
        basename = os.path.basename(file_path)
        dest = os.path.join(CV_DIR, basename)
        shutil.copy(file_path, dest)
        text = extract_text_from_file(dest)
        if not text or text.startswith("Dosya okunamadı"):
            return "", f"⚠️ {text}"
        return text, f"✅ {basename} yüklendi ({len(text.split())} kelime)"
    except Exception as exc:
        return "", f"❌ Hata: {exc}"


# =============================================================================
# 🖥️ Gradio UI
# =============================================================================

with gr.Blocks(title="CV Değerlendirme Agent v2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CV Değerlendirme Agent v2")
    gr.Markdown(
        "CV yukle + is ilani gir + degerlendirmeyi baslat. "
        "v2: structured JSON, critic ve kontrollu rapor revizyonu."
    )

    cv_text_state = gr.State("")

    with gr.Tabs():
        with gr.Tab("Giris"):
            with gr.Row():
                with gr.Column():
                    cv_file = gr.File(label="CV yukle (PDF veya TXT)", type="filepath")
                    cv_status = gr.Textbox(label="CV Durumu", interactive=False)
                    cv_preview = gr.Textbox(
                        label="CV Metni Onizleme", lines=12, interactive=False
                    )
                with gr.Column():
                    job_input = gr.Textbox(
                        label="Is ilani metni",
                        lines=15,
                        placeholder="Pozisyon, gereksinimler, beklentiler...",
                    )

            eval_btn = gr.Button("Degerlendirmeyi Baslat", variant="primary", size="lg")
            eval_status = gr.Textbox(label="Islem Durumu", interactive=False)

        with gr.Tab("Analiz Sonuclari"):
            with gr.Row():
                with gr.Column():
                    cv_info_out = gr.Markdown(
                        value="*CV analizi burada gorunecek.*"
                    )
                with gr.Column():
                    match_out = gr.Markdown(
                        value="*Beceri eslestirmesi burada gorunecek.*"
                    )
            gap_out = gr.Markdown(value="*Gelisim plani burada gorunecek.*")

        with gr.Tab("Final Rapor"):
            report_out = gr.Markdown(value="*Final rapor burada gorunecek.*")
            download_btn = gr.Button("Raporu indir (.txt)")
            download_out = gr.File(label="Rapor dosyasi", visible=False)

        with gr.Tab("Critic"):
            critic_out = gr.Markdown(value="*Critic geri bildirimi burada gorunecek.*")

        with gr.Tab("Agent Adimlari"):
            trace_out = gr.Markdown(value="*Agent adimlari burada gorunecek.*")

    def on_cv_upload(file_path: str):
        text, status = load_cv_file(file_path)
        preview = text[:1000] + "\n...(devami var)" if len(text) > 1000 else text
        return text, status, preview

    cv_file.change(
        fn=on_cv_upload,
        inputs=[cv_file],
        outputs=[cv_text_state, cv_status, cv_preview],
    )

    def on_evaluate(cv_text: str, job_text: str):
        if not cv_text.strip():
            msg = "*CV yuklenmedi.*"
            return msg, msg, msg, msg, msg, msg, "⚠️ Once CV yukleyin."
        if not job_text.strip():
            msg = "*Is ilani girilmedi.*"
            return msg, msg, msg, msg, msg, msg, "⚠️ Is ilani metnini girin."

        cv_info, match, gap, report, critic, trace, status = run_cv_evaluation(
            cv_text, job_text
        )
        return cv_info, match, gap, report, critic, trace, status

    eval_btn.click(
        fn=on_evaluate,
        inputs=[cv_text_state, job_input],
        outputs=[cv_info_out, match_out, gap_out, report_out, critic_out, trace_out, eval_status],
    )

    def on_download(report_text: str):
        if not report_text or report_text.startswith("*"):
            return gr.File(visible=False)
        path = os.path.join(
            REPORT_DIR, f"rapor_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(report_text)
        return gr.File(value=path, visible=True)

    download_btn.click(
        fn=on_download,
        inputs=[report_out],
        outputs=[download_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7863)