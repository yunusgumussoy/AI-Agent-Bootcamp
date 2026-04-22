# =============================================================================
# CV Değerlendirme Agent v3
# =============================================================================
#
# v3 yenilikleri (v2 üzerine):
# - İş ilanı ayrıştırma (zorunlu / tercih / red flag, seviye)
# - Kanıtlı bulgular: her önemli iddia için CV'den kısa alıntı
# - Pydantic ile JSON şema doğrulama + tek seferlik otomatik onarım
# - Adil değerlendirme kuralları (hassas özelliklerden kaçınma)
# - JSON raporu indirme + toplu CV sıralama (aynı ilana göre)
#
# Kurulum:
# pip install langgraph langchain langchain-ollama gradio pypdf langchain-community pydantic
#
# Çalıştırma:
# python 29-cv_evaluation_agent-v3.py
# → http://127.0.0.1:7864

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from typing import Annotated, Any, List, Literal, Optional, TypedDict

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ValidationError, field_validator

# =============================================================================
# Konfigürasyon
# =============================================================================

MODEL_MAIN = "llama3"
MODEL_CRITIC = "llama3"

CV_DIR = "cv_files"
REPORT_DIR = "cv_reports_v3"

os.makedirs(CV_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

LLM_MAIN = ChatOllama(model=MODEL_MAIN, temperature=0)
LLM_CRITIC = ChatOllama(model=MODEL_CRITIC, temperature=0)

# Critic güven skoru bu eşiğin altındaysa rapor bir kez yeniden yazılır (max_revisions ile sınırlı).
CRITIC_REVISE_THRESHOLD = 75

# =============================================================================
# Pydantic şema (v3 structured çıktı)
# =============================================================================


class GroundedFinding(BaseModel):
    """Kanıt: iddia + CV'den birebir veya yakın alıntı."""

    claim: str = Field(..., min_length=3)
    cv_quote: str = Field(..., min_length=3, max_length=600)
    section_hint: str = Field(
        ...,
        description="Örn: Is Deneyimi, Egitim, Beceriler",
    )


class EvalReportV3(BaseModel):
    score: int = Field(ge=0, le=100)
    decision: Literal["strong_yes", "yes", "maybe", "no"]
    subscores: dict[str, int]
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    grounded_findings: List[GroundedFinding] = Field(
        default_factory=list,
        max_length=12,
    )
    missing_skills: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    fairness_notes: List[str] = Field(default_factory=list)
    summary: str = Field(..., min_length=10)

    @field_validator("subscores")
    @classmethod
    def subscores_in_range(cls, v: dict[str, int]) -> dict[str, int]:
        for key, val in v.items():
            if not isinstance(val, int) or not 0 <= val <= 100:
                raise ValueError(f"subscores['{key}'] 0-100 arası tam sayı olmalı")
        expected = {"technical_fit", "experience_depth", "role_level_fit", "communication_evidence"}
        if not expected.issubset(set(v.keys())):
            raise ValueError(
                f"subscores şu anahtarları içermeli: {sorted(expected)}"
            )
        return v


def report_schema_prompt_fragment() -> str:
    return """
JSON alanları (STRICT — tam bu yapı):
{
  "score": 0-100,
  "decision": "strong_yes" | "yes" | "maybe" | "no",
  "subscores": {
    "technical_fit": 0-100,
    "experience_depth": 0-100,
    "role_level_fit": 0-100,
    "communication_evidence": 0-100
  },
  "strengths": ["kısa madde", ...],
  "weaknesses": ["kısa madde", ...],
  "grounded_findings": [
    {"claim": "...", "cv_quote": "CV'den birebir kısa alıntı", "section_hint": "..."}
  ],
  "missing_skills": ["..."],
  "risk_flags": ["..."],
  "fairness_notes": ["Hassas özellik kullanılmadı / varsayım yapılmadı notu"],
  "summary": "2-4 cümle"
}

Kurallar:
- Her önemli güçlü/zayıf iddia için mümkünse grounded_findings'e bir kayıt ekle.
- cv_quote mutlaka verilen CV ham metninden alınmış gibi görünmeli; uydurma.
- Cinsiyet, yaş, medeni hal, din, etnik köken üzerinden değerlendirme yapma; fairness_notes'a kısaca belirt.
"""


# =============================================================================
# State
# =============================================================================


class CVAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    cv_text: str
    job_text: str
    job_parsed: str
    cv_info: str
    match_result: str
    gap_result: str
    report_markdown: str
    report_json: str
    json_repair_count: int
    critic: str
    revision_count: int
    max_revisions: int


# =============================================================================
# Dosya / metin
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


def _extract_tag_block(text: str, tag: str) -> Optional[str]:
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
    json_raw = _extract_tag_block(raw_text, "JSON")
    report_raw = _extract_tag_block(raw_text, "REPORT")
    json_text = json_raw if json_raw is not None else "{}"
    report_text = report_raw.strip() if report_raw is not None else raw_text.strip()
    try:
        parsed = json.loads(json_text)
        json_text = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        json_text = json.dumps(
            {"_error": "JSON parse edilemedi", "raw_hint": json_text[:500]},
            ensure_ascii=False,
            indent=2,
        )
    return json_text, report_text


def normalize_report_json(json_str: str) -> str:
    """Geçerli JSON string üretir (pretty)."""
    data = json.loads(json_str)
    return json.dumps(data, ensure_ascii=False, indent=2)


def validate_eval_report(json_str: str) -> tuple[Optional[EvalReportV3], Optional[str]]:
    try:
        data = json.loads(json_str)
        model = EvalReportV3.model_validate(data)
        return model, None
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, str(exc)


def get_critic_score(critic_text: str) -> Optional[int]:
    if not critic_text.strip():
        return None
    for m in re.finditer(r"(\d{1,3})\s*/\s*100", critic_text):
        return max(0, min(100, int(m.group(1))))
    section = re.search(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?Guven\s+Skoru\s*:?\s*(\d{1,3})\s*(?:/100)?",
        critic_text,
        re.IGNORECASE | re.MULTILINE,
    )
    if section:
        return max(0, min(100, int(section.group(1))))
    return None


# =============================================================================
# Graph node'ları
# =============================================================================


def node_parse_job(state: CVAgentState) -> dict:
    job = state.get("job_text", "").strip()
    if not job:
        return {
            "job_parsed": "İş ilanı boş.",
            "messages": [HumanMessage(content="parse_job: boş ilan")],
        }
    prompt = f"""Aşağıdaki iş ilanını yapılandırılmış şekilde özetle (Türkçe).

İLAN METNİ:
{job[:4000]}

Şu başlıkları kullan:
## Rol ve Seviye
## Zorunlu Gereksinimler
## Tercih Edilenler
## Red Flag / Belirsizlikler
## Anahtar Teknolojiler ve Araçlar
## Yumuşak Beceri Beklentileri

Madde işaretli liste kullan. Tahminde bulunuyorsan bunu açıkça yaz."""
    out = (LLM_MAIN.invoke(prompt).content or "").strip()
    return {
        "job_parsed": out,
        "messages": [HumanMessage(content="parse_job tamamlandı")],
    }


def node_extract_cv_info(state: CVAgentState) -> dict:
    cv_text = state.get("cv_text", "")
    if not cv_text.strip():
        return {
            "cv_info": "CV metni bulunamadı.",
            "messages": [HumanMessage(content="extract_cv_info: CV metni boş")],
        }
    prompt = f"""Aşağıdaki CV metnini analiz et ve yapılandırılmış olarak çıkar (Türkçe).

CV Metni:
{cv_text[:5000]}

Başlıklar:
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
    job_parsed = state.get("job_parsed", "")
    cv_text = state.get("cv_text", "")
    if not cv_info.strip() or not job_parsed.strip():
        return {
            "match_result": "CV bilgisi veya iş ilanı özeti eksik.",
            "messages": [HumanMessage(content="match_skills: eksik girdi")],
        }
    prompt = f"""CV profilini, yapılandırılmış iş ilanı özetiyle karşılaştır.

CV Özeti:
{cv_info[:2800]}

İş İlanı Özeti:
{job_parsed[:2800]}

Kanıt için aşağıdaki HAM CV metninden (gerekirse kısa alıntı yap) yararlan:
--- CV HAM METİN (alıntı için) ---
{cv_text[:3500]}
---

Türkçe çıktı:
## Genel Uyum Skoru
X/100 + gerekçe (zorunlu maddelere özellikle değin)

## Zorunlu Gereksinimler — Karşılandı mı?
(her madde için: Karşılandı / Kısmen / Karşılanmadı + kısa not + mümkünse CV alıntısı)

## Tercih Edilenler
## Kritik Boşluklar
## İşe Alım Riski Notları"""
    result = (LLM_MAIN.invoke(prompt).content or "").strip()
    return {
        "match_result": result,
        "messages": [HumanMessage(content="match_skills tamamlandı")],
    }


def node_gap_analysis(state: CVAgentState) -> dict:
    match_result = state.get("match_result", "")
    job_parsed = state.get("job_parsed", "")
    if not match_result.strip():
        return {
            "gap_result": "Önce beceri eşleştirmesi yapılmalı.",
            "messages": [HumanMessage(content="gap_analysis: match yok")],
        }
    prompt = f"""Eşleştirme ve iş ilanı özetine göre gelişim planı (Türkçe).

Eşleştirme:
{match_result[:2800]}

İlan Özeti:
{job_parsed[:2000]}

## Kritik Eksikler
## Öğrenme Önerileri
## Kısa Vade (0-3 ay)
## Uzun Vade (3-12 ay)
## Alternatif Pozisyon Önerileri"""
    result = (LLM_MAIN.invoke(prompt).content or "").strip()
    return {
        "gap_result": result,
        "messages": [HumanMessage(content="gap_analysis tamamlandı")],
    }


def _report_prompt_v3(
    cv_text: str,
    cv_info: str,
    match_result: str,
    gap_result: str,
    job_parsed: str,
    revision_hint: str = "",
) -> str:
    return f"""Profesyonel CV değerlendirme raporu üret (Türkçe).

HAM CV (kanıt alıntıları için — uydurma):
{cv_text[:4000]}

CV Özeti:
{cv_info[:2000]}

Eşleştirme:
{match_result[:2200]}

Gelişim Analizi:
{gap_result[:2200]}

İş İlanı Özeti:
{job_parsed[:2000]}

REVİZYON NOTU:
{revision_hint or "İlk sürüm."}

{report_schema_prompt_fragment()}

Çıktı formatı (sırayla):
1) <JSON> ... tam bir JSON nesnesi ... </JSON>
2) <REPORT> ... Markdown rapor ... </REPORT>

REPORT bölümleri:
# CV Degerlendirme Raporu (v3)
## Yonetici Ozeti
## Kanitli Bulgular Ozeti
## Aday Profili
## Pozisyona Uyum ve Alt Skorlar
## Guclu Yonler
## Gelisme Alanlari
## Ise Alim Onerisi
## Sonraki Adimlar

JSON dışında ikinci bir ham JSON bloğu yazma."""


def node_generate_report(state: CVAgentState) -> dict:
    prompt = _report_prompt_v3(
        cv_text=state.get("cv_text", ""),
        cv_info=state.get("cv_info", ""),
        match_result=state.get("match_result", ""),
        gap_result=state.get("gap_result", ""),
        job_parsed=state.get("job_parsed", ""),
    )
    raw = (LLM_MAIN.invoke(prompt).content or "").strip()
    report_json, report_markdown = split_structured_output(raw)
    return {
        "report_json": report_json,
        "report_markdown": report_markdown,
        "messages": [HumanMessage(content="generate_report tamamlandı")],
    }


def node_validate_or_repair_json(state: CVAgentState) -> dict:
    """Pydantic doğrula; gerekirse tek LLM onarımı."""
    current = state.get("report_json", "{}")
    _, err = validate_eval_report(current)
    if err is None:
        try:
            current = normalize_report_json(current)
        except Exception:
            pass
        return {
            "report_json": current,
            "messages": [HumanMessage(content="JSON şema doğrulandı")],
        }

    repair_count = state.get("json_repair_count", 0)
    if repair_count >= 1:
        return {
            "report_json": current,
            "messages": [
                HumanMessage(
                    content=f"JSON onarım atlandı (deneme sınırı): {err[:200]}"
                )
            ],
        }

    repair_prompt = f"""Aşağıdaki JSON EvalReportV3 şemasına uymuyor. SADECE geçerli bir JSON nesnesi üret.
Hata mesajı:
{err[:1500]}

Hatalı veya eksik JSON:
{current[:6000]}

Zorunlu şema:
{report_schema_prompt_fragment()}

Yanıtın TEK bir JSON nesnesi olmalı; markdown veya açıklama ekleme."""
    fixed_raw = (LLM_MAIN.invoke(repair_prompt).content or "").strip()
    fixed_raw = fixed_raw.strip()
    if fixed_raw.startswith("```"):
        fixed_raw = re.sub(r"^```(?:json)?\s*", "", fixed_raw)
        fixed_raw = re.sub(r"\s*```$", "", fixed_raw)

    model, err2 = validate_eval_report(fixed_raw)
    if model is not None:
        dumped = json.dumps(
            model.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        return {
            "report_json": dumped,
            "json_repair_count": repair_count + 1,
            "messages": [HumanMessage(content="JSON onarımı başarılı")],
        }

    return {
        "report_json": current,
        "json_repair_count": repair_count + 1,
        "messages": [
            HumanMessage(
                content=f"JSON onarım başarısız: {(err2 or '')[:200]}"
            )
        ],
    }


def node_critic_review(state: CVAgentState) -> dict:
    report_json = state.get("report_json", "{}")
    report_markdown = state.get("report_markdown", "")
    prompt = f"""Raporu kalite açısından değerlendir (Türkçe).

JSON:
{report_json[:2200]}

RAPOR:
{report_markdown[:2800]}

Kontrol listesi:
- Kanıtlar (cv_quote) gerçekten CV ile uyumlu mu (tutarlılık)?
- Şema alanları anlamlı mı?
- Halüsinasyon / abartı var mı?

Format:
## Kritik Hatalar
## Eksikler
## Guclu Yanitlar
## Guven Skoru
X/100"""
    critic = (LLM_CRITIC.invoke(prompt).content or "").strip()
    return {
        "critic": critic,
        "messages": [HumanMessage(content="critic_review tamamlandı")],
    }


def node_revise_report(state: CVAgentState) -> dict:
    hint = f"Critic geri bildirimi:\n{state.get('critic', '')[:2500]}"
    prompt = _report_prompt_v3(
        cv_text=state.get("cv_text", ""),
        cv_info=state.get("cv_info", ""),
        match_result=state.get("match_result", ""),
        gap_result=state.get("gap_result", ""),
        job_parsed=state.get("job_parsed", ""),
        revision_hint=hint,
    )
    raw = (LLM_MAIN.invoke(prompt).content or "").strip()
    report_json, report_markdown = split_structured_output(raw)
    revision_count = state.get("revision_count", 0) + 1
    out: dict[str, Any] = {
        "report_json": report_json,
        "report_markdown": report_markdown,
        "revision_count": revision_count,
        "json_repair_count": 0,
        "messages": [
            HumanMessage(content=f"revise_report tamamlandı ({revision_count})")
        ],
    }
    return out


def route_after_critic(state: CVAgentState) -> str:
    score = get_critic_score(state.get("critic", ""))
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 1)
    if score is not None and score < CRITIC_REVISE_THRESHOLD and revision_count < max_revisions:
        return "revise"
    return "finalize"


def build_cv_graph():
    graph = StateGraph(CVAgentState)
    graph.add_node("parse_job", node_parse_job)
    graph.add_node("extract_cv_info", node_extract_cv_info)
    graph.add_node("match_skills", node_match_skills)
    graph.add_node("gap_analysis", node_gap_analysis)
    graph.add_node("generate_report", node_generate_report)
    graph.add_node("validate_json", node_validate_or_repair_json)
    graph.add_node("critic_review", node_critic_review)
    graph.add_node("revise_report", node_revise_report)

    graph.set_entry_point("parse_job")
    graph.add_edge("parse_job", "extract_cv_info")
    graph.add_edge("extract_cv_info", "match_skills")
    graph.add_edge("match_skills", "gap_analysis")
    graph.add_edge("gap_analysis", "generate_report")
    graph.add_edge("generate_report", "validate_json")
    graph.add_edge("validate_json", "critic_review")
    graph.add_conditional_edges(
        "critic_review",
        route_after_critic,
        {"revise": "revise_report", "finalize": END},
    )
    graph.add_edge("revise_report", "validate_json")
    return graph.compile()


CV_AGENT = build_cv_graph()


# =============================================================================
# Tek CV çalıştırma
# =============================================================================


def run_cv_evaluation(
    cv_text: str, job_text: str
) -> tuple[str, str, str, str, str, str, str, str, str]:
    """
    Dönüş: job_parsed, cv_info, match, gap, report_md, report_json, critic, trace, status
    """
    if not cv_text.strip():
        return "", "", "", "", "", "", "", "", "⚠️ CV metni boş."
    if not job_text.strip():
        return "", "", "", "", "", "", "", "", "⚠️ İş ilanı boş."

    initial: CVAgentState = {
        "messages": [HumanMessage(content="v3 değerlendirme başlıyor.")],
        "cv_text": cv_text,
        "job_text": job_text,
        "job_parsed": "",
        "cv_info": "",
        "match_result": "",
        "gap_result": "",
        "report_markdown": "",
        "report_json": "{}",
        "json_repair_count": 0,
        "critic": "",
        "revision_count": 0,
        "max_revisions": 1,
    }
    final = dict(initial)
    trace_lines = ["### v3 Agent izi\n"]
    step = 0
    keys_track = [
        "job_parsed",
        "cv_info",
        "match_result",
        "gap_result",
        "report_markdown",
        "report_json",
        "json_repair_count",
        "critic",
        "revision_count",
    ]

    try:
        for event in CV_AGENT.stream(initial):
            step += 1
            for node_name, node_out in event.items():
                for k in keys_track:
                    if k in node_out:
                        final[k] = node_out[k]
                msgs = node_out.get("messages", [])
                if msgs:
                    trace_lines.append(f"**{step} — {node_name}** → {msgs[-1].content}")
            trace_lines.append("")
    except Exception as exc:
        trace_lines.append(f"❌ {exc}")
        return (
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "\n".join(trace_lines),
            "❌ Hata.",
        )

    report_md = final.get("report_markdown", "")
    if report_md:
        p = os.path.join(
            REPORT_DIR, f"report_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(p, "w", encoding="utf-8") as f:
            f.write(report_md)
        trace_lines.append(f"Rapor kaydedildi: `{p}`")

    model, _ = validate_eval_report(final.get("report_json", "{}"))
    score_ui = model.score if model else "—"
    cscore = get_critic_score(final.get("critic", ""))
    cpart = f"{cscore}/100" if cscore is not None else "belirlenemedi"
    rep = final.get("json_repair_count", 0)
    rev = final.get("revision_count", 0)
    status = (
        f"✅ Tamamlandı. Model skoru: {score_ui} | Critic: {cpart}\n"
        f"• JSON şema onarımı: **{rep}** — İlk üretilen metin Pydantic şemasına uymazsa "
        f"model tek seferlik düzeltme dener; **0** = ilk hali geçerliydi, **1** = bir düzeltme yapıldı.\n"
        f"• Rapor revizyonu: **{rev}** — Critic güven skoru **{CRITIC_REVISE_THRESHOLD}**’ün altındaysa "
        f"rapor yeniden yazılır (üst sınır: max_revisions); **0** = ilk rapor yeterli bulundu, **1** = bir yeniden yazım oldu."
    )

    return (
        final.get("job_parsed", ""),
        final.get("cv_info", ""),
        final.get("match_result", ""),
        final.get("gap_result", ""),
        final.get("report_markdown", ""),
        final.get("report_json", "{}"),
        final.get("critic", ""),
        "\n".join(trace_lines),
        status,
    )


def load_cv_file(file_path: Optional[str]) -> tuple[str, str]:
    if not file_path:
        return "", "Dosya seçilmedi."
    try:
        basename = os.path.basename(file_path)
        stem, ext = os.path.splitext(basename)
        dest = os.path.join(
            CV_DIR,
            f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}",
        )
        shutil.copy(file_path, dest)
        text = extract_text_from_file(dest)
        if not text or text.startswith("Dosya okunamadı"):
            return "", f"⚠️ {text}"
        return text, f"✅ {basename} kaydedildi ({len(text.split())} kelime)"
    except Exception as exc:
        return "", f"❌ {exc}"


def extract_score_from_json(json_str: str) -> int:
    model, _ = validate_eval_report(json_str)
    if model:
        return model.score
    try:
        data = json.loads(json_str)
        s = data.get("score")
        if isinstance(s, int):
            return s
    except Exception:
        pass
    return -1


def run_batch_ranking(
    job_text: str, items: list[tuple[str, str]]
) -> str:
    """items: (görünen_ad, cv_text) listesi → skora göre sıralı markdown tablo."""
    if not job_text.strip():
        return "*İş ilanı gerekli.*"
    rows: list[tuple[str, int, str, str]] = []
    for label, cv_txt in items:
        if not cv_txt.strip():
            continue
        jp, ci, m, g, md, js, _, _, st = run_cv_evaluation(cv_txt, job_text)
        sc = extract_score_from_json(js)
        decision = "?"
        try:
            data = json.loads(js)
            decision = str(data.get("decision", "?"))
        except Exception:
            pass
        rows.append((label, sc, decision, st.split("|")[0].strip() if st else ""))
    rows.sort(key=lambda x: x[1], reverse=True)
    lines = [
        "| Sıra | Aday / Dosya | Skor | Karar | Durum özeti |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, (lab, sc, dec, brief) in enumerate(rows, 1):
        lines.append(
            f"| {i} | {lab} | {sc if sc >= 0 else '—'} | {dec} | {brief[:80]} |"
        )
    return "\n".join(lines) if len(lines) > 2 else "*Değerlendirilecek CV yok.*"


# =============================================================================
# Gradio
# =============================================================================

with gr.Blocks(title="CV Agent v3", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CV Değerlendirme Agent v3")
    gr.Markdown(
        "İlan özeti + kanıtlı bulgular + Pydantic doğrulama + JSON indir + toplu sıralama."
    )

    cv_text_state = gr.State("")

    with gr.Tabs():
        with gr.Tab("Giriş"):
            with gr.Row():
                with gr.Column():
                    cv_file = gr.File(label="CV (PDF/TXT)", type="filepath")
                    cv_status = gr.Textbox(label="CV durumu", interactive=False)
                    cv_preview = gr.Textbox(
                        label="CV önizleme", lines=10, interactive=False
                    )
                with gr.Column():
                    job_input = gr.Textbox(
                        label="İş ilanı",
                        lines=14,
                        placeholder="Pozisyon, gereksinimler...",
                    )
            eval_btn = gr.Button("Değerlendirmeyi başlat", variant="primary")
            eval_status = gr.Textbox(label="Durum", interactive=False)

            gr.Markdown("### Toplu sıralama (isteğe bağlı)")
            batch_files = gr.File(
                label="Birden fazla CV seç",
                file_count="multiple",
                type="filepath",
            )
            batch_btn = gr.Button("Aynı ilana göre sırala")
            batch_out = gr.Markdown("*Toplu sonuç burada.*")

        with gr.Tab("İlan özeti"):
            job_parsed_out = gr.Markdown("*İlan ayrıştırıldıktan sonra.*")

        with gr.Tab("Analiz"):
            with gr.Row():
                cv_info_out = gr.Markdown()
                match_out = gr.Markdown()
            gap_out = gr.Markdown()

        with gr.Tab("Rapor"):
            report_out = gr.Markdown()
            with gr.Row():
                dl_md = gr.Button("Raporu indir (.md)")
                dl_json = gr.Button("JSON indir")
            dl_file = gr.File(label="İndirilen dosya", visible=False)
            json_inline = gr.Textbox(
                label="Structured JSON (aynı veri — sekme değiştirmeden görmek için)",
                lines=14,
                interactive=False,
                elem_id="v3-json-inline",
            )

        with gr.Tab("JSON önizleme"):
            gr.Markdown(
                "Aşağıdaki kutu değerlendirme bitince güncellenir. Boş görünüyorsa "
                "**Rapor** sekmesindeki JSON alanına bakın veya butona tekrar basın "
                "(bazı Gradio sürümlerinde sekme içi güncelleme gecikebilir)."
            )
            json_preview = gr.Textbox(
                label="report_json",
                lines=22,
                interactive=False,
                elem_id="v3-json-preview",
            )

        with gr.Tab("Critic"):
            critic_out = gr.Markdown()

        with gr.Tab("İz"):
            trace_out = gr.Markdown()

    def on_upload(path):
        text, status = load_cv_file(path)
        preview = text[:900] + "\n..." if len(text) > 900 else text
        return text, status, preview

    cv_file.change(
        fn=on_upload,
        inputs=[cv_file],
        outputs=[cv_text_state, cv_status, cv_preview],
    )

    def on_eval(cv_t, job_t):
        empty_json = gr.update(value="{}")
        if not cv_t.strip():
            m = "*CV yok.*"
            return (
                m,
                m,
                m,
                m,
                m,
                empty_json,
                empty_json,
                m,
                m,
                "⚠️ CV yükleyin.",
            )
        if not job_t.strip():
            m = "*İlan yok.*"
            return (
                m,
                m,
                m,
                m,
                m,
                empty_json,
                empty_json,
                m,
                m,
                "⚠️ İlan girin.",
            )
        jp, ci, mt, gp, rmd, rjs, cr, tr, st = run_cv_evaluation(cv_t, job_t)
        rjs = (rjs or "").strip() or "{}"
        json_upd = gr.update(value=rjs)
        return jp, ci, mt, gp, rmd, json_upd, json_upd, cr, tr, st

    eval_btn.click(
        fn=on_eval,
        inputs=[cv_text_state, job_input],
        outputs=[
            job_parsed_out,
            cv_info_out,
            match_out,
            gap_out,
            report_out,
            json_inline,
            json_preview,
            critic_out,
            trace_out,
            eval_status,
        ],
    )

    def on_dl_md(text):
        if not text or text.startswith("*"):
            return gr.File(visible=False)
        path = os.path.join(
            REPORT_DIR, f"rapor_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return gr.File(value=path, visible=True)

    def on_dl_json(js):
        if not js or js.strip() in ("{}", ""):
            return gr.File(visible=False)
        path = os.path.join(
            REPORT_DIR, f"rapor_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(js)
        return gr.File(value=path, visible=True)

    dl_md.click(fn=on_dl_md, inputs=[report_out], outputs=[dl_file])
    dl_json.click(
        fn=on_dl_json,
        inputs=[json_inline],
        outputs=[dl_file],
    )

    def on_batch(files, job_t):
        if not job_t.strip():
            return "*İlan gerekli.*"
        if not files:
            return "*En az bir CV seçin.*"
        items: list[tuple[str, str]] = []
        for p in files:
            if not p:
                continue
            t, _ = load_cv_file(p)
            if t:
                items.append((os.path.basename(p), t))
        return run_batch_ranking(job_t, items)

    batch_btn.click(
        fn=on_batch,
        inputs=[batch_files, job_input],
        outputs=[batch_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7864)
