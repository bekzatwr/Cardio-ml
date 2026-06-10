"""
services/summary_service.py — AI клиникалық қорытынды
CardioTracker ML v2.3

Түзетулер v2.3:
  ✅ _format_history_diff() — соңғы 2 визиттің ↑↓→ айырмашылығы
  ✅ _format_alerts()       — алерттер приоритет бойынша (🔴🟠🟡⚪)
  ✅ build_summary_prompt() — v2.0/v3.0 LLM-ке жіберілетін структуралы промпт
  ✅ generate_summary()     — v1.0 rule-based сводка (өзгерген жоқ)
  ✅ datetime.now(datetime.UTC) → datetime.now(timezone.utc) (Python 3.11+)

Edge case қорғанысы:
  ✅ visit_history None/[] → "Бірінші визит" режимі
  ✅ None мәндер барлық параметр тексерулерінде
  ✅ ТЗ: AI диагноз қоймайды, емдеу ұсынбайды
"""

from datetime import datetime, timezone
from typing import List, Optional

from schemas import (
    SummaryRequest,
    SummaryResponse,
    VisitRecord,
    PatientData,
)
from services.risk_service import classify_risk
from services.alert_service import check_alerts, AlertCheckRequest


# ═════════════════════════════════════════════════════════════════
# УТИЛИТТЕР
# ═════════════════════════════════════════════════════════════════

def safe_last(history: Optional[List[VisitRecord]]) -> Optional[VisitRecord]:
    if not history:
        return None
    return history[-1]


def fmt(val: Optional[float], unit: str = "", precision: int = 1) -> str:
    if val is None:
        return "деректер жоқ"
    return f"{round(val, precision)}{unit}"


# ═════════════════════════════════════════════════════════════════
# ДИНАМИКА БЛОГЫ (rule-based, v1.0)
# ═════════════════════════════════════════════════════════════════

HIGHER_IS_BETTER = {"ef", "six_min_walk", "hemoglobin"}
LOWER_IS_BETTER  = {"nt_probnp", "creatinine", "urea", "bilirubin"}

PARAM_LABELS = {
    "ef":           ("ФВ/ЛЖ",     "%"),
    "six_min_walk": ("6-мин тест", "м"),
    "nt_probnp":    ("NT-proBNP",  "пг/мл"),
    "creatinine":   ("Креатинин",  "мкмоль/л"),
    "hemoglobin":   ("Гемоглобин", "г/л"),
    "urea":         ("Несепнәр",   "ммоль/л"),
    "bilirubin":    ("Билирубин",  "мкмоль/л"),
    "weight_kg":    ("Салмақ",     "кг"),
}


def _build_dynamics(
    current: PatientData,
    prev: Optional[VisitRecord],
) -> dict:
    improved, worsened, stable = [], [], []
    for field, (label, unit) in PARAM_LABELS.items():
        curr_val = getattr(current, field, None)
        prev_val = getattr(prev, field, None) if prev else None
        if curr_val is None or prev_val is None:
            continue
        diff = curr_val - prev_val
        if abs(diff) < 0.01:
            stable.append(f"{label}: {fmt(curr_val, unit)}")
            continue
        entry = f"{label}: {fmt(prev_val, unit)} → {fmt(curr_val, unit)}"
        if field in HIGHER_IS_BETTER:
            (improved if diff > 0 else worsened).append(entry)
        elif field in LOWER_IS_BETTER:
            (improved if diff < 0 else worsened).append(entry)
        else:
            stable.append(entry)
    return {"improved": improved, "worsened": worsened, "stable": stable}


# ═════════════════════════════════════════════════════════════════
# ПРОМПТ УТИЛИТТЕРІ (v2.0/v3.0 LLM үшін)
# ═════════════════════════════════════════════════════════════════

def _format_history_diff(
    current: PatientData,
    history: List[VisitRecord],
) -> str:
    """
    Соңғы 2 визиттің айырмашылығын ↑↓→ белгілерімен форматтайды.
    LLM промптында АЛДЫҢҒЫ ВИЗИТПЕН САЛЫСТЫРУ бөліміне кіреді.

    Args:
      current: ағымдағы визит деректері
      history: визиттер тізімі (ескіден жаңаға)

    Returns:
      Форматталған мәтін, мысалы:
        ФВ/ЛЖ: 42.0% → 38.0% ↓ (-4.0%)
        NT-proBNP: 800 пг/мл → 1200 пг/мл ↑ (+400.0)
        Гемоглобин: → 108.0 г/л (алдыңғы деректер жоқ)

    Edge cases:
      - history бос болса → "Алдыңғы визит деректері жоқ"
      - мән болмаса → өткізіп жіберіледі
    """
    if not history:
        return "Алдыңғы визит деректері жоқ (бірінші визит)."

    prev = history[-1]
    lines = []

    for field, (label, unit) in PARAM_LABELS.items():
        curr_val = getattr(current, field, None)
        prev_val = getattr(prev, field, None)

        if curr_val is None:
            continue

        if prev_val is None:
            lines.append(f"{label}: → {fmt(curr_val, unit)} (алдыңғы деректер жоқ)")
            continue

        diff    = curr_val - prev_val
        diff_pct = round(diff / prev_val * 100, 1) if prev_val != 0 else 0

        if abs(diff) < 0.01:
            arrow = "→"
            sign  = ""
        elif field in HIGHER_IS_BETTER:
            arrow = "↑" if diff > 0 else "↓"
            sign  = "+" if diff > 0 else ""
        elif field in LOWER_IS_BETTER:
            arrow = "↓" if diff < 0 else "↑"
            sign  = "+" if diff > 0 else ""
        else:
            arrow = "→"
            sign  = "+" if diff > 0 else ""

        lines.append(
            f"{label}: {fmt(prev_val, unit)} → {fmt(curr_val, unit)} "
            f"{arrow} ({sign}{diff_pct}%)"
        )

    return "\n".join(lines) if lines else "Салыстыруға деректер жеткіліксіз."


def _format_alerts(alerts: list) -> str:
    """
    Алерттерді приоритет бойынша форматтайды.
    LLM промптында БЕЛСЕНДІ АЛЕРТТЕР бөліміне кіреді.

    Priority → emoji:
      CRITICAL → 🔴
      HIGH     → 🟠
      MEDIUM   → 🟡
      LOW      → ⚪

    Args:
      alerts: SingleAlert объекттерінің тізімі

    Returns:
      Форматталған мәтін, мысалы:
        🔴 [ALERT-06] Критикалық ЭКГ өзгерістері: ST өзгерістері.
        🟠 [TREND-02] NT-proBNP 2 визит бойы өсіп жатыр.

    Edge cases:
      - бос тізім → "Белсенді алерттер жоқ."
    """
    if not alerts:
        return "Белсенді алерттер жоқ."

    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    priority_emoji = {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MEDIUM":   "🟡",
        "LOW":      "⚪",
    }

    sorted_alerts = sorted(
        alerts,
        key=lambda a: priority_order.get(getattr(a, "priority", "LOW"), 3),
    )

    lines = []
    for alert in sorted_alerts:
        code     = getattr(alert, "alert_code", "?")
        priority = getattr(alert, "priority",   "LOW")
        message  = getattr(alert, "message",    "")
        emoji    = priority_emoji.get(priority, "⚪")
        lines.append(f"{emoji} [{code}] {message}")

    return "\n".join(lines)


def build_summary_prompt(
    visit: PatientData,
    history: List[VisitRecord],
    alerts: list,
    risk_group: str,
    risk_score: float,
) -> str:
    """
    v2.0/v3.0-да LLM-ке жіберілетін структуралы промпт жасайды.

    Ереже (ТЗ §3.4):
      - Лечение ҰСЫНБА — тек фактілерді сипатта
      - Тек ауытқуларды және трендті атап өт
      - 5-7 жолдан аспасын
      - Тілі: қазақ

    Қолданылуы (v2.0-да):
      prompt = build_summary_prompt(visit, history, alerts, risk_group, risk_score)
      response = llm_client.generate(prompt)  # OpenAI / Anthropic / etc.

    v1.0-да бұл функция қолданылмайды — generate_summary() rule-based жұмыс істейді.
    """
    history_diff = _format_history_diff(visit, history)
    alerts_text  = _format_alerts(alerts)

    prompt = f"""Сен кардиология ассистентісің. Дәрігерге қысқа клиникалық сводка жаз.

ЕРЕЖЕ (міндетті):
- Лечение ҰСЫНБА — тек фактілерді сипатта
- Тек ауытқуларды және трендті атап өт
- 5-7 жолдан аспасын
- Тілі: қазақ (немесе орыс, дәрігер нені қалайды)
- Диагноз ҚОЙМА

СОҢҒЫ ВИЗИТ ДЕРЕКТЕРІ:
ФВ/ЛЖ: {fmt(visit.ef, '%')} | NT-proBNP: {fmt(visit.nt_probnp, ' пг/мл')}
6-мин тест: {fmt(visit.six_min_walk, ' м')} | Hb: {fmt(visit.hemoglobin, ' г/л')}
Cr: {fmt(visit.creatinine, ' мкмоль/л')} | Несепнәр: {fmt(visit.urea, ' ммоль/л')}
Билирубин: {fmt(visit.bilirubin, ' мкмоль/л')} | АСТ/АЛТ: {fmt(visit.ast)}/{fmt(visit.alt)}
ЭКГ: {'ФП' if visit.ecg_af else ''} {'ST↓' if visit.ecg_st_changes else ''} {'тахи' if visit.ecg_tachycardia else ''} {'блокада' if visit.ecg_blockade else ''} (жоқ болса — норма)
Риск тобы: {risk_group} (балл: {risk_score:.2f})

АЛДЫҢҒЫ ВИЗИТПЕН САЛЫСТЫРУ:
{history_diff}

БЕЛСЕНДІ АЛЕРТТЕР ({len(alerts)} дана):
{alerts_text}

Қысқа клиникалық сводканы жаз (5-7 жол, тек фактілер):"""

    return prompt


# ═════════════════════════════════════════════════════════════════
# КЛИНИКАЛЫҚ ИНТЕРПРЕТАЦИЯ (rule-based, v1.0)
# ═════════════════════════════════════════════════════════════════

def _interpret_ef(ef: Optional[float]) -> Optional[str]:
    if ef is None: return None
    if ef < 30:  return f"ФВ/ЛЖ {ef}% — критикалық систолалық дисфункция (D тобы)."
    if ef < 40:  return f"ФВ/ЛЖ {ef}% — ауыр систолалық дисфункция (C→D тобы)."
    if ef < 50:  return f"ФВ/ЛЖ {ef}% — орташа систолалық дисфункция (C тобы)."
    if ef < 55:  return f"ФВ/ЛЖ {ef}% — шекаралық деңгей."
    return f"ФВ/ЛЖ {ef}% — норма аймағында."


def _interpret_nyha(walk: Optional[float]) -> Optional[str]:
    if walk is None: return None
    if walk < 150:  return f"6-мин тест {walk:.0f}м — ФК IV (ауыр шектеу)."
    if walk < 426:  return f"6-мин тест {walk:.0f}м — ФК III (айтарлықтай шектеу)."
    if walk <= 550: return f"6-мин тест {walk:.0f}м — ФК II (жеңіл шектеу)."
    return f"6-мин тест {walk:.0f}м — ФК I (норма)."


def _interpret_renal(
    creatinine: Optional[float],
    urea: Optional[float],
    ef: Optional[float],
) -> Optional[str]:
    if creatinine is None: return None
    ef_low = ef is not None and ef < 45
    if creatinine > 150 and ef_low:
        return f"Cr {creatinine} мкмоль/л + ФВ/ЛЖ {ef}% — кардиоренальный синдром қаупі."
    if creatinine > 150:
        return f"Cr {creatinine} мкмоль/л — бүйрек функциясы бұзылған."
    if creatinine > 110:
        note = f" Несепнәр да жоғары: {urea} ммоль/л." if urea and urea > 8.3 else ""
        return f"Cr {creatinine} мкмоль/л — шекаралық деңгей.{note}"
    return None


def _interpret_anemia(hb: Optional[float], ef: Optional[float]) -> Optional[str]:
    if hb is None or hb >= 120: return None
    ef_note = f" ФВ/ЛЖ {ef}% — жүктеме одан да артады." if ef and ef < 50 else ""
    if hb < 90:  return f"Ауыр анемия Hb {hb} г/л.{ef_note}"
    if hb < 110: return f"Анемия Hb {hb} г/л.{ef_note}"
    return f"Гемоглобин шекарада: Hb {hb} г/л.{ef_note}"


def _interpret_ecg(data: PatientData) -> Optional[str]:
    issues = []
    if data.ecg_af:          issues.append("жыбырлау аритмиясы")
    if data.ecg_tachycardia: issues.append("тахикардия")
    if data.ecg_blockade:    issues.append("блокада")
    if data.ecg_st_changes:  issues.append("ST өзгерістері")
    if issues:
        return f"ЭКГ: {', '.join(issues)}."
    return None


# ═════════════════════════════════════════════════════════════════
# МӘТІНДІК СВОДКА (rule-based, v1.0)
# ═════════════════════════════════════════════════════════════════

def _build_summary_text(
    risk_group: str,
    risk_score: float,
    risk_color: str,
    alert_count: int,
    critical_count: int,
    dynamics: dict,
    clinical_notes: List[str],
    attention_zones: List[str],
    is_first_visit: bool,
    data: PatientData,
    recommendation: str,
) -> str:
    lines = []

    lines.append("══ ЖАЛПЫ ЖАҒДАЙ ══════════════════════════════════")
    lines.append(f"Риск тобы: {risk_group} (балл: {risk_score:.2f})")
    if alert_count > 0:
        crit_note = f", оның ішінде {critical_count} CRITICAL" if critical_count > 0 else ""
        lines.append(f"Белсенді алерттер: {alert_count}{crit_note}.")
    if is_first_visit:
        lines.append("Бұл — пациенттің бірінші визиті. Динамика бағасы жоқ.")

    lines.append("")
    lines.append("══ НЕГІЗГІ КӨРСЕТКІШТЕР ══════════════════════════")
    for note in filter(None, [
        _interpret_ef(data.ef),
        _interpret_nyha(data.six_min_walk),
        _interpret_renal(data.creatinine, data.urea, data.ef),
        _interpret_anemia(data.hemoglobin, data.ef),
        _interpret_ecg(data),
    ]):
        lines.append(f"  • {note}")

    if not is_first_visit:
        lines.append("")
        lines.append("══ ДИНАМИКА (алдыңғы визитпен) ═══════════════════")
        if dynamics["improved"]:
            lines.append(f"  ✅ Жақсарды: {'; '.join(dynamics['improved'])}.")
        if dynamics["worsened"]:
            lines.append(f"  ❌ Нашарлады: {'; '.join(dynamics['worsened'])}.")
        if dynamics["stable"]:
            lines.append(f"  → Тұрақты: {'; '.join(dynamics['stable'][:3])}.")
        if not dynamics["improved"] and not dynamics["worsened"]:
            lines.append("  → Маңызды өзгерістер жоқ.")

    if clinical_notes:
        lines.append("")
        lines.append("══ КЛИНИКАЛЫҚ ЕСКЕРТУЛЕР ═════════════════════════")
        for note in clinical_notes:
            lines.append(f"  ⚠️  {note}")

    if attention_zones:
        lines.append("")
        lines.append("══ НАЗАР АУДАРЫҢЫЗ ═══════════════════════════════")
        for zone in attention_zones:
            lines.append(f"  🔸 {zone}")

    lines.append("")
    lines.append("══ ҰСЫНЫС ════════════════════════════════════════")
    lines.append(f"  {recommendation}")
    lines.append("")
    lines.append(
        "─────────────────────────────────────────────────────\n"
        "AI сводкасы тек фактілер мен ауытқуларды сипаттайды.\n"
        "Диагноз қою және емдеу тағайындау дәрігердің құзыретінде."
    )

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════
# НЕГІЗГІ ФУНКЦИЯ
# ═════════════════════════════════════════════════════════════════

def generate_summary(req: SummaryRequest) -> SummaryResponse:
    """
    Дәрігерге арналған AI-сводка генерациясы (v1.0 rule-based).

    v2.0-да: build_summary_prompt() → LLM → summary_text
    v1.0-да: rule-based сводка (_build_summary_text)

    FIX v2.3: datetime.UTC → timezone.utc (Python 3.11 compatibility)
    """
    data    = req.patient_data
    history = req.visit_history or []
    is_first_visit = len(history) == 0

    risk_result = classify_risk(data)

    alert_count    = 0
    critical_count = 0
    alert_list     = []
    try:
        alert_req  = AlertCheckRequest(current=data, visit_history=history)
        alert_list = check_alerts(alert_req)
        alert_count    = len(alert_list)
        critical_count = sum(1 for a in alert_list if a.priority == "CRITICAL")
    except Exception:
        pass

    prev     = safe_last(history)
    dynamics = _build_dynamics(data, prev)

    clinical_notes = []
    if data.ef and data.ef < 45 and data.creatinine and data.creatinine > 100:
        clinical_notes.append(
            f"Кардиоренальный синдром белгілері: ФВ {data.ef}% + Cr {data.creatinine}."
        )
    if data.ef and data.ef < 45 and data.hemoglobin and data.hemoglobin < 110:
        clinical_notes.append(
            f"Кардио-анемиялық синдром: ФВ {data.ef}% + Hb {data.hemoglobin} г/л."
        )
    if data.ecg_af and data.ef and data.ef < 50:
        clinical_notes.append(
            f"ФП + ФВ/ЛЖ {data.ef}% — жыбырлау аритмиясы жүрек шығарымын азайтады."
        )
    if data.ast and data.alt and data.alt > 0 and (data.ast / data.alt) > 2.0:
        ratio = round(data.ast / data.alt, 2)
        clinical_notes.append(f"АСТ/АЛТ={ratio} (>2.0) — іркілісті гепатопатия мүмкін.")

    attention_zones = []
    if data.ef and data.ef < 30:
        attention_zones.append(f"ФВ/ЛЖ {data.ef}% — критикалық (D тобы)")
    if data.nt_probnp and data.nt_probnp > 900:
        attention_zones.append(f"NT-proBNP {data.nt_probnp:.0f} пг/мл — жоғары риск")
    if data.hemoglobin and data.hemoglobin < 110:
        attention_zones.append(f"Анемия: Hb {data.hemoglobin} г/л")
    if data.ecg_st_changes:
        attention_zones.append("ЭКГ: ST өзгерістері — ишемия белгісі")
    if data.six_min_walk and data.six_min_walk < 150:
        attention_zones.append(f"6-мин тест {data.six_min_walk:.0f}м — ФК IV")

    # v1.0: rule-based сводка
    # v2.0-да: build_summary_prompt() → LLM
    summary_text = _build_summary_text(
        risk_group=risk_result.risk_group,
        risk_score=risk_result.risk_score,
        risk_color=risk_result.risk_color,
        alert_count=alert_count,
        critical_count=critical_count,
        dynamics=dynamics,
        clinical_notes=clinical_notes,
        attention_zones=attention_zones,
        is_first_visit=is_first_visit,
        data=data,
        recommendation=risk_result.recommendation,
    )

    return SummaryResponse(
        patient_id=data.patient_id,
        summary_text=summary_text,
        risk_group=risk_result.risk_group,
        alert_count=alert_count,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )