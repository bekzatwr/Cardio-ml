"""
services/summary_service.py — AI клиникалық қорытынды
CardioTracker ML v2.2

Edge case қорғанысы:
  ✅ visit_history None немесе [] болса → "Бірінші визит" режимі
  ✅ visit_history[-1] → safe_last() арқылы
  ✅ None мәндер барлық параметр тексерулерінде
  ✅ ТЗ талабы: AI диагноз қоймайды, емдеу ұсынбайды

Түзетулер v2.2:
  ✅ RiskPatientData alias жойылды — risk_service.py-де ол жоқ енді
     Орнына PatientData тікелей classify_risk-ке беріледі (schemas.PatientData)
  ✅ generate_summary — RiskPatientData(...) → data тікелей (PatientData)
  ✅ summary_text-те "бірінші визит" сөзі бар (test_summary_first_visit тексереді)
  ✅ summary_text-те "ДИНАМИКА" секциясы бар тарих болса (test_summary_with_history)
"""

from datetime import datetime
from typing import List, Optional

from schemas import (
    SummaryRequest,
    SummaryResponse,
    VisitRecord,
    PatientData,
)
# FIX v2.2: PatientData as RiskPatientData alias жойылды
# risk_service.py-де ендігі PatientData = schemas.PatientData (бір класс)
from services.risk_service import classify_risk
from services.alert_service import check_alerts, AlertCheckRequest


# ═════════════════════════════════════════════════════════════════
# УТИЛИТТЕР
# ═════════════════════════════════════════════════════════════════

def safe_last(history: Optional[List[VisitRecord]]) -> Optional[VisitRecord]:
    """history[-1] қауіпсіз баламасы."""
    if not history:
        return None
    return history[-1]


def fmt(val: Optional[float], unit: str = "", precision: int = 1) -> str:
    """None-safe мән форматтауы."""
    if val is None:
        return "деректер жоқ"
    return f"{round(val, precision)}{unit}"


# ═════════════════════════════════════════════════════════════════
# ДИНАМИКА БЛОГЫ
# ═════════════════════════════════════════════════════════════════

# Жоғары = жақсы параметрлер
HIGHER_IS_BETTER = {"ef", "six_min_walk", "hemoglobin"}
# Төмен = жақсы параметрлер
LOWER_IS_BETTER  = {"nt_probnp", "creatinine", "urea", "bilirubin"}

PARAM_LABELS = {
    "ef":           ("ФВ/ЛЖ",       "%"),
    "six_min_walk": ("6-мин тест",   "м"),
    "nt_probnp":    ("NT-proBNP",    "пг/мл"),
    "creatinine":   ("Креатинин",    "мкмоль/л"),
    "hemoglobin":   ("Гемоглобин",   "г/л"),
    "urea":         ("Несепнәр",     "ммоль/л"),
    "bilirubin":    ("Билирубин",    "мкмоль/л"),
    "weight_kg":    ("Салмақ",       "кг"),
}


def _build_dynamics(
    current: PatientData,
    prev: Optional[VisitRecord],
) -> dict:
    """
    Ағымдағы vs алдыңғы визит динамикасы.
    EDGE CASE: prev None болса — барлық тренд "жаңа деректер" болады.
    """
    improved, worsened, stable = [], [], []

    for field, (label, unit) in PARAM_LABELS.items():
        curr_val = getattr(current, field, None)
        prev_val = getattr(prev, field, None) if prev else None

        if curr_val is None:
            continue
        if prev_val is None:
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
# КЛИНИКАЛЫҚ ИНТЕРПРЕТАЦИЯ
# ═════════════════════════════════════════════════════════════════

def _interpret_ef(ef: Optional[float]) -> Optional[str]:
    if ef is None:
        return None
    if ef < 30:
        return f"ФВ/ЛЖ {ef}% — критикалық систолалық дисфункция (D тобы)."
    if ef < 40:
        return f"ФВ/ЛЖ {ef}% — ауыр систолалық дисфункция (C→D тобы)."
    if ef < 50:
        return f"ФВ/ЛЖ {ef}% — орташа систолалық дисфункция (C тобы)."
    if ef < 55:
        return f"ФВ/ЛЖ {ef}% — шекаралық деңгей (B тобы)."
    return f"ФВ/ЛЖ {ef}% — норма аймағында."


def _interpret_nyha(walk: Optional[float]) -> Optional[str]:
    if walk is None:
        return None
    if walk < 150:  return f"6-мин тест {walk:.0f}м — ФК IV (ауыр шектеу)."
    if walk < 426:  return f"6-мин тест {walk:.0f}м — ФК III (айтарлықтай шектеу)."
    if walk <= 550: return f"6-мин тест {walk:.0f}м — ФК II (жеңіл шектеу)."
    return f"6-мин тест {walk:.0f}м — ФК I (норма)."


def _interpret_renal(
    creatinine: Optional[float],
    urea: Optional[float],
    ef: Optional[float],
) -> Optional[str]:
    if creatinine is None:
        return None
    ef_low = ef is not None and ef < 45
    if creatinine > 150 and ef_low:
        return (
            f"Cr {creatinine} мкмоль/л + ФВ/ЛЖ {ef}% — "
            f"кардиоренальный синдром қаупі."
        )
    if creatinine > 150:
        return f"Cr {creatinine} мкмоль/л — бүйрек функциясы бұзылған."
    if creatinine > 110:
        note = f" Несепнәр да жоғары: {urea} ммоль/л." if urea and urea > 8.3 else ""
        return f"Cr {creatinine} мкмоль/л — шекаралық деңгей.{note}"
    return None


def _interpret_anemia(hb: Optional[float], ef: Optional[float]) -> Optional[str]:
    if hb is None or hb >= 120:
        return None
    ef_note = f" ФВ/ЛЖ {ef}% — жүктеме одан да артады." if ef and ef < 50 else ""
    if hb < 90:
        return f"Ауыр анемия Hb {hb} г/л.{ef_note}"
    if hb < 110:
        return f"Анемия Hb {hb} г/л.{ef_note}"
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
# МӘТІНДІК СВОДКА ГЕНЕРАЦИЯСЫ
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

    # ── 1. Жалпы жағдай ──────────────────────────────────────────
    lines.append("══ ЖАЛПЫ ЖАҒДАЙ ══════════════════════════════════")
    lines.append(f"Риск тобы: {risk_group} (балл: {risk_score:.2f})")
    if alert_count > 0:
        crit_note = f", оның ішінде {critical_count} CRITICAL" if critical_count > 0 else ""
        lines.append(f"Белсенді алерттер: {alert_count}{crit_note}.")

    # FIX: "бірінші визит" сөзі болуы тиіс (test_summary_first_visit тексереді)
    if is_first_visit:
        lines.append("Бұл — пациенттің бірінші визиті. Динамика бағасы жоқ.")

    # ── 2. Негізгі көрсеткіштер ───────────────────────────────────
    lines.append("")
    lines.append("══ НЕГІЗГІ КӨРСЕТКІШТЕР ══════════════════════════")
    notes = [
        _interpret_ef(data.ef),
        _interpret_nyha(data.six_min_walk),
        _interpret_renal(data.creatinine, data.urea, data.ef),
        _interpret_anemia(data.hemoglobin, data.ef),
        _interpret_ecg(data),
    ]
    for note in notes:
        if note:
            lines.append(f"  • {note}")

    # ── 3. ДИНАМИКА (бірінші визит емес болса) ───────────────────
    # FIX: "ДИНАМИКА" сөзі болуы тиіс (test_summary_with_history тексереді)
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

    # ── 4. Клиникалық ескертулер ──────────────────────────────────
    if clinical_notes:
        lines.append("")
        lines.append("══ КЛИНИКАЛЫҚ ЕСКЕРТУЛЕР ═════════════════════════")
        for note in clinical_notes:
            lines.append(f"  ⚠️  {note}")

    # ── 5. Назар аудару зоналары ──────────────────────────────────
    if attention_zones:
        lines.append("")
        lines.append("══ НАЗАР АУДАРЫҢЫЗ ═══════════════════════════════")
        for zone in attention_zones:
            lines.append(f"  🔸 {zone}")

    # ── 6. Ұсыныс ─────────────────────────────────────────────────
    lines.append("")
    lines.append("══ ҰСЫНЫС ════════════════════════════════════════")
    lines.append(f"  {recommendation}")

    # ── 7. Disclaimer (ТЗ талабы) ────────────────────────────────
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
    Дәрігерге арналған AI-сводка генерациясы.

    Edge case-тар:
      - visit_history None/[] → is_first_visit=True, динамика бөлімі өткізіледі
      - visit_history[-1] → safe_last() арқылы (IndexError жоқ)
      - alert_service exception шығарса → alert_count=0 қайтарылады

    FIX v2.2:
      - RiskPatientData(...) → classify_risk(data) тікелей
        Себебі: risk_service.py-де ендігі schemas.PatientData = PatientData
        Сондықтан req.patient_data тікелей classify_risk-ке беріледі.
    """
    data    = req.patient_data
    history = req.visit_history or []   # ← None-safe

    is_first_visit = len(history) == 0

    # ── Риск тобы ────────────────────────────────────────────────
    # FIX v2.2: RiskPatientData(...) конвертациясы жойылды
    # PatientData (schemas) = classify_risk күтетін тип
    risk_result = classify_risk(data)

    # ── Алерттер ────────────────────────────────────────────────
    alert_count    = 0
    critical_count = 0
    try:
        alert_req  = AlertCheckRequest(current=data, visit_history=history)
        alert_list = check_alerts(alert_req)
        alert_count    = len(alert_list)
        critical_count = sum(1 for a in alert_list if a.priority == "CRITICAL")
    except Exception:
        pass   # ← alert_service сынса — сводка жалғасады

    # ── Динамика ─────────────────────────────────────────────────
    prev     = safe_last(history)
    dynamics = _build_dynamics(data, prev)

    # ── Клиникалық ескертулер ────────────────────────────────────
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

    # ── Назар аудару зоналары ────────────────────────────────────
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

    # ── Мәтіндік сводка ──────────────────────────────────────────
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
        generated_at=datetime.utcnow().isoformat(),
    )