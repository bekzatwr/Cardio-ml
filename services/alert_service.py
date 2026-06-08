"""
services/alert_service.py — Клиникалық алерттердің бизнес-логикасы
CardioTracker ML v2.2

Түзетулер v2.2:
  ✅ ALERT-03 priority градациясы:
       Hb < 80  → CRITICAL  (бұрын < 90 болатын)
       80 ≤ Hb < 100 → HIGH
       100 ≤ Hb < 110 → MEDIUM
  ✅ ALERT-04 абсолютты шекара (>150 тарихсыз) ЖОЙЫЛДЫ:
       creatinine=300, тарих жоқ → ALERT-04 жоқ (ТЗ: тек >30% динамика)
  ✅ ALERT-05 priority градациясы:
       NT > 5000        → CRITICAL
       1800 < NT ≤ 5000 → HIGH
       900 < NT ≤ 1800  → MEDIUM
"""

from typing import List, Optional
from schemas import (
    AlertCheckRequest,
    AlertResponse,
    SingleAlert,
    NtProBnpParadoxRequest,
    NtProBnpParadoxResponse,
    VisitRecord,
    PatientData,
)


# ═════════════════════════════════════════════════════════════════
# УТИЛИТТЕР
# ═════════════════════════════════════════════════════════════════

def safe_last(history):
    if not history:
        return None
    return history[-1]

def safe_first(history):
    if not history:
        return None
    return history[0]

def pct_change(current, previous):
    if previous == 0:
        return None
    return ((current - previous) / previous) * 100

def has_min_history(history, min_count=1):
    return bool(history) and len(history) >= min_count


# ═════════════════════════════════════════════════════════════════
# БЛОК 1 — БАЗАЛЫҚ АЛЕРТТЕР
# ═════════════════════════════════════════════════════════════════

def _alert_01_critical_ef(data):
    if data.ef is None:
        return None
    if data.ef < 30:
        return SingleAlert(
            alert_code="ALERT-01",
            priority="CRITICAL",
            message=f"D тобы: Критикалық төмен фракция выброс — ФВ/ЛЖ {data.ef}%. Жедел кардиологиялық бағалау қажет.",
            parameter="ФВ/ЛЖ",
            value=data.ef,
            threshold=30.0,
        )
    return None


def _alert_02_nt_rising(data, history):
    if data.nt_probnp is None:
        return None
    prev = safe_last(history)
    if prev is None or prev.nt_probnp is None:
        return None
    change = pct_change(data.nt_probnp, prev.nt_probnp)
    if change is None or change <= 10:
        return None
    ef_improved   = (data.ef is not None and prev.ef is not None and data.ef > prev.ef + 2)
    walk_improved = (data.six_min_walk is not None and prev.six_min_walk is not None
                     and data.six_min_walk > prev.six_min_walk + 30)
    if ef_improved or walk_improved:
        detail = []
        if ef_improved:
            detail.append(f"ФВ/ЛЖ жақсарды: {prev.ef}% → {data.ef}%")
        if walk_improved:
            detail.append(f"6-мин тест жақсарды: {prev.six_min_walk}м → {data.six_min_walk}м")
        return SingleAlert(
            alert_code="ALERT-02",
            priority="CRITICAL",
            message=(
                f"NT-proBNP парадоксы: NT {round(change,1)}% өсті "
                f"({prev.nt_probnp:.0f} → {data.nt_probnp:.0f} пг/мл), "
                f"бірақ {'; '.join(detail)}. Терапияны қайта қарау ұсынылады."
            ),
            parameter="NT-proBNP",
            value=data.nt_probnp,
            threshold=prev.nt_probnp,
        )
    return None


def _alert_03_anemia(data):
    """
    FIX v2.2:
      Hb < 80   → CRITICAL
      80 ≤ < 100 → HIGH
      100 ≤ < 110 → MEDIUM
    """
    if data.hemoglobin is None:
        return None
    if data.hemoglobin >= 110:
        return None
    if data.hemoglobin < 80:
        priority = "CRITICAL"
        grade    = "өте ауыр дәреже"
    elif data.hemoglobin < 100:
        priority = "HIGH"
        grade    = "ауыр дәреже"
    else:
        priority = "MEDIUM"
        grade    = "орташа дәреже"
    return SingleAlert(
        alert_code="ALERT-03",
        priority=priority,
        message=f"Анемия ({grade}): Hb {data.hemoglobin} г/л (<110). Жүрекке қосымша жүктеме.",
        parameter="Гемоглобин",
        value=data.hemoglobin,
        threshold=110.0,
    )


def _alert_04_creatinine(data, history):
    """
    FIX v2.2: абсолютты шекара (>150) жойылды.
    Тек >30% динамика тексеріледі, тарих міндетті.
    creatinine=300 тарихсыз → алерт жоқ.
    """
    if data.creatinine is None:
        return None
    prev = safe_last(history)
    if prev is None or prev.creatinine is None:
        return None
    change = pct_change(data.creatinine, prev.creatinine)
    if change is None:
        return None
    if change >= 30:
        priority = "CRITICAL" if change > 50 else "HIGH"
        return SingleAlert(
            alert_code="ALERT-04",
            priority=priority,
            message=(
                f"Креатинин {round(change,1)}% өсті "
                f"({prev.creatinine} → {data.creatinine} мкмоль/л). "
                f"Нефротоксикалық препараттарды тексеру ұсынылады."
            ),
            parameter="Креатинин",
            value=data.creatinine,
            threshold=prev.creatinine * 1.30,
        )
    return None


def _alert_05_high_nt(data):
    """
    FIX v2.2:
      NT > 5000        → CRITICAL
      1800 < NT ≤ 5000 → HIGH
      900 < NT ≤ 1800  → MEDIUM
    """
    if data.nt_probnp is None:
        return None
    if data.nt_probnp <= 900:
        return None
    if data.nt_probnp > 5000:
        priority = "CRITICAL"
        level    = "критикалық риск"
    elif data.nt_probnp > 1800:
        priority = "HIGH"
        level    = "өте жоғары риск"
    else:
        priority = "MEDIUM"
        level    = "жоғары риск"
    return SingleAlert(
        alert_code="ALERT-05",
        priority=priority,
        message=f"NT-proBNP {level} аймағында: {data.nt_probnp:.0f} пг/мл (норма <125).",
        parameter="NT-proBNP",
        value=data.nt_probnp,
        threshold=900.0,
    )


def _alert_06_ecg(data):
    issues = []
    if data.ecg_af:           issues.append("Жыбырлау аритмиясы (ФП)")
    if data.ecg_tachycardia:  issues.append("Тахикардия")
    if data.ecg_blockade:     issues.append("Блокада")
    if data.ecg_st_changes:   issues.append("ST өзгерістері")
    if not issues:
        return None
    is_critical = data.ecg_af or data.ecg_st_changes
    return SingleAlert(
        alert_code="ALERT-06",
        priority="CRITICAL" if is_critical else "HIGH",
        message=f"Критикалық ЭКГ өзгерістері: {', '.join(issues)}.",
        parameter="ЭКГ",
        value=None,
        threshold=None,
    )


# ═════════════════════════════════════════════════════════════════
# БЛОК 2 — КОМБИНАЦИЯЛАНҒАН АЛЕРТТЕР
# ═════════════════════════════════════════════════════════════════

def _combo_01_cardiorenal(data, history):
    if data.ef is None or data.creatinine is None:
        return None
    if data.ef >= 40:
        return None
    if data.creatinine > 150:
        return SingleAlert(
            alert_code="COMBO-01",
            priority="CRITICAL",
            message=(
                f"Кардиоренальный синдром: ФВ/ЛЖ {data.ef}% + "
                f"Креатинин {data.creatinine} мкмоль/л. Нефролог консультациясы ұсынылады."
            ),
            parameter="ФВ/ЛЖ + Креатинин",
            value=data.creatinine,
            threshold=150.0,
        )
    prev = safe_last(history)
    if prev is None or prev.creatinine is None:
        return None
    change = pct_change(data.creatinine, prev.creatinine)
    if change is not None and change > 15 and data.creatinine > 100:
        return SingleAlert(
            alert_code="COMBO-01",
            priority="HIGH",
            message=(
                f"Кардиоренальный синдром қаупі: ФВ/ЛЖ {data.ef}% + "
                f"Креатинин {round(change,1)}% өсіп жатыр "
                f"({prev.creatinine} → {data.creatinine}). Диуретик дозасын қадағалаңыз."
            ),
            parameter="ФВ/ЛЖ + Креатинин",
            value=data.creatinine,
            threshold=prev.creatinine * 1.15,
        )
    return None


def _combo_05_triple_risk(data):
    if data.ef is None or data.creatinine is None or data.hemoglobin is None:
        return None
    if data.ef < 40 and data.creatinine > 150 and data.hemoglobin < 110:
        return SingleAlert(
            alert_code="COMBO-05",
            priority="CRITICAL",
            message=(
                f"ҮШТІК ҚАУІП: ФВ/ЛЖ {data.ef}% + "
                f"Cr {data.creatinine} мкмоль/л + Hb {data.hemoglobin} г/л. "
                f"Жедел консилиум ұсынылады."
            ),
            parameter="ФВ/ЛЖ + Cr + Hb",
            value=None,
            threshold=None,
        )
    return None


# ═════════════════════════════════════════════════════════════════
# БЛОК 3 — ТРЕНД АЛЕРТТЕРІ
# ═════════════════════════════════════════════════════════════════

def _trend_01_ef_decline(data, history):
    if data.ef is None or not has_min_history(history, 2):
        return None
    ef_series = [v.ef for v in history if v.ef is not None]
    ef_series.append(data.ef)
    if len(ef_series) < 2:
        return None
    window = ef_series[-3:] if len(ef_series) >= 3 else ef_series
    if all(window[i] > window[i+1] for i in range(len(window)-1)):
        total_drop = round(window[0] - window[-1], 1)
        if total_drop >= 3:
            return SingleAlert(
                alert_code="TREND-01",
                priority="HIGH",
                message=(
                    f"ФВ/ЛЖ үздіксіз төмендеуде: {' → '.join(str(v) for v in window)}%. "
                    f"Жалпы төмендеу: {total_drop}%. Терапия тиімсіздігі немесе прогрессия."
                ),
                parameter="ФВ/ЛЖ тренд",
                value=data.ef,
                threshold=window[0],
            )
    return None


def _trend_02_nt_escalation(data, history):
    if data.nt_probnp is None or not has_min_history(history, 1):
        return None
    nt_series = [v.nt_probnp for v in history if v.nt_probnp is not None]
    nt_series.append(data.nt_probnp)
    if len(nt_series) < 2:
        return None
    window = nt_series[-3:] if len(nt_series) >= 3 else nt_series
    if all(window[i] < window[i+1] for i in range(len(window)-1)):
        change = pct_change(window[-1], window[0])
        if change is not None and change > 20:
            return SingleAlert(
                alert_code="TREND-02",
                priority="HIGH",
                message=(
                    f"NT-proBNP {len(nt_series)} визит бойы өсіп жатыр: "
                    f"{' → '.join(f'{v:.0f}' for v in window)} пг/мл "
                    f"(+{round(change,1)}%). Терапия жеткіліксіз."
                ),
                parameter="NT-proBNP тренд",
                value=data.nt_probnp,
                threshold=window[0],
            )
    return None


def _trend_03_walk_decline(data, history):
    if data.six_min_walk is None or not has_min_history(history, 1):
        return None
    walk_series = [v.six_min_walk for v in history if v.six_min_walk is not None]
    walk_series.append(data.six_min_walk)
    if len(walk_series) < 2:
        return None
    window = walk_series[-3:] if len(walk_series) >= 3 else walk_series
    if all(window[i] > window[i+1] for i in range(len(window)-1)):
        total_drop = round(window[0] - window[-1])
        if total_drop >= 60:
            return SingleAlert(
                alert_code="TREND-03",
                priority="MEDIUM",
                message=(
                    f"6-мин тест үздіксіз азаюда: {' → '.join(str(v)+'м' for v in window)}. "
                    f"Барлық төмендеу: {total_drop}м."
                ),
                parameter="6-мин тест тренд",
                value=data.six_min_walk,
                threshold=window[0],
            )
    return None


def _trend_04_weight_change(data, history):
    if data.weight_kg is None or not has_min_history(history, 1):
        return None
    prev = safe_last(history)
    if prev is None or prev.weight_kg is None:
        return None
    diff = data.weight_kg - prev.weight_kg
    if diff >= 2.0:
        return SingleAlert(
            alert_code="TREND-04",
            priority="MEDIUM",
            message=(
                f"Салмақ өсті: {prev.weight_kg} → {data.weight_kg} кг "
                f"(+{round(diff,1)} кг). Сұйықтық іркілуі мүмкін."
            ),
            parameter="Салмақ",
            value=data.weight_kg,
            threshold=prev.weight_kg + 2.0,
        )
    return None


# ═════════════════════════════════════════════════════════════════
# НЕГІЗГІ ФУНКЦИЯ
# ═════════════════════════════════════════════════════════════════

def check_alerts(req: AlertCheckRequest) -> List[SingleAlert]:
    data    = req.current
    history = req.visit_history or []

    alerts: List[SingleAlert] = []

    for fn in [
        lambda: _alert_01_critical_ef(data),
        lambda: _alert_02_nt_rising(data, history),
        lambda: _alert_03_anemia(data),
        lambda: _alert_04_creatinine(data, history),
        lambda: _alert_05_high_nt(data),
        lambda: _alert_06_ecg(data),
    ]:
        r = fn()
        if r: alerts.append(r)

    for fn in [
        lambda: _combo_01_cardiorenal(data, history),
        lambda: _combo_05_triple_risk(data),
    ]:
        r = fn()
        if r: alerts.append(r)

    if has_min_history(history, 1):
        for fn in [
            lambda: _trend_01_ef_decline(data, history),
            lambda: _trend_02_nt_escalation(data, history),
            lambda: _trend_03_walk_decline(data, history),
            lambda: _trend_04_weight_change(data, history),
        ]:
            r = fn()
            if r: alerts.append(r)

    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    alerts.sort(key=lambda a: priority_order.get(a.priority, 99))
    return alerts


# ═════════════════════════════════════════════════════════════════
# NT-proBNP ПАРАДОКСЫ
# ═════════════════════════════════════════════════════════════════

def check_nt_paradox(req: NtProBnpParadoxRequest) -> NtProBnpParadoxResponse:
    nt  = req.nt_probnp
    ef  = req.ef
    sym = req.has_symptoms

    if nt is None or ef is None:
        return NtProBnpParadoxResponse(
            patient_id=req.patient_id,
            paradox_detected=False,
            message="NT-proBNP немесе ФВ/ЛЖ деректері жоқ — парадокс тексерілмеді.",
            recommendation="NT-proBNP және ФВ/ЛЖ мәндерін енгізіңіз.",
            nt_probnp_value=nt,
            ef_value=ef,
        )

    if nt < 125 and ef < 35 and sym:
        return NtProBnpParadoxResponse(
            patient_id=req.patient_id,
            paradox_detected=True,
            message=(
                f"NT-proBNP парадоксы анықталды: NT {nt:.0f} пг/мл (<125, норма), "
                f"бірақ ФВ/ЛЖ {ef}% (<35%) және клиникалық симптомдар бар."
            ),
            recommendation=(
                "Семіздікте NT-proBNP дилюциясы мүмкін. "
                "Альтернативті маркерлерді (BNP, ST2) тексеру ұсынылады."
            ),
            nt_probnp_value=nt,
            ef_value=ef,
        )

    return NtProBnpParadoxResponse(
        patient_id=req.patient_id,
        paradox_detected=False,
        message=f"NT-proBNP парадоксы жоқ. NT={nt:.0f} пг/мл, ФВ/ЛЖ={ef}%.",
        recommendation="Стандартты мониторинг жалғастыру.",
        nt_probnp_value=nt,
        ef_value=ef,
    )