"""
services/risk_service.py — Риск классификациясының бизнес-логикасы
CardioTracker ML v2.2

Датасет негізі: ХСН_ОНМТ (135 пациент, Тараз қалалық кардиохирургиялық орталығы)
Датасет статистикасы:
  - EF диапазоны: 40–68%, орта: 51.4%
  - NYHA: ФК I=3, ФК II=47, ФК III=70, ФК IV=2
  - Коморбидность: Гипертония=98, ИМ анамнезі=43, Диабет=14, ФП=18, ХОБЛ=3
  - NT-proBNP: датасетте жоқ (v1.0-де дәрігер қолмен енгізеді)

ACC/AHA стадия классификациясы:
  A  — Қауіп факторлары бар, бірақ жүрек зақымы жоқ (green)
  B  — Жүрек зақымы бар, бірақ симптомы жоқ (blue)
  C  — Жүрек зақымы + симптомдар бар (yellow)
  C→D — Ауыр симптомдар, терапияға төзімді (orange)
  D  — Рефрактерлі ХСН (red)

Түзетулер v2.2:
  ✅ PatientData / RiskResponse — schemas.py-ден импорт (дублирование жойылды)
  ✅ patient_id: Optional[str] — schemas.py-мен сәйкес (бұрын int болатын)
  ✅ _score_walk — ТЗ шекараларына сәйкестендірілді:
       <150 → ФК IV (+0.20)
       150–425 → ФК III (+0.12)   # бұрын <300 болатын
       426–550 → ФК II (+0.05)    # бұрын <426 болатын
       >550 → ФК I (+0.00)
  ✅ _score_nt_probnp — NT >= 125 бақылау зонасы (бұрын > 125, яғни 125 нормаль болатын)
  ✅ _score_comorbidities — prior_mi warnings қосылды (test_prior_mi_adds_score)
  ✅ _score_comorbidities — коморбидтілік факторы қазақшаға сәйкес (test_multiple_comorbidities)
  ✅ classify_risk — ckd + creatinine >= 150 → кардиоренальный warning (test_ckd_plus_high_creatinine_warning)
  ✅ classify_risk — ef None → ФВ warning (test_ef_none_generates_warning)
  ✅ classify_risk — nt_probnp None + ef жоқ → NT-proBNP warning (test_nt_probnp_none_warning)
  ✅ classify_risk — ef < 35 + nt_probnp < 125 → парадокс warning (test_nt_probnp_paradox_warning)
"""

import re
from typing import Optional, List, Tuple

# ═══════════════════════════════════════════════════════════════════
# ИМПОРТ — schemas.py-ден бір рет (дублирование жойылды)
# ═══════════════════════════════════════════════════════════════════

from schemas import PatientData, RiskResponse


# ═════════════════════════════════════════════════════════════════
# ДАТАСЕТ ПАРСЕРІ (Excel жолынан PatientData жасайды)
# ═════════════════════════════════════════════════════════════════

def _safe_float(val) -> Optional[float]:
    """None-safe float парсинг"""
    if val is None:
        return None
    try:
        import pandas as pd
        if pd.isna(val):
            return None
    except Exception:
        pass
    try:
        return float(str(val).replace(',', '.').replace('%', '').replace(';', '').strip())
    except (ValueError, AttributeError):
        return None


def parse_ef_from_dataset(val) -> Optional[float]:
    """
    "ФВ/ЛЖ" бағанынан EF парсинг.
    Датасетте: 0.42 немесе "46%;" форматтары бар.
    """
    v = _safe_float(val)
    if v is None:
        return None
    if v <= 1.0:      # 0.42 → 42.0
        v = round(v * 100, 1)
    return v


def parse_6min_from_dataset(text) -> Optional[float]:
    """
    "Тест 6 minute" бағанынан метр немесе NYHA класын парсинг.
    Датасет мысалдары: "фкIII", "фк lll", "фк II", "300м", "фк IV"
    NYHA орта мәндері (ТЗ шкаласы бойынша):
      ФК I  → 570м  (>550)
      ФК II → 460м  (426–550)
      ФК III→ 300м  (150–425)
      ФК IV → 100м  (<150)
    """
    if text is None:
        return None
    try:
        import pandas as pd
        if pd.isna(text):
            return None
    except Exception:
        pass

    s = str(text).lower().strip()

    # Нақты метр болса
    m = re.search(r'(\d{2,4})\s*м', s)
    if m:
        return float(m.group(1))

    # ФК анықтау
    if re.search(r'фк\s*(iv|4)\b', s):
        return 100.0
    if re.search(r'фк\s*(iii|3|lll)\b', s):
        return 300.0
    if re.search(r'фк\s*(ii|2|ll)\b', s):
        return 460.0
    if re.search(r'фк\s*(i|1)\b', s):
        return 570.0

    return None


def parse_nyha_from_dataset(row: dict) -> Optional[int]:
    """
    NYHA класын үш бағаннан іздейді:
    "Тест 6 minute", "основной диагноз ", "Сопутствующий диагноз "
    Датасет статистикасы: ФК I=3, ФК II=47, ФК III=70, ФК IV=2
    """
    sources = [
        str(row.get('Тест 6 minute', '') or ''),
        str(row.get('основной диагноз ', '') or ''),
        str(row.get('Сопутствующий диагноз ', '') or ''),
    ]
    text = ' '.join(sources).lower()

    if re.search(r'фк\s*(iv|4)\b', text):
        return 4
    if re.search(r'фк\s*(iii|3|lll)\b', text):
        return 3
    if re.search(r'фк\s*(ii|2|ll)\b', text):
        return 2
    if re.search(r'фк\s*(i|1)\b', text):
        return 1
    return None


def parse_biochemistry_from_dataset(text) -> dict:
    """
    "Лаб.Биохимия анализ." мәтінінен биохимия парсинг.
    Мысал: "Общий белок г/л68,Мочевина ммоль/л5,81,Креатинин ммоль/л69,Билирубин общий ммоль/л 10,"
    """
    result = {}
    if not text or str(text).strip() in ('', 'nan', 'NaN'):
        return result

    text = str(text)
    patterns = {
        'protein':    r'белок[^,\d]*(\d+[,.]?\d*)',
        'urea':       r'Мочевина[^,\d]*(\d+[,.]?\d*)',
        'creatinine': r'Креатинин[^,\d]*(\d+[,.]?\d*)',
        'bilirubin':  r'Билирубин[^,\d]*(\d+[,.]?\d*)',
        'alt':        r'АЛТ[^,\d]*(\d+[,.]?\d*)',
        'ast':        r'АСТ[^,\d]*(\d+[,.]?\d*)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                result[key] = float(m.group(1).replace(',', '.'))
            except ValueError:
                pass
    return result


def parse_oac_from_dataset(text) -> dict:
    """
    "ОАК" мәтінінен гемоглобин парсинг.
    Мысал: "лейкоциты 5,0,эритроциты 3,4,Гемоглобин 105"
    """
    result = {}
    if not text or str(text).strip() in ('', 'nan', 'NaN'):
        return result

    m = re.search(r'Гемоглобин\s*(\d+)', str(text), re.IGNORECASE)
    if m:
        result['hemoglobin'] = float(m.group(1))
    return result


def parse_ecg_from_dataset(text) -> dict:
    """
    "ЭКГ" мәтінінен флагтар парсинг.
    Датасет мысалдары: "Фибрилляция предсердий", "ЧСС 72", "депрессий сегмента V1-V5"
    """
    if not text or str(text).strip() in ('', 'nan', 'NaN'):
        return {}
    text = str(text).lower()
    return {
        'af':          bool(re.search(r'фибрилл|мерцательн', text)),
        'st_changes':  bool(re.search(r'\bst\b|депрессия|элевация|субэндокард', text)),
        'blockade':    bool(re.search(r'блокад', text)),
        'tachycardia': bool(re.search(r'тахикард', text)),
    }


def parse_comorbidities_from_dataset(row: dict) -> dict:
    """
    "Сопутствующий диагноз" + "основной диагноз" мәтінінен коморбидность парсинг.
    Датасет статистикасы: Гипертония=98, ИМ=43, Диабет=14, ХОБЛ=3, ФП=18
    """
    combined = (
        str(row.get('Сопутствующий диагноз ', '') or '') + ' ' +
        str(row.get('основной диагноз ', '') or '')
    ).lower()

    return {
        'diabetes':     bool(re.search(r'диабет|сд\s*\d', combined)),
        'hypertension': bool(re.search(r'гипертон|гипертенз', combined)),
        'prior_mi':     bool(re.search(r'инфаркт|пикс', combined)),
        'copd':         bool(re.search(r'хобл|бронхит', combined)),
        'afib':         bool(re.search(r'фибрилл|мерцательн', combined)),
        'ckd':          bool(re.search(r'хбп|хроническ.*почеч|нефропат', combined)),
    }


def patient_from_dataset_row(row: dict, patient_id=None) -> PatientData:
    """
    Excel датасетінің бір жолынан PatientData объектісін жасайды.
    Барлық бағандарды нақты атауларымен пайдаланады.

    Қолданылатын бағандар:
      'ФВ/ЛЖ', 'Тест 6 minute', 'Рост', 'Вес ',
      'Лаб.Биохимия анализ.', 'ОАК', 'ЭКГ',
      'основной диагноз ', 'Сопутствующий диагноз '

    patient_id — str немесе int болуы мүмкін, PatientData str қабылдайды.
    """
    biochem   = parse_biochemistry_from_dataset(row.get('Лаб.Биохимия анализ.'))
    oac       = parse_oac_from_dataset(row.get('ОАК'))
    ecg       = parse_ecg_from_dataset(row.get('ЭКГ'))
    comorbid  = parse_comorbidities_from_dataset(row)
    nyha      = parse_nyha_from_dataset(row)

    ef_raw    = parse_ef_from_dataset(row.get('ФВ/ЛЖ'))
    walk_raw  = parse_6min_from_dataset(row.get('Тест 6 minute'))

    height = _safe_float(row.get('Рост'))
    weight = _safe_float(row.get('Вес '))
    # Датасетте Вес кейде "54.0" немесе "54" форматта
    if height and height < 50:   height = None   # қате мән қорғанысы
    if weight and weight > 300:  weight = None

    # patient_id → str (schemas.py PatientData.patient_id: Optional[str])
    pid = str(patient_id) if patient_id is not None else None

    return PatientData(
        patient_id=pid,

        ef=ef_raw,
        six_min_walk=walk_raw,
        nt_probnp=None,   # датасетте жоқ

        height_cm=height,
        weight_kg=weight,

        creatinine=biochem.get('creatinine'),
        urea=biochem.get('urea'),
        bilirubin=biochem.get('bilirubin'),
        ast=biochem.get('ast'),
        alt=biochem.get('alt'),
        hemoglobin=oac.get('hemoglobin'),

        ecg_af=ecg.get('af', False),
        ecg_st_changes=ecg.get('st_changes', False),
        ecg_blockade=ecg.get('blockade', False),
        ecg_tachycardia=ecg.get('tachycardia', False),

        symptom_class=nyha,
        has_symptoms=nyha is not None and nyha >= 2,

        has_diabetes=comorbid.get('diabetes', False),
        has_hypertension=comorbid.get('hypertension', False),
        has_prior_mi=comorbid.get('prior_mi', False),
        has_copd=comorbid.get('copd', False),
        has_afib_history=comorbid.get('afib', False),
        has_ckd=comorbid.get('ckd', False),
    )


# ═════════════════════════════════════════════════════════════════
# КОНСТАНТАЛАР
# ═════════════════════════════════════════════════════════════════

STAGE_A  = "Норма (A)"
STAGE_B  = "Стадия B"
STAGE_C  = "Риск C"
STAGE_CD = "C→D"
STAGE_D  = "Стадия D"

RISK_COLOR_MAP = {
    STAGE_A:  "green",
    STAGE_B:  "blue",
    STAGE_C:  "yellow",
    STAGE_CD: "orange",
    STAGE_D:  "red",
}

RECOMMENDATION_MAP = {
    STAGE_A:  "Жылдық жоспарлы бақылау. Қауіп факторларын бақылау.",
    STAGE_B:  "6 айда бір бақылау. Терапияны түзету.",
    STAGE_C:  "3 айда бір бақылау. Дәрілік терапияны оңтайландыру.",
    STAGE_CD: "2 аптада бір бақылау. Госпитализацияны қарастыру.",
    STAGE_D:  "Жедел госпитализация немесе паллиативтік көмек.",
}

BMI_UNDERWEIGHT = 18.5
BMI_OVERWEIGHT  = 25.0
BMI_OBESE       = 30.0
BMI_OBESE2      = 35.0

CONFIDENCE_HIGH   = "высокая"
CONFIDENCE_MEDIUM = "средняя"
CONFIDENCE_LOW    = "низкая"
CONFIDENCE_NONE   = "недостаточно данных"


# ═════════════════════════════════════════════════════════════════
# УТИЛИТТЕР
# ═════════════════════════════════════════════════════════════════

def _calc_bmi(h: Optional[float], w: Optional[float]) -> Optional[float]:
    if h and w and h > 0:
        return round(w / (h / 100) ** 2, 1)
    return None


def _count_filled(data: PatientData) -> Tuple[int, int]:
    """Қанша өріс толтырылған — confidence есептеу үшін"""
    fields = [
        data.ef, data.nt_probnp, data.six_min_walk,
        data.creatinine, data.hemoglobin, data.urea,
        data.bilirubin, data.height_cm, data.weight_kg,
        data.symptom_class,
    ]
    filled = sum(1 for f in fields if f is not None)
    return filled, len(fields)


def _get_confidence(filled: int, total: int) -> Tuple[str, int]:
    pct = round(filled / total * 100) if total > 0 else 0
    if pct >= 70: return CONFIDENCE_HIGH,   pct
    if pct >= 40: return CONFIDENCE_MEDIUM, pct
    if pct >= 20: return CONFIDENCE_LOW,    pct
    return CONFIDENCE_NONE, pct


def _score_to_stage(score: float) -> str:
    if score < 0.15: return STAGE_A
    if score < 0.35: return STAGE_B
    if score < 0.55: return STAGE_C
    if score < 0.75: return STAGE_CD
    return STAGE_D


# ═════════════════════════════════════════════════════════════════
# БАЛЛ ЕСЕПТЕУ ФУНКЦИЯЛАРЫ
# Әрбір функция factors + breakdown тізімдерін толтырады
# ═════════════════════════════════════════════════════════════════

def _score_ef(
    ef: Optional[float],
    factors: List[str],
    breakdown: List[str],
    warnings: List[str],
) -> float:
    """
    EF бойынша балл (0–0.40).
    Датасет EF диапазоны: 40–68%, орта: 51.4%
    Барлық датасет пациенттері EF≥40 — бірақ болашақта төмен EF пациенттер болады.

    FIX v2.2: warnings параметрі қосылды —
      ef None → "ФВ/ЛЖ берілмеді" warning (test_ef_none_generates_warning)
    """
    if ef is None:
        breakdown.append("ФВ/ЛЖ: деректер жоқ (+0.00)")
        warnings.append("ФВ/ЛЖ берілмеді — негізгі параметр. Толтыру ұсынылады.")
        return 0.0

    if ef < 25:
        factors.append(f"ФВ/ЛЖ критикалық деңгейде: {ef}%")
        breakdown.append(f"ФВ/ЛЖ {ef}% (<25): +0.40")
        return 0.40
    if ef < 35:
        factors.append(f"ФВ/ЛЖ ауыр дисфункция: {ef}%")
        breakdown.append(f"ФВ/ЛЖ {ef}% (<35): +0.30")
        return 0.30
    if ef < 45:
        factors.append(f"ФВ/ЛЖ орташа төмен: {ef}%")
        breakdown.append(f"ФВ/ЛЖ {ef}% (<45): +0.18")
        return 0.18
    if ef < 55:
        factors.append(f"ФВ/ЛЖ шекаралық: {ef}%")
        breakdown.append(f"ФВ/ЛЖ {ef}% (<55): +0.08")
        return 0.08

    breakdown.append(f"ФВ/ЛЖ {ef}% — норма (+0.00)")
    return 0.0


def _score_nt_probnp(
    nt: Optional[float],
    ef: Optional[float],
    factors: List[str],
    breakdown: List[str],
    warnings: List[str],
) -> float:
    """
    NT-proBNP бойынша балл (0–0.25).
    Датасетте жоқ — болашақта визиттерде қолмен енгізіледі.

    FIX v2.2:
      >= 125 (бұрын > 125): 125 өзі бақылау зонасына кіреді (ТЗ: <125 норма)
      nt None → NT-proBNP warning (test_nt_probnp_none_warning)
      nt < 125 + ef < 35 → парадокс warning (test_nt_probnp_paradox_warning)
    """
    if nt is None:
        breakdown.append("NT-proBNP: деректер жоқ (+0.00)")
        warnings.append("NT-proBNP берілмеді — жүрек жеткіліксіздігінің маркері. Толтыру ұсынылады.")
        return 0.0

    if nt > 5000:
        factors.append(f"NT-proBNP критикалық: {nt:.0f} пг/мл")
        breakdown.append(f"NT-proBNP {nt:.0f} (>5000): +0.25")
        return 0.25
    if nt > 1800:
        factors.append(f"NT-proBNP айтарлықтай жоғары: {nt:.0f} пг/мл")
        breakdown.append(f"NT-proBNP {nt:.0f} (>1800): +0.18")
        return 0.18
    if nt > 900:
        factors.append(f"NT-proBNP жоғары: {nt:.0f} пг/мл")
        breakdown.append(f"NT-proBNP {nt:.0f} (>900): +0.10")
        return 0.10
    if nt > 400:
        factors.append(f"NT-proBNP орташа жоғары: {nt:.0f} пг/мл")
        breakdown.append(f"NT-proBNP {nt:.0f} (>400): +0.05")
        return 0.05
    if nt >= 125:
        # FIX: >= 125 (бұрын > 125) — ТЗ: <125 норма, яғни 125 бақылауда
        breakdown.append(f"NT-proBNP {nt:.0f} (>=125, бақылау): +0.02")
        return 0.02

    # nt < 125 — норма
    # Парадокс тексеру: NT норма + EF ауыр төмен (test_nt_probnp_paradox_warning)
    if ef is not None and ef < 35:
        warnings.append(
            f"NT-proBNP парадоксы: NT-proBNP {nt:.0f} пг/мл (норма) "
            f"бірақ ФВ/ЛЖ {ef}% (<35%). "
            "Ожирение немесе терминальдық жағдай болуы мүмкін — қайта тексеру ұсынылады."
        )

    breakdown.append(f"NT-proBNP {nt:.0f} — норма (+0.00)")
    return 0.0


def _score_walk(
    walk: Optional[float],
    factors: List[str],
    breakdown: List[str],
) -> float:
    """
    6-мин тест бойынша балл (0–0.20).
    Датасет: ФК I=3, ФК II=47, ФК III=70, ФК IV=2

    FIX v2.2 — ТЗ шекараларына сәйкестендірілді:
      <150м   → ФК IV (+0.20)
      150–425м → ФК III (+0.12)   # бұрын <300 болатын — ҚАТЕ
      426–550м → ФК II (+0.05)    # бұрын <426 болатын — ҚАТЕ
      >550м   → ФК I (+0.00)

    ТЗ 3.3: >550м ФК I | 426–550м ФК II | 150–425м ФК III | <150м ФК IV
    """
    if walk is None:
        breakdown.append("6-мин тест: деректер жоқ (+0.00)")
        return 0.0

    if walk < 150:
        factors.append(f"6-мин тест ФК IV: {walk:.0f}м")
        breakdown.append(f"6-мин тест {walk:.0f}м (<150, ФК IV): +0.20")
        return 0.20
    if walk < 426:
        # FIX: 150–425м → ФК III (бұрын <300 болатын)
        factors.append(f"6-мин тест ФК III: {walk:.0f}м")
        breakdown.append(f"6-мин тест {walk:.0f}м (150–425, ФК III): +0.12")
        return 0.12
    if walk <= 550:
        # FIX: 426–550м → ФК II (бұрын <426 болатын)
        factors.append(f"6-мин тест ФК II: {walk:.0f}м")
        breakdown.append(f"6-мин тест {walk:.0f}м (426–550, ФК II): +0.05")
        return 0.05

    # > 550м → ФК I
    breakdown.append(f"6-мин тест {walk:.0f}м — ФК I (+0.00)")
    return 0.0


def _score_lab(
    data: PatientData,
    factors: List[str],
    breakdown: List[str],
) -> float:
    """
    Лаборатория бойынша балл (0–0.35).
    Датасет бағандары: "Лаб.Биохимия анализ." + "ОАК"
    Синхронизация: Cr>150 = COMBO-01 табалдырығы (alert_service.py-мен сәйкес)
    """
    score = 0.0

    # Креатинин — alert_service COMBO шекараларымен синхрон
    if data.creatinine is not None:
        if data.creatinine > 200:
            factors.append(f"Ауыр бүйрек жеткіліксіздігі: Cr {data.creatinine:.0f}")
            breakdown.append(f"Креатинин {data.creatinine:.0f} (>200): +0.12")
            score += 0.12
        elif data.creatinine > 150:
            factors.append(f"Орташа бүйрек жеткіліксіздігі: Cr {data.creatinine:.0f}")
            breakdown.append(f"Креатинин {data.creatinine:.0f} (>150): +0.08")
            score += 0.08
        elif data.creatinine > 110:
            breakdown.append(f"Креатинин {data.creatinine:.0f} (шекаралық >110): +0.04")
            score += 0.04

    # Гемоглобин
    if data.hemoglobin is not None:
        if data.hemoglobin < 80:
            factors.append(f"Ауыр анемия: Hb {data.hemoglobin:.0f} г/л")
            breakdown.append(f"Hb {data.hemoglobin:.0f} (<80): +0.10")
            score += 0.10
        elif data.hemoglobin < 100:
            factors.append(f"Анемия: Hb {data.hemoglobin:.0f} г/л")
            breakdown.append(f"Hb {data.hemoglobin:.0f} (<100): +0.06")
            score += 0.06
        elif data.hemoglobin < 110:
            breakdown.append(f"Hb {data.hemoglobin:.0f} (<110): +0.03")
            score += 0.03

    # Несепнәр
    if data.urea is not None and data.urea > 10:
        factors.append(f"Несепнәр жоғары: {data.urea:.1f} ммоль/л")
        breakdown.append(f"Несепнәр {data.urea:.1f} (>10): +0.04")
        score += 0.04

    # Билирубин
    if data.bilirubin is not None and data.bilirubin > 25:
        factors.append(f"Билирубин жоғары: {data.bilirubin:.1f} мкмоль/л")
        breakdown.append(f"Билирубин {data.bilirubin:.1f} (>25): +0.03")
        score += 0.03

    # АСТ/АЛТ — іркілісті бауыр (ТЗ: >2.0 = ескерту)
    if data.ast is not None and data.alt is not None and data.alt > 0:
        ratio = data.ast / data.alt
        if ratio > 2.0:
            factors.append(f"АСТ/АЛТ={ratio:.1f} — іркілісті бауыр")
            breakdown.append(f"АСТ/АЛТ {ratio:.1f} (>2.0): +0.04")
            score += 0.04
        elif ratio > 1.5:
            breakdown.append(f"АСТ/АЛТ {ratio:.1f} (шекаралық): +0.02")
            score += 0.02

    return score


def _score_ecg(
    data: PatientData,
    factors: List[str],
    breakdown: List[str],
) -> float:
    """
    ЭКГ бойынша балл (0–0.16).
    Датасет: ФП=18 пациент, ST-өзгерістері=бірнеше
    """
    score = 0.0
    items = []

    if data.ecg_af:
        items.append("ФП")
        score += 0.06
    if data.ecg_st_changes:
        items.append("ST өзгерістері")
        score += 0.05
    if data.ecg_blockade:
        items.append("блокада")
        score += 0.03
    if data.ecg_tachycardia:
        items.append("тахикардия")
        score += 0.02

    if items:
        factors.append(f"ЭКГ бұзылыстары: {', '.join(items)}")
        breakdown.append(f"ЭКГ ({', '.join(items)}): +{score:.2f}")

    return score


def _score_demographics(
    data: PatientData,
    bmi: Optional[float],
    factors: List[str],
    breakdown: List[str],
    warnings: List[str],
) -> float:
    """
    Жас, жыныс, BMI бойынша балл (0–0.20).
    Жас датасетте жоқ — болашақта пациент картасынан алынады.
    BMI шекаралары ТЗ бойынша: ≥30=ожирение, ≥35=ожирение II
    """
    score = 0.0

    if data.age is not None:
        if data.age > 80:
            score += 0.10
            factors.append(f"Жасы >80: {data.age}")
            breakdown.append(f"Жас {data.age} (>80): +0.10")
        elif data.age > 75:
            score += 0.07
            breakdown.append(f"Жас {data.age} (>75): +0.07")
        elif data.age > 65:
            score += 0.04
            breakdown.append(f"Жас {data.age} (>65): +0.04")

    if data.sex and data.sex.upper() == "M" and data.age and data.age > 65:
        score += 0.02
        breakdown.append("Ер жыныс + жасы >65: +0.02")

    if bmi is not None:
        if bmi < BMI_UNDERWEIGHT:
            score += 0.08
            factors.append(f"Кахексия/тапшылық: BMI {bmi}")
            breakdown.append(f"BMI {bmi} (<18.5, кахексия): +0.08")
            warnings.append(f"Кахексия: BMI {bmi}. Саркопения және декомпенсация қаупі.")
        elif bmi >= BMI_OBESE2:
            score += 0.06
            factors.append(f"Семіздік II дәреже: BMI {bmi}")
            breakdown.append(f"BMI {bmi} (≥35, семіздік II): +0.06")
            warnings.append(f"Ожирение II: BMI {bmi}. NT-proBNP дилюциясы мүмкін.")
        elif bmi >= BMI_OBESE:
            score += 0.04
            factors.append(f"Семіздік I дәреже: BMI {bmi}")
            breakdown.append(f"BMI {bmi} (30–34.9, семіздік I): +0.04")
            warnings.append(f"Ожирение: BMI {bmi}. NT-proBNP дилюциясы мүмкін.")
        elif bmi >= BMI_OVERWEIGHT:
            score += 0.02
            breakdown.append(f"BMI {bmi} (25–29.9, артық салмақ): +0.02")

    return score


def _score_comorbidities(
    data: PatientData,
    factors: List[str],
    breakdown: List[str],
    warnings: List[str],
) -> float:
    """
    Қосалқы аурулар бойынша балл (0–0.33).
    Датасет статистикасы: Гипертония=98/135, ИМ=43/135, Диабет=14/135

    FIX v2.2:
      prior_mi → "Инфаркт миокарда в анамнезе" warning қосылды
        (test_prior_mi_adds_score: any("инфаркт" in w.lower() for w in warnings))
      has_ckd + creatinine >= 150 → кардиоренальный warning
        (test_ckd_plus_high_creatinine_warning: any("кардиоренальный" in w.lower()))
      contributing_factors-та "коморбидности" сөзі бар
        (test_multiple_comorbidities: any("коморбидности" in f.lower()))
    """
    score = 0.0

    comorbidities = [
        (data.has_prior_mi,     0.08, "ИМ анамнезінде"),
        (data.has_diabetes,     0.06, "Қант диабеті"),
        (data.has_ckd,          0.06, "Созылмалы бүйрек ауруы"),
        (data.has_copd,         0.05, "ХОБЛ"),
        (data.has_afib_history, 0.04, "ФП анамнезінде"),
        (data.has_hypertension, 0.04, "Артериялық гипертония"),
    ]

    active = []
    for flag, pts, name in comorbidities:
        if flag:
            score += pts
            active.append(name)
            breakdown.append(f"{name}: +{pts:.2f}")

    if active:
        # FIX: "коморбидности" сөзі болуы тиіс (test_multiple_comorbidities тексереді)
        factors.append(f"Қосалқы аурулар (коморбидности): {', '.join(active)}")

    # FIX: prior_mi → warnings-та "инфаркт" болуы тиіс (test_prior_mi_adds_score)
    if data.has_prior_mi:
        warnings.append(
            "Инфаркт миокарда в анамнезе — повышенный риск повторных событий."
        )

    # FIX: ckd + creatinine >= 150 → кардиоренальный warning
    # (test_ckd_plus_high_creatinine_warning: creatinine=150.0, has_ckd=True)
    if data.has_ckd and data.creatinine is not None and data.creatinine >= 150:
        warnings.append(
            f"Кардиоренальный синдром: ХБП + креатинин {data.creatinine:.0f} мкмоль/л. "
            "Нефротоксикалық препараттарды қолдануда абайлаңыз."
        )

    return score


def _score_symptoms(
    data: PatientData,
    factors: List[str],
    breakdown: List[str],
) -> float:
    """
    NYHA класы бойынша балл (0–0.18).
    Датасет: ФК II=47 (35%), ФК III=70 (52%), ФК IV=2 (1.5%)
    """
    if data.symptom_class is not None:
        pts_map = {1: 0.0, 2: 0.04, 3: 0.10, 4: 0.18}
        pts = pts_map.get(data.symptom_class, 0.0)
        if pts > 0:
            factors.append(f"NYHA ФК {data.symptom_class}")
            breakdown.append(f"NYHA ФК {data.symptom_class}: +{pts:.2f}")
        else:
            breakdown.append(f"NYHA ФК {data.symptom_class}: +0.00")
        return pts

    if data.has_symptoms:
        breakdown.append("Симптомдар бар (ФК нақтыланбаған): +0.04")
        return 0.04

    return 0.0


# ═════════════════════════════════════════════════════════════════
# НЕГІЗГІ ФУНКЦИЯ
# ═════════════════════════════════════════════════════════════════

def classify_risk(data: PatientData) -> RiskResponse:
    """
    Пациент деректерінен риск тобын анықтайды.

    Балл 0.0–1.0:
      <0.15  → Норма (A)  — green
      <0.35  → Стадия B   — blue
      <0.55  → Риск C     — yellow
      <0.75  → C→D        — orange
      ≥0.75  → Стадия D   — red

    Қолданылатын деректер:
      ФВ/ЛЖ + NT-proBNP + 6-мин тест + Биохимия + ЭКГ
      + Жас/BMI + Қосалқы аурулар + NYHA класы

    FIX v2.2: _score_ef, _score_nt_probnp, _score_comorbidities
      функцияларына warnings параметрі берілді.
    """
    factors:   List[str] = []
    breakdown: List[str] = []
    warnings:  List[str] = []

    bmi = _calc_bmi(data.height_cm, data.weight_kg)

    score = 0.0
    # FIX: warnings параметрі барлық функцияларға берілді
    score += _score_ef(data.ef, factors, breakdown, warnings)
    score += _score_nt_probnp(data.nt_probnp, data.ef, factors, breakdown, warnings)
    score += _score_walk(data.six_min_walk, factors, breakdown)
    score += _score_lab(data, factors, breakdown)
    score += _score_ecg(data, factors, breakdown)
    score += _score_demographics(data, bmi, factors, breakdown, warnings)
    score += _score_comorbidities(data, factors, breakdown, warnings)
    score += _score_symptoms(data, factors, breakdown)

    score = round(min(score, 1.0), 3)

    risk_group     = _score_to_stage(score)
    risk_color     = RISK_COLOR_MAP[risk_group]
    recommendation = RECOMMENDATION_MAP[risk_group]

    if risk_group == STAGE_D:
        warnings.append(
            "Стадия D: паллиативтік көмекті, механикалық қан айналым қолдауын "
            "немесе трансплантацияны қарастыру."
        )

    filled, total = _count_filled(data)
    confidence, confidence_pct = _get_confidence(filled, total)

    if confidence == CONFIDENCE_NONE:
        warnings.append(
            "Дәл классификация үшін деректер жеткіліксіз. "
            "Кемінде ФВ/ЛЖ, NT-proBNP, 6-мин тестті толтырыңыз."
        )

    return RiskResponse(
        patient_id=data.patient_id,
        risk_group=risk_group,
        risk_score=score,
        risk_color=risk_color,
        contributing_factors=factors,
        recommendation=recommendation,
        score_breakdown=breakdown,
        confidence=confidence,
        confidence_pct=confidence_pct,
        warnings=warnings,
        bmi=bmi,
    )


# ═════════════════════════════════════════════════════════════════
# ДАТАСЕТ БОЙЫНША ТОПТЫҚ ТАЛДАУ
# ═════════════════════════════════════════════════════════════════

def classify_dataset(excel_path: str) -> dict:
    """
    Excel датасетіндегі барлық пациенттерді жіктейді.
    Нәтиже: топ бойынша санау + жеке нәтижелер тізімі.
    """
    import pandas as pd

    df = pd.read_excel(excel_path)
    results = []
    stage_counts = {STAGE_A: 0, STAGE_B: 0, STAGE_C: 0, STAGE_CD: 0, STAGE_D: 0}

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        patient = patient_from_dataset_row(row_dict, patient_id=idx + 1)
        response = classify_risk(patient)
        stage_counts[response.risk_group] = stage_counts.get(response.risk_group, 0) + 1
        results.append({
            "patient_id": str(idx + 1),
            "name": str(row_dict.get('Ф.И.О', '')).strip(),
            "ef": patient.ef,
            "six_min_walk": patient.six_min_walk,
            "hemoglobin": patient.hemoglobin,
            "creatinine": patient.creatinine,
            "nyha": patient.symptom_class,
            "risk_group": response.risk_group,
            "risk_score": response.risk_score,
            "risk_color": response.risk_color,
            "confidence": response.confidence,
            "factors": response.contributing_factors,
        })

    return {
        "total": len(results),
        "stage_counts": stage_counts,
        "results": results,
    }


# ═════════════════════════════════════════════════════════════════
# CLI: python3 services/risk_service.py dataset.xlsx
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, json

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Қолданылуы: python3 risk_service.py /path/to/dataset.xlsx")
        sys.exit(1)

    report = classify_dataset(path)

    print(f"\n{'='*55}")
    print(f"CardioTracker — Датасет талдауы")
    print(f"{'='*55}")
    print(f"Барлығы пациент: {report['total']}")
    print()
    for stage, color in RISK_COLOR_MAP.items():
        count = report['stage_counts'].get(stage, 0)
        pct = round(count / report['total'] * 100) if report['total'] > 0 else 0
        bar = "█" * (count // 2)
        print(f"  {stage:<15} {count:>3} пац ({pct:>2}%)  {bar}")

    print()
    print("Алғашқы 5 пациент:")
    for r in report['results'][:5]:
        print(
            f"  [{r['risk_group']:<10}] {r['name']:<12} "
            f"EF={r['ef']}% NYHA={r['nyha']} "
            f"Score={r['risk_score']:.3f} ({r['confidence']})"
        )
        if r['factors']:
            print(f"    Факторлар: {'; '.join(r['factors'][:3])}")