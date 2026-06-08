"""
data_parser.py — ХСН датасетін оқу және парсинг жасау утилиті.
CardioTracker ML v2.2

Датасет форматтары (нақты ХСН_ОНМТ_новый_.xlsx бағандарынан):
  ФВ/ЛЖ        : 0.42 | "46%;" | "42.0"  → float (%)
  Тест 6 minute: "фкIII" | "фк lll" | "ФК II" | "300м" → float (метр)
  Биохимия     : "Мочевина ммоль/л5,9,Креатинин ммоль/л70,..." мәтіні
  ОАК          : "лейкоциты 5,0,эритроциты 3,4,Гемоглобин 105" мәтіні
  ЭКГ          : "Фибрилляция предсердий" | "депрессий сегмента V1-V5"
  Рост/Вес     : бағандар бойынша (санмен)
  NT-proBNP    : датасетте жоқ → None (визитте қолмен енгізіледі)

Датасет статистикасы (135 пациент):
  ФВ/ЛЖ:      130/135 (96%) — диапазон 40–68%, орта 51.4%
  Тест 6 мин: 117/122 (96%) — парсинг сәтті
  Гемоглобин: 120/135 (89%) — 14 пациент <110 г/л (ALERT-03 зонасы)
  Креатинин:  120/135 (89%) — диапазон 7–115 мкмоль/л
  ФП (ЭКГ):   13/135 пациент
  ST өзгерісі: 14/135 пациент

Түзетулер v2.2:
  ✅ parse_6min_walk: "фк" (нөмірсіз) → None + parse_warnings жазу
  ✅ parse_6min_walk: "1+" (митральдық балл) → None (дұрыс мінез-құлық)
  ✅ parse_6min_walk: бос жолдар "      " → None
  ✅ parse_nyha_class: diagnosis + six_min бағандарынан бірлескен іздеу
  ✅ parse_comorbidities: жеке функция (risk_service.py-мен синхрон)
  ✅ load_dataset: parse_warnings есебі (қанша жол парсинг болмады)
  ✅ get_dataset_stats: ecg детальды статистика қосылды
  ✅ CLI: толық есеп + алғашқы 3 пациент JSON-мен
"""

import re
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  ПАРСЕРЛЕР
# ═════════════════════════════════════════════════════════════════════════════

def parse_ef_lv(val) -> Optional[float]:
    """
    ФВ/ЛЖ → float (%).
    0.42 → 42.0 | "46%;" → 46.0 | "42.5" → 42.5

    Датасетте барлық мәндер 0.40–0.68 форматында (дробь).
    Физиологиялық шек: 5–100%.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).replace('%', '').replace(';', '').replace(',', '.').strip()
    try:
        v = float(s)
        if v <= 1.0:                   # 0.42 форматы → 42.0%
            v = round(v * 100, 1)
        if not (5.0 <= v <= 100.0):    # физиологиялық диапазон
            return None
        return round(v, 1)
    except ValueError:
        return None


def parse_6min_walk(text) -> Optional[float]:
    """
    6-минуттық тест → метр (float).

    Датасеттегі нақты форматтар:
      "фкIII", "фк lll", "ФК II", "фк ll", "фк IV", "ФК I"
      "300м", "350 м"
      "фк"    → None (нөмір жоқ)
      "1+"    → None (митральдық регургитация баллы, тест емес)
      "      " → None (бос)

    ТЗ шкаласы бойынша орта мәндер:
      ФК IV (<150м)   → 75м
      ФК III (150–425м) → 250м  (консервативті орта)
      ФК II (426–550м) → 450м
      ФК I (>550м)    → 560м
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None

    text = str(text).strip()
    if not text:
        return None

    t = text.lower()

    # Нақты метр: "320 м", "320м"
    m = re.search(r'(\d{2,4})\s*м', t)
    if m:
        val = float(m.group(1))
        if 10 <= val <= 1000:
            return val

    # ФК IV
    if re.search(r'фк\s*(iv|4)\b', t):
        return 75.0

    # ФК III (lll — кириллица/латын аралас жазу)
    if re.search(r'фк\s*(iii|3|lll)\b', t):
        return 250.0

    # ФК II (ll — латын L аралас)
    if re.search(r'фк\s*(ii|2|ll)\b', t):
        return 450.0

    # ФК I (тек "i" немесе "1" болса ғана — "ii" мен шатастырмау үшін \b)
    if re.search(r'фк\s*(i|1)\b', t) and not re.search(r'фк\s*(ii|iii|iv)', t):
        return 560.0

    # FIX v2.2: "фк" (нөмірсіз), "1+", бос жолдар → None, лог жазу
    if 'фк' in t:
        logger.debug("parse_6min_walk: 'фк' нөмірсіз, мән анықталмады: %r", text)
    return None


def parse_biochemistry(text) -> dict:
    """
    Биохимия мәтінінен сандық мәндерді шығару.

    Датасет мысалы:
      "Общий белок г/л61,Мочевина ммоль/л5,9,Креатинин ммоль/л70,
       Билирубин общийммоль/л 10,"

    Қайтарады: {urea, creatinine, bilirubin, protein}
    Физиологиялық диапазон тексеруі қосылған.
    """
    result = {}
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return result
    text = str(text)

    patterns = {
        'urea':       r'[Мм]очевина[^\d]*(\d+[,.]?\d*)',
        'creatinine': r'[Кк]реатинин[^\d]*(\d+[,.]?\d*)',
        'bilirubin':  r'[Бб]илирубин[^\d]*(\d+[,.]?\d*)',
        'protein':    r'[Бб]елок[^\d]*(\d+[,.]?\d*)',
    }
    # Физиологиялық шектер
    limits = {
        'urea':       (0, 100),
        'creatinine': (0, 2000),
        'bilirubin':  (0, 500),
        'protein':    (0, 200),
    }

    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            try:
                val = float(m.group(1).replace(',', '.'))
                lo, hi = limits[key]
                if lo < val < hi:
                    result[key] = val
            except ValueError:
                pass
    return result


def parse_oac(text) -> dict:
    """
    ОАК (жалпы қан анализі) мәтінінен мәндер шығару.

    Датасет мысалы:
      "лейкоциты 5,0,эритроциты 3,4,Гемоглобин 105"

    Қайтарады: {hemoglobin, leukocytes, erythrocytes}
    """
    result = {}
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return result
    text = str(text)

    patterns = {
        'hemoglobin':   r'[Гг]емоглобин[^\d]*(\d+[,.]?\d*)',
        'leukocytes':   r'[Лл]ейкоциты[^\d]*(\d+[,.]?\d*)',
        'erythrocytes': r'[Ээ]ритроциты[^\d]*(\d+[,.]?\d*)',
    }
    limits = {
        'hemoglobin':   (30,  300),
        'leukocytes':   (0,   100),
        'erythrocytes': (0,   20),
    }

    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            try:
                val = float(m.group(1).replace(',', '.'))
                lo, hi = limits[key]
                if lo < val < hi:
                    result[key] = val
            except ValueError:
                pass
    return result


def parse_ecg_flags(text) -> dict:
    """
    ЭКГ мәтінінен клиникалық флагтарды анықтау.

    Датасет мысалдары:
      "Фибрилляция предсердий постоянная ЧЖС 70-80 уд/мин"
      "депрессий сегмента V1-V5 ГЛЖ"
      "Ритм синусовый ЧСС 65 уд/мин"

    Қайтарады: {atrial_fibrillation, tachycardia, bundle_branch_block, st_changes}
    """
    default = {
        'atrial_fibrillation': False,
        'tachycardia':         False,
        'bundle_branch_block': False,
        'st_changes':          False,
    }
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return default

    t = str(text).lower()
    return {
        'atrial_fibrillation': bool(re.search(r'фибрилл|мерцател', t)),
        'tachycardia':         bool(re.search(r'тахикард', t)),
        'bundle_branch_block': bool(re.search(r'блокад', t)),
        'st_changes':          bool(re.search(
            r'\bst\b|сегмент\s*st|депресси|элевац|субэндокард', t
        )),
    }


def parse_height_weight(row: pd.Series) -> tuple:
    """
    Рост (см) және Вес (кг) бағандарынан мәндерді шығару.

    Датасеттегі нақты баған атаулары: 'Рост', 'Вес '
    Бірнеше вариантты тексереді.
    """
    height_keys = ['Рост', 'рост', 'Height', 'height', 'Рост (см)', 'Рост,см']
    weight_keys = ['Вес ', 'Вес', 'вес', 'Weight', 'weight', 'Вес (кг)', 'Вес,кг']

    height = None
    weight = None

    for k in height_keys:
        if k in row.index:
            try:
                v = float(str(row[k]).replace(',', '.'))
                if 100 <= v <= 220:
                    height = v
                    break
            except (ValueError, TypeError):
                pass

    for k in weight_keys:
        if k in row.index:
            try:
                v = float(str(row[k]).replace(',', '.'))
                if 30 <= v <= 250:
                    weight = v
                    break
            except (ValueError, TypeError):
                pass

    return height, weight


def parse_nyha_class(row_or_text) -> Optional[int]:
    """
    NYHA функционалдық класын анықтайды.

    FIX v2.2: pd.Series немесе str қабылдайды.
    pd.Series болса — 'Тест 6 minute' + 'основной диагноз ' бағандарын бірге тексереді.
    str болса — тікелей мәтіннен іздейді.

    Датасет статистикасы: ФК I=3, ФК II=47, ФК III=70, ФК IV=2
    """
    if isinstance(row_or_text, pd.Series):
        sources = [
            str(row_or_text.get('Тест 6 minute', '') or ''),
            str(row_or_text.get('основной диагноз ', '') or ''),
            str(row_or_text.get('Сопутствующий диагноз ', '') or ''),
        ]
        t = ' '.join(sources).lower()
    else:
        if row_or_text is None or (isinstance(row_or_text, float) and pd.isna(row_or_text)):
            return None
        t = str(row_or_text).lower()

    if re.search(r'(фк|nyha)\s*(iv|4)\b',    t): return 4
    if re.search(r'(фк|nyha)\s*(iii|3|lll)\b', t): return 3
    if re.search(r'(фк|nyha)\s*(ii|2|ll)\b',   t): return 2
    if re.search(r'(фк|nyha)\s*(i|1)\b',       t) and \
       not re.search(r'(фк|nyha)\s*(ii|iii|iv)', t): return 1
    return None


def parse_comorbidities(row: pd.Series) -> dict:
    """
    Қосалқы аурулардың флагтарын анықтайды.
    FIX v2.2: risk_service.py _score_comorbidities-мен синхрон.

    Датасет статистикасы: Гипертония=98, ИМ=43, Диабет=14, ФП=18, ХОБЛ=3
    """
    combined = (
        str(row.get('Сопутствующий диагноз ', '') or '') + ' ' +
        str(row.get('основной диагноз ', '') or '')
    ).lower()

    return {
        'has_hypertension': bool(re.search(r'гипертон|гипертенз', combined)),
        'has_prior_mi':     bool(re.search(r'инфаркт|пикс', combined)),
        'has_diabetes':     bool(re.search(r'диабет|сд\s*\d', combined)),
        'has_copd':         bool(re.search(r'хобл|бронхит', combined)),
        'has_afib_history': bool(re.search(r'фибрилл|мерцательн', combined)),
        'has_ckd':          bool(re.search(r'хбп|хроническ.*почеч|нефропат', combined)),
    }


def _safe_str(val) -> Optional[str]:
    """None / NaN → None, иначе stripped str."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return s if s and s.lower() != 'nan' else None


# ═════════════════════════════════════════════════════════════════════════════
#  БАҒАН МАППИНГІ
# ═════════════════════════════════════════════════════════════════════════════

COLUMN_MAP = {
    # Пациент
    'name':       ['Ф.И.О', 'ФИО', 'Имя', 'Пациент', 'Ф.И.О.'],
    'diagnosis':  ['основной диагноз ', 'Диагноз', 'диагноз', 'основной диагноз'],
    'blood_type': ['Группа крови ', 'Группа крови', 'Кровь'],
    # Клиника
    'ef_lv':      ['ФВ/ЛЖ', 'ФВ ЛЖ', 'EF', 'ФВ'],
    'six_min':    ['Тест 6 minute', 'Тест 6 мин', '6 мин тест', 'Тест_6мин'],
    'ecg':        ['ЭКГ', 'экг', 'ECG'],
    # Биохимия
    'biochem':    ['Лаб.Биохимия анализ.', 'Биохимия', 'биохимия', 'Биохим'],
    # ОАК
    'oac':        ['ОАК', 'оак', 'Общий анализ крови'],
    # Эхо
    'kdo':        ['ЭХОКГ КДО', 'КДО', 'kdo'],
    'ef_rv':      ['ФВ/ПЖ', 'ФВ ПЖ', 'ef_rv'],
}


def _get_col(row: pd.Series, key: str):
    """Баған атауларының тізімінен бірінші табылғанды қайтарады."""
    for col_name in COLUMN_MAP.get(key, []):
        if col_name in row.index:
            return row[col_name]
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  ЖОЛДЫ ПАРСИНГ
# ═════════════════════════════════════════════════════════════════════════════

def parse_row(row: pd.Series) -> dict:
    """
    Excel жолын → CardioTracker форматына айналдыру.
    Барлық өрістер қауіпсіз: None болуы мүмкін.

    FIX v2.2: comorbidities + nyha parse_row ішінде есептеледі
    """
    biochem       = parse_biochemistry(_get_col(row, 'biochem'))
    oac           = parse_oac(_get_col(row, 'oac'))
    ecg_flags     = parse_ecg_flags(_get_col(row, 'ecg'))
    height, weight = parse_height_weight(row)
    comorbidities = parse_comorbidities(row)
    nyha          = parse_nyha_class(row)

    return {
        # Пациент
        'name':       _safe_str(_get_col(row, 'name')),
        'blood_type': _safe_str(_get_col(row, 'blood_type')),
        'diagnosis':  _safe_str(_get_col(row, 'diagnosis')),

        # 8 клиникалық параметр
        'ef_lv':        parse_ef_lv(_get_col(row, 'ef_lv')),
        'six_min_walk': parse_6min_walk(_get_col(row, 'six_min')),
        'nt_pro_bnp':   None,   # датасетте жоқ — визитте қолмен енгізіледі

        # Антропометрия
        'height_cm': height,
        'weight_kg': weight,

        # ЭКГ
        'ecg_flags': ecg_flags,
        'ecg_text':  _safe_str(_get_col(row, 'ecg')),

        # ОАК
        'hemoglobin':   oac.get('hemoglobin'),
        'leukocytes':   oac.get('leukocytes'),
        'erythrocytes': oac.get('erythrocytes'),

        # Биохимия
        'urea':       biochem.get('urea'),
        'creatinine': biochem.get('creatinine'),
        'bilirubin':  biochem.get('bilirubin'),
        'protein':    biochem.get('protein'),

        # NYHA класы
        'symptom_class': nyha,
        'has_symptoms':  nyha is not None and nyha >= 2,

        # Коморбидность (risk_service.py-мен синхрон)
        **comorbidities,

        # Эхо
        'kdo':   _get_col(row, 'kdo'),
        'ef_rv': _get_col(row, 'ef_rv'),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  ДАТАСЕТ ЖҮКТЕУ
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str, skip_empty: bool = True) -> list:
    """
    Excel датасетін оқып, барлық пациенттерді парсинг жасайды.

    Args:
        path:        .xlsx немесе .xls файл жолы
        skip_empty:  ФВ/ЛЖ да аты да жоқ жолдарды өткізіп жіберу

    Returns:
        list[dict] — parse_row() нәтижелері

    FIX v2.2: parse_warnings есебі — қанша жол парсинг болмады
    """
    try:
        df = pd.read_excel(path, dtype=str)
    except Exception as e:
        raise ValueError(f"Excel оқу қатесі: {e}")

    patients       = []
    skipped        = 0
    walk_failed    = 0
    ef_failed      = 0

    for _, row in df.iterrows():
        parsed = parse_row(row)

        if skip_empty:
            has_ef   = parsed.get('ef_lv') is not None
            has_name = parsed.get('name') is not None
            if not has_ef and not has_name:
                skipped += 1
                continue

        # Парсинг статистикасы
        if parsed.get('ef_lv') is None and _get_col(row, 'ef_lv') is not None:
            ef_failed += 1
        if parsed.get('six_min_walk') is None and _get_col(row, 'six_min') is not None:
            walk_failed += 1

        patients.append(parsed)

    logger.info(
        "[data_parser] Оқылды: %d пациент | Өткізілді: %d бос жол | "
        "ФВ сәтсіз: %d | Тест сәтсіз: %d",
        len(patients), skipped, ef_failed, walk_failed
    )
    print(
        f"[data_parser] Оқылды: {len(patients)} пациент | "
        f"Өткізілді: {skipped} | ФВ parse қате: {ef_failed} | "
        f"Тест parse қате: {walk_failed}"
    )
    return patients


# ═════════════════════════════════════════════════════════════════════════════
#  СТАТИСТИКА
# ═════════════════════════════════════════════════════════════════════════════

def get_dataset_stats(patients: list) -> dict:
    """
    Датасет статистикасы — қанша өріс толтырылған.
    FIX v2.2: ЭКГ детальды статистика + коморбидность қосылды.
    """
    total = len(patients)
    if total == 0:
        return {}

    fields = [
        'ef_lv', 'six_min_walk', 'hemoglobin', 'creatinine',
        'urea', 'bilirubin', 'height_cm', 'weight_kg',
    ]

    stats = {'total': total}
    for f in fields:
        filled = sum(1 for p in patients if p.get(f) is not None)
        stats[f] = {'filled': filled, 'pct': round(filled / total * 100)}

    # ЭКГ детальды
    ecg_any = sum(
        1 for p in patients if any(p.get('ecg_flags', {}).values())
    )
    ecg_af = sum(
        1 for p in patients if p.get('ecg_flags', {}).get('atrial_fibrillation')
    )
    ecg_st = sum(
        1 for p in patients if p.get('ecg_flags', {}).get('st_changes')
    )
    stats['ecg_any']           = {'filled': ecg_any, 'pct': round(ecg_any / total * 100)}
    stats['ecg_af']            = {'filled': ecg_af,  'pct': round(ecg_af  / total * 100)}
    stats['ecg_st_changes']    = {'filled': ecg_st,  'pct': round(ecg_st  / total * 100)}

    # Коморбидность
    for flag in ['has_hypertension','has_prior_mi','has_diabetes','has_copd','has_afib_history','has_ckd']:
        cnt = sum(1 for p in patients if p.get(flag))
        stats[flag] = {'filled': cnt, 'pct': round(cnt / total * 100)}

    # NYHA
    for fc in [1, 2, 3, 4]:
        cnt = sum(1 for p in patients if p.get('symptom_class') == fc)
        stats[f'nyha_{fc}'] = {'filled': cnt, 'pct': round(cnt / total * 100)}

    return stats


# ═════════════════════════════════════════════════════════════════════════════
#  CLI: python3 data_parser.py dataset.xlsx
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import json
    import sys

    logging.basicConfig(level=logging.WARNING)

    path = sys.argv[1] if len(sys.argv) > 1 else 'dataset.xlsx'
    patients = load_dataset(path)

    stats = get_dataset_stats(patients)
    total = stats['total']

    print(f"\n{'='*55}")
    print(f"CardioTracker — Датасет статистикасы")
    print(f"{'='*55}")
    print(f"Барлығы пациент: {total}")

    print(f"\n── Клиникалық параметрлер ──────────────────────────")
    for f in ['ef_lv','six_min_walk','hemoglobin','creatinine','urea','bilirubin','height_cm','weight_kg']:
        v = stats.get(f, {})
        print(f"  {f:<18}: {v.get('filled',0):>3}/{total}  ({v.get('pct',0)}%)")

    print(f"\n── ЭКГ ─────────────────────────────────────────────")
    for k in ['ecg_any','ecg_af','ecg_st_changes']:
        v = stats.get(k, {})
        print(f"  {k:<18}: {v.get('filled',0):>3}/{total}  ({v.get('pct',0)}%)")

    print(f"\n── Коморбидность ───────────────────────────────────")
    for k in ['has_hypertension','has_prior_mi','has_diabetes','has_copd','has_afib_history','has_ckd']:
        v = stats.get(k, {})
        print(f"  {k:<22}: {v.get('filled',0):>3}/{total}  ({v.get('pct',0)}%)")

    print(f"\n── NYHA ────────────────────────────────────────────")
    for fc in [1,2,3,4]:
        v = stats.get(f'nyha_{fc}', {})
        bar = '█' * v.get('filled', 0)
        print(f"  ФК {fc}: {v.get('filled',0):>3} пац ({v.get('pct',0):>2}%)  {bar}")

    print(f"\n── Алғашқы 3 пациент ───────────────────────────────")
    for p in patients[:3]:
        clean = {k: v for k, v in p.items()
                 if v is not None and v is not False and v != {}}
        print(json.dumps(clean, ensure_ascii=False, indent=2))
        print()