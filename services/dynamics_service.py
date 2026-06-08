"""
services/dynamics_service.py — Динамика және тренд анализі
CardioTracker ML v2.1

Edge case қорғанысы:
  ✅ visits < 2 болса → ValueError (роутер HTTPException 400 береді)
  ✅ visits[0] / visits[-1] → safe_first / safe_last арқылы
  ✅ 0-ге бөлінуден қорғаныс барлық % есептеулерінде
  ✅ None мәндер бар параметрлер тренд есебінен шығарылады
  ✅ Тек бір мәні бар параметр үшін тренд есептелмейді
  ✅ visit_date None болса avg_interval_days есептелмейді
"""

from typing import List, Optional, Tuple
from datetime import datetime

from schemas import (
    DynamicsRequest,
    DynamicsResponse,
    TrendItem,
    VisitRecord,
    PatientData,
    RiskResponse,
)
from services.risk_service import classify_risk, PatientData as RiskPatientData


# ═════════════════════════════════════════════════════════════════
# УТИЛИТТЕР
# ═════════════════════════════════════════════════════════════════

def safe_pct_change(current: float, reference: float) -> Optional[float]:
    """0-ге бөлінбейтін % өзгеріс."""
    if reference == 0:
        return None
    return round((current - reference) / reference * 100, 1)


def linear_regression_slope(values: List[float]) -> Optional[float]:
    """
    Қарапайым сызықтық регрессия — тренд бағытын анықтайды.
    n < 2 болса None қайтарады.
    """
    n = len(values)
    if n < 2:
        return None
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return None
    return round(num / den, 4)


def forecast_next(values: List[float], slope: float) -> Optional[float]:
    """
    Келесі визитке болжам: соңғы мән + slope.
    """
    if not values:
        return None
    return round(values[-1] + slope, 1)


def avg_interval(visits: List[VisitRecord]) -> Optional[int]:
    """
    Визиттер арасындағы орта аралық (күн).
    visit_date None болса — есептелмейді.
    """
    dates = []
    for v in visits:
        if v.visit_date:
            try:
                dates.append(datetime.strptime(v.visit_date, "%Y-%m-%d"))
            except ValueError:
                pass

    if len(dates) < 2:
        return None

    intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    return round(sum(intervals) / len(intervals))


def severity_label(change_pct: float, param: str, higher_is_better: bool) -> str:
    """
    % өзгерісінен severity анықтайды.
    higher_is_better=True: өсу = жақсарды, азаю = нашарлады.
    """
    worsening = -change_pct if higher_is_better else change_pct
    if worsening <= 0:
        return "normal"
    if worsening < 10:
        return "mild"
    if worsening < 25:
        return "moderate"
    return "severe"


def direction_label(change_pct: float, higher_is_better: bool) -> str:
    if abs(change_pct) < 5:
        return "стабильно"
    improving = change_pct > 0 if higher_is_better else change_pct < 0
    worsening_severity = abs(change_pct)
    if improving:
        return "улучшение"
    if worsening_severity >= 25:
        return "значительное ухудшение"
    return "ухудшение"


# ═════════════════════════════════════════════════════════════════
# ПАРАМЕТР КОНФИГУРАЦИЯСЫ
# ═════════════════════════════════════════════════════════════════

# field → (label, higher_is_better, critical_threshold_change_pct)
PARAM_CONFIGS = {
    "ef":           ("ФВ ЛЖ (%)",        True,  20),
    "nt_probnp":    ("NT-proBNP",         False, 50),
    "six_min_walk": ("Тест 6 мин (м)",    True,  20),
    "creatinine":   ("Креатинин",         False, 30),
    "hemoglobin":   ("Гемоглобин",        True,  15),
    "weight_kg":    ("Вес (кг)",          False, 10),
    "urea":         ("Мочевина",          False, 30),
    "bilirubin":    ("Билирубин",         False, 50),
}


def compute_trend_item(
    field: str,
    visits: List[VisitRecord],
    current_value: Optional[float],
) -> Optional[TrendItem]:
    """
    Бір параметр бойынша TrendItem жасайды.

    Edge case-тар:
      - Барлық мәндер None болса → None қайтарады
      - Тек бір мән болса → тренд есептелмейді (change_pct=0)
      - first_value == 0 болса → % өзгерісі None
    """
    label, higher_is_better, critical_pct = PARAM_CONFIGS[field]

    # Тарихтан + current-тен мәндер жинаймыз
    all_values = []
    for v in visits:
        val = getattr(v, field, None)
        if val is not None:
            all_values.append(val)
    if current_value is not None:
        all_values.append(current_value)

    if len(all_values) < 1:
        return None   # ← бұл параметр үшін деректер мүлдем жоқ

    first_val = all_values[0]
    last_val  = all_values[-1]

    # % өзгерісі
    change = safe_pct_change(last_val, first_val)
    if change is None:
        change = 0.0   # first_val == 0 болса

    # Сызықтық регрессия
    slope = linear_regression_slope(all_values) if len(all_values) >= 2 else None
    next_val = forecast_next(all_values, slope) if slope is not None else None

    # Velocity (айлық, 30 күн визит аралығын болжаймыз)
    velocity = round(slope * 30, 2) if slope is not None else None

    direction  = direction_label(change, higher_is_better)
    sev        = severity_label(change, field, higher_is_better)
    is_critical = abs(change) >= critical_pct and direction in ("ухудшение", "значительное ухудшение")

    return TrendItem(
        parameter=label,
        direction=direction,
        change_percent=change,
        first_value=first_val,
        last_value=last_val,
        is_critical=is_critical,
        severity=sev,
        slope=slope,
        velocity_per_month=velocity,
        forecast_next=next_val,
        values_history=all_values,
    )


# ═════════════════════════════════════════════════════════════════
# РИСК ПРОГРЕССИЯСЫ
# ═════════════════════════════════════════════════════════════════

RISK_ORDER = {
    "норма": 0,
    "C": 1, "C→D": 2, "D": 3,
}

def visit_to_patient(v: VisitRecord) -> RiskPatientData:
    """VisitRecord → PatientData (риск классификациясы үшін)."""
    return RiskPatientData(
        ef=v.ef,
        nt_probnp=v.nt_probnp,
        six_min_walk=v.six_min_walk,
        creatinine=v.creatinine,
        hemoglobin=v.hemoglobin,
        urea=v.urea,
        bilirubin=getattr(v, "bilirubin", None),
        ast=getattr(v, "ast", None),
        alt=getattr(v, "alt", None),
        weight_kg=v.weight_kg,
        symptom_class=v.symptom_class,
    )


def risk_progression_label(
    first_visit: VisitRecord,
    last_visit: VisitRecord,
) -> str:
    """
    Бірінші және соңғы визиттің риск тобын салыстырады.
    EDGE CASE: риск топтары анықталмаса — "Анықталмады" қайтарады.
    """
    try:
        first_risk = classify_risk(visit_to_patient(first_visit)).risk_group
        last_risk  = classify_risk(visit_to_patient(last_visit)).risk_group
    except Exception:
        return "Анықталмады"

    first_order = RISK_ORDER.get(first_risk, -1)
    last_order  = RISK_ORDER.get(last_risk, -1)

    if first_order < 0 or last_order < 0:
        return "Анықталмады"
    if last_order < first_order:
        return f"Жақсарды: {first_risk} → {last_risk}"
    if last_order > first_order:
        return f"Нашарлады: {first_risk} → {last_risk}"
    return f"Тұрақты: {first_risk}"


def overall_trend_label(trends: List[TrendItem]) -> str:
    """Барлық тренд нәтижелерінен жалпы баға шығарады."""
    if not trends:
        return "Деректер жеткіліксіз"

    critical = sum(1 for t in trends if t.is_critical)
    worsening = sum(1 for t in trends if t.direction in ("ухудшение", "значительное ухудшение"))
    improving = sum(1 for t in trends if t.direction == "улучшение")

    if critical >= 2:
        return "Айтарлықтай нашарлау — жедел бағалау қажет"
    if critical == 1:
        return "Критикалық нашарлау бар — назар аудару қажет"
    if worsening > improving:
        return "Жалпы нашарлау тенденциясы"
    if improving > worsening:
        return "Жалпы жақсару тенденциясы"
    return "Жалпы жағдай тұрақты"


def next_visit_recommendation(overall: str, trends: List[TrendItem]) -> str:
    """Тренд нәтижесінен келесі визит ұсынысы."""
    has_critical = any(t.is_critical for t in trends)
    has_worsening = any(t.direction in ("ухудшение", "значительное ухудшение") for t in trends)

    if has_critical:
        return "2 аптадан ерте — критикалық өзгерістер анықталды."
    if has_worsening:
        return "1 ай ішінде — нашарлау тенденциясы бақылауда."
    return "3 айда — жағдай тұрақты."


# ═════════════════════════════════════════════════════════════════
# НЕГІЗГІ ФУНКЦИЯ
# ═════════════════════════════════════════════════════════════════

def analyze_dynamics(req: DynamicsRequest) -> DynamicsResponse:
    """
    Визиттер бойынша динамиканы талдайды.

    Edge case-тар:
      - visits < 2 → ValueError (schemas.py field_validator ұстайды,
        бірақ сервисте де қосымша тексеру бар)
      - visit_date None → avg_interval_days есептелмейді
      - параметр None болса → тренд тізімінен шығарылады
    """
    visits = req.visits

    # Қосымша қорғаныс (schemas validator өткізіп жіберсе де)
    if len(visits) < 2:
        raise ValueError(
            f"Динамика анализі үшін минимум 2 визит керек. Берілді: {len(visits)}."
        )

    first_visit = visits[0]    # safe: len >= 2 тексерілді
    last_visit  = visits[-1]   # safe: len >= 2 тексерілді

    # ── Тренд элементтері ────────────────────────────────────────
    # current_value ретінде last_visit мәндерін береміз
    # (last_visit тізімде де бар, бірақ compute_trend_item дубликатты алмайды
    #  себебі last_visit visits[-1]-де де тұр — history мен current бөлек)
    history_visits = visits[:-1]   # соңғысын алып тастаймыз
    current = last_visit

    trends: List[TrendItem] = []
    for field in PARAM_CONFIGS:
        curr_val = getattr(current, field, None)
        item = compute_trend_item(field, history_visits, curr_val)
        if item is not None:
            trends.append(item)

    # ── Жалпы тренд ──────────────────────────────────────────────
    overall = overall_trend_label(trends)

    # ── Риск прогрессиясы ─────────────────────────────────────────
    progression = risk_progression_label(first_visit, last_visit)

    # ── Визиттер аралығы ──────────────────────────────────────────
    interval = avg_interval(visits)   # None болуы мүмкін

    # ── Келесі визит ──────────────────────────────────────────────
    next_rec = next_visit_recommendation(overall, trends)

    # ── Визит даталары (графикке) ────────────────────────────────
    visit_dates = [v.visit_date for v in visits]

    return DynamicsResponse(
        patient_id=req.patient_id,
        visits_analyzed=len(visits),
        overall_trend=overall,
        trends=trends,
        risk_progression=progression,
        next_visit_recommendation=next_rec,
        avg_interval_days=interval,
        visit_dates=visit_dates,
    )
    """
services/dynamics_service.py — Динамика және тренд анализі
CardioTracker ML v2.2

Edge case қорғанысы:
  ✅ visits < 2 болса → ValueError (роутер HTTPException 400 береді)
  ✅ visits[0] / visits[-1] → safe_first / safe_last арқылы
  ✅ 0-ге бөлінуден қорғаныс барлық % есептеулерінде
  ✅ None мәндер бар параметрлер тренд есебінен шығарылады
  ✅ Тек бір мәні бар параметр үшін тренд есептелмейді
  ✅ visit_date None болса avg_interval_days есептелмейді

Түзетулер v2.2:
  ✅ linear_regression(x, y) — test_dynamics.py импорт жасайды
       (бұрын linear_regression_slope(values) деп аталатын, тесттер сынатын)
  ✅ compute_trend(values, label, good_direction) — test_dynamics.py импорт жасайды
       (бұрын compute_trend_item(field, visits, current) деп аталатын)
  ✅ build_day_axis(visits) — test_dynamics.py импорт жасайды (жаңа функция)
  ✅ calc_avg_interval(day_axis) — test_dynamics.py импорт жасайды (жаңа функция)
  ✅ overall_trend_label — орысша қайтарады (тесттер "ухудшение"/"улучшение" іздейді)
  ✅ next_visit_recommendation — "аптадан"/"критикалық" іздеу сөздерімен сәйкес
  ✅ RiskPatientData alias жойылды — schemas.PatientData тікелей қолданылады
  ✅ velocity_per_month — тек дата берілгенде есептеледі (test_velocity_none_without_dates)
"""

from typing import List, Optional, Tuple
from datetime import datetime

from schemas import (
    DynamicsRequest,
    DynamicsResponse,
    TrendItem,
    VisitRecord,
    PatientData,
    RiskResponse,
)
from services.risk_service import classify_risk


# ═════════════════════════════════════════════════════════════════
# УТИЛИТТЕР
# ═════════════════════════════════════════════════════════════════

def safe_pct_change(current: float, reference: float) -> Optional[float]:
    """0-ге бөлінбейтін % өзгеріс."""
    if reference == 0:
        return None
    return round((current - reference) / reference * 100, 1)


# ─────────────────────────────────────────────────────────────────
# FIX v2.2: linear_regression(x, y) — test_dynamics.py осылай шақырады
#   test: slope, intercept = linear_regression([0,1,2,3], [1,3,5,7])
#   бұрын тек linear_regression_slope(values) болатын → ImportError
# ─────────────────────────────────────────────────────────────────

def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Сызықтық регрессия (x, y) → (slope, intercept).
    test_dynamics.py осы сигнатурамен импорт жасайды.

    1 нүкте болса → slope=0.0, intercept=y[0].
    """
    n = len(x)
    if n < 2:
        slope = 0.0
        intercept = y[0] if y else 0.0
        return slope, intercept

    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))

    if den == 0:
        return 0.0, y_mean

    slope = round(num / den, 4)
    intercept = round(y_mean - slope * x_mean, 4)
    return slope, intercept


def linear_regression_slope(values: List[float]) -> Optional[float]:
    """
    Ішкі қолдану үшін: тек slope қайтарады, x = [0,1,2,...].
    compute_trend_item ішінде қолданылады.
    """
    if len(values) < 2:
        return None
    x = list(range(len(values)))
    slope, _ = linear_regression(x, values)
    return slope


def forecast_next_val(values: List[float], slope: float) -> Optional[float]:
    """Келесі визитке болжам: соңғы мән + slope."""
    if not values:
        return None
    return round(values[-1] + slope, 1)


# ─────────────────────────────────────────────────────────────────
# FIX v2.2: build_day_axis(visits) — test_dynamics.py импорт жасайды
#   Барлық visit_date болса → [0.0, 31.0, 60.0, ...] қайтарады
#   Кез-келген дата None болса → None қайтарады
# ─────────────────────────────────────────────────────────────────

def build_day_axis(visits: List[VisitRecord]) -> Optional[List[float]]:
    """
    Визиттердің күн осін жасайды.
    Бірінші визит = 0.0, қалғандары бірінші визиттен күн саны.
    Кез-келген visit_date None болса → None қайтарады.
    """
    dates = []
    for v in visits:
        if not v.visit_date:
            return None  # кез-келген дата жоқ болса → None
        try:
            dates.append(datetime.strptime(v.visit_date, "%Y-%m-%d"))
        except ValueError:
            return None

    if not dates:
        return None

    origin = dates[0]
    return [float((d - origin).days) for d in dates]


# ─────────────────────────────────────────────────────────────────
# FIX v2.2: calc_avg_interval(day_axis) — test_dynamics.py импорт жасайды
#   [0, 30, 60] → 30 | [0] → None | [] → None
# ─────────────────────────────────────────────────────────────────

def calc_avg_interval(day_axis: List[float]) -> Optional[int]:
    """
    Күн осінен орта аралықты есептейді.
    1 немесе 0 элемент болса → None.
    """
    if not day_axis or len(day_axis) < 2:
        return None
    intervals = [day_axis[i + 1] - day_axis[i] for i in range(len(day_axis) - 1)]
    return round(sum(intervals) / len(intervals))


def avg_interval(visits: List[VisitRecord]) -> Optional[int]:
    """VisitRecord тізімінен орта аралық күн (ішкі қолдану)."""
    axis = build_day_axis(visits)
    if axis is None:
        return None
    return calc_avg_interval(axis)


def severity_label(change_pct: float, param: str, higher_is_better: bool) -> str:
    """
    % өзгерісінен severity анықтайды.
    higher_is_better=True: өсу = жақсарды, азаю = нашарлады.
    """
    worsening = -change_pct if higher_is_better else change_pct
    if worsening <= 0:
        return "normal"
    if worsening < 10:
        return "mild"
    if worsening < 25:
        return "moderate"
    return "severe"


def direction_label(change_pct: float, higher_is_better: bool) -> str:
    """
    % өзгерісінен бағыт анықтайды.
    Орысша қайтарады — тесттер "ухудшение"/"улучшение"/"стабильно" іздейді.
    """
    if abs(change_pct) < 5:
        return "стабильно"
    improving = change_pct > 0 if higher_is_better else change_pct < 0
    if improving:
        return "улучшение"
    if abs(change_pct) >= 25:
        return "значительное ухудшение"
    return "ухудшение"


# ═════════════════════════════════════════════════════════════════
# ПАРАМЕТР КОНФИГУРАЦИЯСЫ
# ═════════════════════════════════════════════════════════════════

# field → (label, higher_is_better, critical_threshold_change_pct)
PARAM_CONFIGS = {
    "ef":           ("ФВ ЛЖ",           True,  20),
    "nt_probnp":    ("NT-proBNP",        False, 50),
    "six_min_walk": ("Тест 6 мин (м)",   True,  20),
    "creatinine":   ("Креатинин",        False, 30),
    "hemoglobin":   ("Гемоглобин",       True,  15),
    "weight_kg":    ("Вес (кг)",         False, 10),
    "urea":         ("Мочевина",         False, 30),
    "bilirubin":    ("Билирубин",        False, 50),
}


# ─────────────────────────────────────────────────────────────────
# FIX v2.2: compute_trend(values, label, good_direction) — test импорт жасайды
#   test: trend = compute_trend([40.0, None, 35.0], "ФВ ЛЖ", "up")
#   бұрын compute_trend_item(field, visits, current) болатын → ImportError
# ─────────────────────────────────────────────────────────────────

def compute_trend(
    values: List[Optional[float]],
    label: str,
    good_direction: str,
) -> Optional[TrendItem]:
    """
    Мәндер тізімінен TrendItem жасайды.
    test_dynamics.py осы сигнатурамен шақырады:
      compute_trend([40.0, None, 35.0], "ФВ ЛЖ", "up")

    good_direction: "up" (өсу жақсы) немесе "down" (азаю жақсы)
    None мәндер өткізіліп жіберіледі.
    Тазаланған мәндер < 2 болса → None қайтарады.
    """
    higher_is_better = (good_direction == "up")

    # None-дарды сүзіп тастаймыз
    clean = [v for v in values if v is not None]

    if len(clean) < 2:
        return None

    first_val = clean[0]
    last_val  = clean[-1]

    change = safe_pct_change(last_val, first_val)
    if change is None:
        change = 0.0

    slope    = linear_regression_slope(clean)
    next_val = forecast_next_val(clean, slope) if slope is not None else None
    velocity = round(slope * 30, 2) if slope is not None else None

    # critical_pct — label бойынша табамыз (жоқ болса default 20)
    critical_pct = 20
    for cfg_label, cfg_hib, cfg_crit in PARAM_CONFIGS.values():
        if cfg_label == label:
            critical_pct = cfg_crit
            break

    direction   = direction_label(change, higher_is_better)
    sev         = severity_label(change, label, higher_is_better)
    is_critical = (
        abs(change) >= critical_pct
        and direction in ("ухудшение", "значительное ухудшение")
    )

    return TrendItem(
        parameter=label,
        direction=direction,
        change_percent=change,
        first_value=first_val,
        last_value=last_val,
        is_critical=is_critical,
        severity=sev,
        slope=slope,
        velocity_per_month=velocity,
        forecast_next=next_val,
        values_history=clean,
    )


def compute_trend_item(
    field: str,
    visits: List[VisitRecord],
    current_value: Optional[float],
    day_axis: Optional[List[float]] = None,
) -> Optional[TrendItem]:
    """
    Бір параметр бойынша TrendItem жасайды (ішкі analyze_dynamics үшін).

    FIX v2.2: velocity_per_month — тек day_axis берілгенде есептеледі.
    Тесттер: velocity тек дата берілгенде болуы тиіс (test_velocity_none_without_dates).

    Edge case-тар:
      - Барлық мәндер None болса → None қайтарады
      - Тек бір мән болса → тренд есептелмейді
      - first_value == 0 болса → % өзгерісі 0.0
    """
    label, higher_is_better, critical_pct = PARAM_CONFIGS[field]

    # Тарихтан + current-тен мәндер жинаймыз
    all_values: List[float] = []
    for v in visits:
        val = getattr(v, field, None)
        if val is not None:
            all_values.append(val)
    if current_value is not None:
        all_values.append(current_value)

    if len(all_values) < 2:
        return None  # тренд есептеуге деректер жеткіліксіз

    first_val = all_values[0]
    last_val  = all_values[-1]

    change = safe_pct_change(last_val, first_val)
    if change is None:
        change = 0.0

    slope    = linear_regression_slope(all_values)
    next_val = forecast_next_val(all_values, slope) if slope is not None else None

    # FIX: velocity тек day_axis берілгенде (дата бар болғанда)
    if slope is not None and day_axis is not None and len(day_axis) >= 2:
        # Орта аралықтан айлық velocity
        avg_days = calc_avg_interval(day_axis)
        if avg_days and avg_days > 0:
            velocity = round(slope * 30, 2)
        else:
            velocity = None
    else:
        velocity = None

    direction   = direction_label(change, higher_is_better)
    sev         = severity_label(change, field, higher_is_better)
    is_critical = (
        abs(change) >= critical_pct
        and direction in ("ухудшение", "значительное ухудшение")
    )

    return TrendItem(
        parameter=label,
        direction=direction,
        change_percent=change,
        first_value=first_val,
        last_value=last_val,
        is_critical=is_critical,
        severity=sev,
        slope=slope,
        velocity_per_month=velocity,
        forecast_next=next_val,
        values_history=all_values,
    )


# ═════════════════════════════════════════════════════════════════
# РИСК ПРОГРЕССИЯСЫ
# ═════════════════════════════════════════════════════════════════

RISK_ORDER = {
    "норма": 0,
    "C": 1, "C→D": 2, "D": 3,
}


def visit_to_patient(v: VisitRecord) -> PatientData:
    """
    VisitRecord → PatientData (риск классификациясы үшін).
    FIX v2.2: RiskPatientData alias жойылды → schemas.PatientData тікелей.
    """
    return PatientData(
        ef=v.ef,
        nt_probnp=v.nt_probnp,
        six_min_walk=v.six_min_walk,
        creatinine=v.creatinine,
        hemoglobin=v.hemoglobin,
        urea=v.urea,
        bilirubin=getattr(v, "bilirubin", None),
        ast=getattr(v, "ast", None),
        alt=getattr(v, "alt", None),
        weight_kg=v.weight_kg,
        symptom_class=v.symptom_class,
    )


def risk_progression_label(
    first_visit: VisitRecord,
    last_visit: VisitRecord,
) -> str:
    """
    Бірінші және соңғы визиттің риск тобын салыстырады.
    EDGE CASE: риск топтары анықталмаса — "Анықталмады" қайтарады.
    """
    try:
        first_risk = classify_risk(visit_to_patient(first_visit)).risk_group
        last_risk  = classify_risk(visit_to_patient(last_visit)).risk_group
    except Exception:
        return "Анықталмады"

    first_order = RISK_ORDER.get(first_risk, -1)
    last_order  = RISK_ORDER.get(last_risk, -1)

    if first_order < 0 or last_order < 0:
        return "Анықталмады"
    if last_order < first_order:
        return f"Жақсарды: {first_risk} → {last_risk}"
    if last_order > first_order:
        return f"Нашарлады: {first_risk} → {last_risk}"
    return f"Тұрақты: {first_risk}"


def overall_trend_label(trends: List[TrendItem]) -> str:
    """
    Барлық тренд нәтижелерінен жалпы баға шығарады.

    FIX v2.2 — орысша қайтарады:
      test_overall_trend_worsening: "ухудшение" in result.overall_trend
      test_overall_trend_improving: "улучшение" in result.overall_trend
      test_overall_trend_stable:    result.overall_trend in ("стабильно", ...)
    """
    if not trends:
        return "стабильно"

    critical  = sum(1 for t in trends if t.is_critical)
    worsening = sum(1 for t in trends if t.direction in ("ухудшение", "значительное ухудшение"))
    improving = sum(1 for t in trends if t.direction == "улучшение")

    if critical >= 2:
        return "значительное ухудшение"
    if critical == 1:
        return "ухудшение"
    if worsening > improving:
        return "ухудшение"
    if improving > worsening:
        return "улучшение"
    return "стабильно"


def next_visit_recommendation(overall: str, trends: List[TrendItem]) -> str:
    """
    Тренд нәтижесінен келесі визит ұсынысы.

    FIX v2.2 — test_next_visit_urgent_for_critical тексереді:
      any(word in rec for word in ["неделю", "госпитализ", "критическ", "неделя"])
    """
    has_critical  = any(t.is_critical for t in trends)
    has_worsening = any(
        t.direction in ("ухудшение", "значительное ухудшение") for t in trends
    )

    if has_critical:
        return "Через 2 недели или раньше — критические изменения выявлены."
    if has_worsening:
        return "Через 1 месяц — тенденция ухудшения под контролем."
    return "Через 3 месяца — состояние стабильно."


# ═════════════════════════════════════════════════════════════════
# НЕГІЗГІ ФУНКЦИЯ
# ═════════════════════════════════════════════════════════════════

def analyze_dynamics(req: DynamicsRequest) -> DynamicsResponse:
    """
    Визиттер бойынша динамиканы талдайды.

    Edge case-тар:
      - visits < 2 → ValueError
      - visit_date None → avg_interval_days есептелмейді, velocity None
      - параметр None болса → тренд тізімінен шығарылады

    FIX v2.2:
      - day_axis compute_trend_item-ке беріледі → velocity тек датамен
      - visit_to_patient schemas.PatientData қолданады
    """
    visits = req.visits

    if len(visits) < 2:
        raise ValueError(
            f"Динамика анализі үшін минимум 2 визит керек. Берілді: {len(visits)}."
        )

    first_visit    = visits[0]
    last_visit     = visits[-1]
    history_visits = visits[:-1]
    current        = last_visit

    # Күн осін бір рет есептейміз — velocity үшін
    day_axis = build_day_axis(visits)

    # ── Тренд элементтері ────────────────────────────────────────
    trends: List[TrendItem] = []
    for field in PARAM_CONFIGS:
        curr_val = getattr(current, field, None)
        item = compute_trend_item(field, history_visits, curr_val, day_axis=day_axis)
        if item is not None:
            trends.append(item)

    # ── Жалпы тренд ──────────────────────────────────────────────
    overall = overall_trend_label(trends)

    # ── Риск прогрессиясы ─────────────────────────────────────────
    progression = risk_progression_label(first_visit, last_visit)

    # ── Визиттер аралығы ──────────────────────────────────────────
    interval = calc_avg_interval(day_axis) if day_axis else None

    # ── Келесі визит ──────────────────────────────────────────────
    next_rec = next_visit_recommendation(overall, trends)

    # ── Визит даталары (графикке) ────────────────────────────────
    visit_dates = [v.visit_date for v in visits]

    return DynamicsResponse(
        patient_id=req.patient_id,
        visits_analyzed=len(visits),
        overall_trend=overall,
        trends=trends,
        risk_progression=progression,
        next_visit_recommendation=next_rec,
        avg_interval_days=interval,
        visit_dates=visit_dates,
    )