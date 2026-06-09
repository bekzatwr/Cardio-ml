"""
services/ml_validator_service.py — ML Training Data Validator
CardioTracker ML v2.2

Мақсат:
  training_data кестесінде жиналған жазбаларды талдап, v3.0 ML моделіне
  дайындықты бағалайды.

Не жасайды:
  1. JSON жазбаларды Pandas DataFrame-ге айналдырады (flatten)
  2. Corrupted жазбаларды анықтап шығарады
  3. Пропуск статистикасын есептейді
  4. Класс үлестірімін тексереді (дисбаланс ескертуі)
  5. RandomForestClassifier оқытып, Feature Importance шығарады
  6. v3.0 дайындық баллын есептейді

Архитектура:
  - analyze_collected_data() — негізгі функция
  - _flatten_records() — JSON → DataFrame
  - _detect_corrupted() — мусорлық жазбаларды анықтау
  - _compute_missing_stats() — пропуск статистикасы
  - _compute_class_distribution() — класс үлестірімі
  - _train_baseline_model() — RandomForest + Feature Importance
  - _compute_readiness() — v3.0 дайындық баллы
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════
# КОНСТАНТАЛАР
# ═════════════════════════════════════════════════════════════════

# 8 негізгі клиникалық параметр (ТЗ §3.3)
CLINICAL_FEATURES = [
    "ef_lv",
    "nt_probnp",
    "six_min_walk",
    "hemoglobin",
    "creatinine",
    "urea",
    "ast",
    "alt",
]

# Маңызды параметрлер — осылардың бәрі жоқ болса → corrupted
CRITICAL_FEATURES = ["ef_lv", "nt_probnp", "six_min_walk"]

# Дәрігер шешімінен алынатын label
LABEL_COLUMN = "risk_group"

# v3.0 мақсаттары
V3_TARGET_RECORDS   = 3000
V3_MIN_RECORDS      = 100   # базалық модель үшін минимум
V3_GOOD_RECORDS     = 500   # жақсы модель үшін
V3_GREAT_RECORDS    = 1000  # тамаша модель үшін

# Класс дисбаланс шекарасы
MIN_CLASS_PCT = 5.0   # бір топ < 5% болса → ескерту

# Дайындық деңгейлері
READINESS_LABELS = [
    (0,   25,  "недостаточно"),
    (25,  50,  "базовый"),
    (50,  75,  "хороший"),
    (75,  101, "отличный"),
]


# ═════════════════════════════════════════════════════════════════
# 1. ДЕРЕКТЕРДІ FLATTEN ЖАСАУ
# ═════════════════════════════════════════════════════════════════

def _flatten_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    training_data жазбаларын плоский DataFrame-ге айналдырады.

    Кіріс форматы:
      {
        "record_id": "td_142_87_...",
        "visit_id": 142,
        "patient_id": 42,
        "doctor_id": 2,
        "label_type": "gold_label",
        "input_features": {"ef_lv": 38.0, "nt_probnp": 1200.0, ...},
        "doctor_decision": {"diagnosis": "...", "risk_group": "C", ...},
        "created_at": "2024-02-15T09:30:00"
      }

    Шығыс: плоский DataFrame барлық өрістермен.

    Edge cases:
      - input_features / doctor_decision string болса → json.loads()
      - None болса → бос dict
      - Белгісіз өрістер → NaN
    """
    import json

    rows = []
    for rec in records:
        row = {
            "record_id":   rec.get("record_id"),
            "visit_id":    rec.get("visit_id"),
            "patient_id":  rec.get("patient_id"),
            "doctor_id":   rec.get("doctor_id"),
            "label_type":  rec.get("label_type", "secondary_label"),
            "created_at":  rec.get("created_at"),
        }

        # input_features flatten
        inp = rec.get("input_features") or {}
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except Exception:
                inp = {}
        for key, val in inp.items():
            row[f"feat_{key}"] = val

        # doctor_decision flatten
        dec = rec.get("doctor_decision") or {}
        if isinstance(dec, str):
            try:
                dec = json.loads(dec)
            except Exception:
                dec = {}
        row["risk_group"]        = dec.get("risk_group") or inp.get("risk_group")
        row["diagnosis"]         = dec.get("diagnosis")
        row["next_visit_days"]   = dec.get("next_visit_days")
        row["change_reason"]     = dec.get("change_reason")
        row["medications_count"] = len(dec.get("medications") or [])

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Сандық өрістерді дұрыс типке айналдыру
    numeric_cols = [c for c in df.columns if c.startswith("feat_")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ═════════════════════════════════════════════════════════════════
# 2. CORRUPTED ЖАЗБАЛАРДЫ АНЫҚТАУ
# ═════════════════════════════════════════════════════════════════

def _detect_corrupted(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Мусорлық жазбаларды анықтап, екіге бөледі.

    Corrupted деп саналады: CRITICAL_FEATURES барлығы NaN болса
    (ef_lv, nt_probnp, six_min_walk — үшеуі де жоқ болса).

    Бір маңызды параметр болса жеткілікті — жазба жарамды.

    Returns:
      (valid_df, corrupted_df)
    """
    if df.empty:
        return df.copy(), df.copy()

    critical_cols = [f"feat_{f}" for f in CRITICAL_FEATURES if f"feat_{f}" in df.columns]

    if not critical_cols:
        # Бағандар мүлде жоқ — барлығы corrupted
        logger.warning("_detect_corrupted: маңызды бағандар жоқ, барлық жазба corrupted")
        return pd.DataFrame(columns=df.columns), df.copy()

    # Барлық маңызды бағандар NaN болса → corrupted
    all_critical_missing = df[critical_cols].isnull().all(axis=1)

    valid_df     = df[~all_critical_missing].copy()
    corrupted_df = df[all_critical_missing].copy()

    return valid_df, corrupted_df


# ═════════════════════════════════════════════════════════════════
# 3. ПРОПУСК СТАТИСТИКАСЫ
# ═════════════════════════════════════════════════════════════════

def _compute_missing_stats(df: pd.DataFrame) -> List[Dict]:
    """
    Әр клиникалық параметр бойынша пропуск санын есептейді.

    Returns:
      List[{feature, missing_count, missing_pct, is_critical}]
    """
    if df.empty:
        return []

    stats = []
    for feat in CLINICAL_FEATURES:
        col = f"feat_{feat}"
        if col not in df.columns:
            missing_count = len(df)
        else:
            missing_count = int(df[col].isnull().sum())

        missing_pct = round(missing_count / len(df) * 100, 1) if len(df) > 0 else 0.0

        stats.append({
            "feature":       feat,
            "missing_count": missing_count,
            "missing_pct":   missing_pct,
            "is_critical":   feat in CRITICAL_FEATURES,
        })

    return sorted(stats, key=lambda x: (-x["missing_pct"], x["feature"]))


# ═════════════════════════════════════════════════════════════════
# 4. КЛАСС ҮЛЕСТІРІМІ
# ═════════════════════════════════════════════════════════════════

def _compute_class_distribution(
    df: pd.DataFrame,
) -> Tuple[Dict[str, int], Optional[str]]:
    """
    Дәрігер шешімдері бойынша риск топтарының саны мен ескертуі.

    Returns:
      (distribution_dict, warning_message or None)
    """
    if df.empty or LABEL_COLUMN not in df.columns:
        return {}, "Класс таңбалары жоқ — risk_group бағаны табылмады."

    # NaN болса жою
    valid = df[LABEL_COLUMN].dropna()
    if valid.empty:
        return {}, "Барлық risk_group мәндері бос."

    dist = valid.value_counts().to_dict()
    dist = {str(k): int(v) for k, v in dist.items()}

    # Дисбаланс тексеру
    total = sum(dist.values())
    warning = None
    minority_classes = []
    for cls, cnt in dist.items():
        pct = cnt / total * 100
        if pct < MIN_CLASS_PCT:
            minority_classes.append(f"{cls} ({cnt} жазба, {pct:.1f}%)")

    if minority_classes:
        warning = (
            f"Класс дисбалансы: {', '.join(minority_classes)} — "
            f"v3.0 оқытуда SMOTE немесе class_weight='balanced' қажет."
        )

    return dist, warning


# ═════════════════════════════════════════════════════════════════
# 5. RANDOMFOREST + FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════

def _train_baseline_model(
    df: pd.DataFrame,
    top_n: int = 5,
) -> Tuple[List[Dict], Optional[float], Optional[str]]:
    """
    RandomForestClassifier оқытып, Feature Importance шығарады.

    Args:
      df:    жарамды жазбалар DataFrame
      top_n: қанша параметр қайтару керек

    Returns:
      (feature_importance_list, accuracy, warning_message)

    Edge cases:
      - < 10 жазба → модель оқытылмайды
      - NaN мәндер → медиана imputation (ValueError: Input contains NaN болмайды)
      - 1 ғана класс → модель оқытылмайды
      - class_weight='balanced' → дисбаланс әсерін азайту
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    # Параметрлер бағандары
    feature_cols = [f"feat_{f}" for f in CLINICAL_FEATURES if f"feat_{f}" in df.columns]
    label_col    = LABEL_COLUMN

    if len(feature_cols) < 3:
        return [], None, "Параметр бағандары жеткіліксіз — оқыту мүмкін емес."

    if label_col not in df.columns:
        return [], None, "risk_group бағаны жоқ — дәрігер шешімдері жоқ."

    # Label + feature бар жазбаларды ғана алу
    subset = df[feature_cols + [label_col]].dropna(subset=[label_col])

    if len(subset) < 10:
        return [], None, (
            f"Деректер жеткіліксіз: {len(subset)} жазба. "
            f"RandomForest оқыту үшін минимум 10 жазба керек."
        )

    if subset[label_col].nunique() < 2:
        return [], None, (
            f"Тек 1 класс бар ({subset[label_col].unique()[0]}) — "
            f"модель оқыту мүмкін емес. Әртүрлі риск топтары керек."
        )

    # FIX: NaN → медиана imputation (ValueError: Input contains NaN болмайды)
    X = subset[feature_cols].copy()
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    # Label encoding
    le = LabelEncoder()
    y  = le.fit_transform(subset[label_col].astype(str))

    # RandomForest оқыту
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",   # дисбаланс әсерін азайту
        random_state=42,
        n_jobs=-1,
    )

    # Cross-val accuracy (аз деректерде 3-fold)
    n_splits = min(3, len(subset) // 3)
    accuracy = None
    warning  = None

    if n_splits >= 2:
        try:
            scores   = cross_val_score(rf, X, y, cv=n_splits, scoring="accuracy")
            accuracy = round(float(scores.mean()), 3)
        except Exception as exc:
            logger.warning("cross_val_score қате: %s", exc)
            warning = f"Cross-validation орындалмады: {exc}"
    else:
        warning = f"Деректер аз ({len(subset)} жазба) — cross-val жасалмады."

    # Толық деректерде оқыту (feature importance үшін)
    rf.fit(X, y)

    # Feature importance
    importances = rf.feature_importances_
    feat_names  = [col.replace("feat_", "") for col in feature_cols]

    imp_pairs = sorted(
        zip(feat_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    result = [
        {
            "feature":    name,
            "importance": round(float(imp), 4),
            "rank":       i + 1,
        }
        for i, (name, imp) in enumerate(imp_pairs)
    ]

    if len(subset) < 50:
        msg = (
            f"Ескерту: {len(subset)} жазба аз — feature importance "
            f"тұрақсыз болуы мүмкін. 50+ жазбадан кейін сенімді нәтиже шығады."
        )
        warning = (warning + " | " + msg) if warning else msg

    return result, accuracy, warning


# ═════════════════════════════════════════════════════════════════
# 6. V3.0 ДАЙЫНДЫҚ БАЛЛЫ
# ═════════════════════════════════════════════════════════════════

def _compute_readiness(
    valid_count:        int,
    gold_count:         int,
    critical_missing:   float,
    class_dist:         Dict[str, int],
    model_trained:      bool,
) -> Tuple[float, str, int]:
    """
    v3.0 ML моделіне дайындық баллын 0–100 шкаласында есептейді.

    Балл компоненттері:
      40% — жазбалар саны (3000 мақсат)
      20% — gold label үлесі (Алмас шешімдері)
      20% — деректер толықтығы (маңызды параметрлер)
      20% — класс теңдігі

    Returns:
      (score_0_100, readiness_label, records_needed)
    """
    total_valid = max(valid_count, 0)

    # 1. Жазбалар саны (40%)
    count_score = min(total_valid / V3_TARGET_RECORDS, 1.0) * 40

    # 2. Gold label үлесі (20%) — мақсат ≥30%
    gold_pct = (gold_count / total_valid * 100) if total_valid > 0 else 0
    gold_score = min(gold_pct / 30.0, 1.0) * 20

    # 3. Деректер толықтығы (20%) — мақсат <10% пропуск
    completeness = max(0, 100 - critical_missing) / 100
    completeness_score = completeness * 20

    # 4. Класс теңдігі (20%)
    balance_score = 0.0
    if class_dist:
        total_cls = sum(class_dist.values())
        if total_cls > 0:
            min_pct = min(v / total_cls * 100 for v in class_dist.values())
            # Минималды класс ≥5% болса толық балл
            balance_score = min(min_pct / MIN_CLASS_PCT, 1.0) * 20

    total_score    = round(count_score + gold_score + completeness_score + balance_score, 1)
    records_needed = max(0, V3_TARGET_RECORDS - total_valid)

    # Дайындық деңгейі
    label = "недостаточно"
    for lo, hi, lbl in READINESS_LABELS:
        if lo <= total_score < hi:
            label = lbl
            break

    return total_score, label, records_needed


# ═════════════════════════════════════════════════════════════════
# 7. НЕГІЗГІ ФУНКЦИЯ
# ═════════════════════════════════════════════════════════════════

def analyze_collected_data(
    raw_data: List[Dict[str, Any]],
    top_n_features: int = 5,
) -> Dict[str, Any]:
    """
    training_data жазбаларын толық талдайды.

    Args:
      raw_data:       training_data кестесіндегі жазбалар тізімі
      top_n_features: feature importance-та қанша параметр

    Returns:
      DataQualityReport-пен сәйкес dict

    Edge cases:
      - Бос тізім → нөлдік есеп
      - Барлық жазба corrupted → model_trained=False
      - RandomForest NaN → медиана imputation (crash жоқ)
    """
    from datetime import timezone

    now = datetime.now(timezone.utc).isoformat()

    if not raw_data:
        return {
            "total_records":        0,
            "valid_records":        0,
            "corrupted_records":    0,
            "corrupted_pct":        0.0,
            "gold_label_count":     0,
            "gold_label_pct":       0.0,
            "class_distribution":   {},
            "class_balance_warning": "Деректер жоқ.",
            "missing_values":       [],
            "critical_missing_pct": 0.0,
            "feature_importance":   [],
            "model_trained":        False,
            "model_accuracy":       None,
            "model_warning":        "training_data бос.",
            "readiness_score":      0.0,
            "readiness_label":      "недостаточно",
            "records_needed_for_v3": V3_TARGET_RECORDS,
            "generated_at":         now,
        }

    # ── 1. Flatten ────────────────────────────────────────────────
    df = _flatten_records(raw_data)
    total = len(df)

    # ── 2. Corrupted анықтау ──────────────────────────────────────
    valid_df, corrupted_df = _detect_corrupted(df)
    valid_count     = len(valid_df)
    corrupted_count = len(corrupted_df)
    corrupted_pct   = round(corrupted_count / total * 100, 1) if total > 0 else 0.0

    # ── 3. Gold label ─────────────────────────────────────────────
    gold_count = int((df.get("label_type", pd.Series([])) == "gold_label").sum()) \
                 if "label_type" in df.columns else 0
    gold_pct   = round(gold_count / total * 100, 1) if total > 0 else 0.0

    # ── 4. Пропуск статистикасы ───────────────────────────────────
    missing_stats = _compute_missing_stats(valid_df)
    critical_stats = [s for s in missing_stats if s["is_critical"]]
    critical_missing_pct = (
        round(sum(s["missing_pct"] for s in critical_stats) / len(critical_stats), 1)
        if critical_stats else 0.0
    )

    # ── 5. Класс үлестірімі ───────────────────────────────────────
    class_dist, class_warning = _compute_class_distribution(valid_df)

    # ── 6. RandomForest ───────────────────────────────────────────
    feat_importance, accuracy, model_warning = _train_baseline_model(
        valid_df, top_n=top_n_features
    )
    model_trained = len(feat_importance) > 0

    # ── 7. Дайындық баллы ─────────────────────────────────────────
    readiness_score, readiness_label, records_needed = _compute_readiness(
        valid_count      = valid_count,
        gold_count       = gold_count,
        critical_missing = critical_missing_pct,
        class_dist       = class_dist,
        model_trained    = model_trained,
    )

    logger.info(
        "[ml_validator] total=%d valid=%d corrupted=%d gold=%d "
        "readiness=%.1f%% (%s)",
        total, valid_count, corrupted_count, gold_count,
        readiness_score, readiness_label,
    )

    return {
        "total_records":         total,
        "valid_records":         valid_count,
        "corrupted_records":     corrupted_count,
        "corrupted_pct":         corrupted_pct,
        "gold_label_count":      gold_count,
        "gold_label_pct":        gold_pct,
        "class_distribution":    class_dist,
        "class_balance_warning": class_warning,
        "missing_values":        missing_stats,
        "critical_missing_pct":  critical_missing_pct,
        "feature_importance":    feat_importance,
        "model_trained":         model_trained,
        "model_accuracy":        accuracy,
        "model_warning":         model_warning,
        "readiness_score":       readiness_score,
        "readiness_label":       readiness_label,
        "records_needed_for_v3": records_needed,
        "generated_at":          now,
    }