"""
routers/training_data.py — Training Data жинау эндпоинті
CardioTracker ML v2.2

ТЗ негізі (§3.5 + §4):
  "Барлық дәрігер тағайындаулары training_data кестесіне сақталады.
   Бұл v3.0 ML моделінің негізі."

  Кесте өрістері:
    id, visit_id, prescription_id, input_features (jsonb),
    doctor_decision (jsonb), doctor_id, created_at

Архитектура:
  Spring Boot → дәрігер тағайындауды prescriptions кестесіне сақтайды
             → содан POST /ml/save-training-data шақырады
             → FastAPI input_features + doctor_decision валидациялайды
             → JSON файлға сақтайды (v1.0 прототип)
             → v3.0-да PostgreSQL-ге жазылады

v1.0 шектеуі:
  Нурсултан VPS пен PostgreSQL орнатқанша деректер JSON файлда сақталады.
  Бұл дұрыс — main мақсат деректерді ЖОҒАЛТПАУ, кесте кейін жасалады.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  СХЕМАЛАР
# ─────────────────────────────────────────────────────────────────────────────

class InputFeatures(BaseModel):
    """
    Визиттің 8 клиникалық параметрі — ТЗ §3.3 бойынша.
    Бұл болашақта ML моделінің X (feature vector) болады.
    """
    # Негізгі 8 параметр (ТЗ §3.3)
    ef_lv: Optional[float]        = Field(default=None, description="ФВ/ЛЖ, %")
    six_min_walk: Optional[float] = Field(default=None, description="Тест 6 мин, м")
    nt_probnp: Optional[float]    = Field(default=None, description="NT-proBNP, пг/мл")
    hemoglobin: Optional[float]   = Field(default=None, description="Гемоглобин, г/л")
    creatinine: Optional[float]   = Field(default=None, description="Креатинин, мкмоль/л")
    urea: Optional[float]         = Field(default=None, description="Мочевина, ммоль/л")
    alt: Optional[float]          = Field(default=None, description="АЛТ, Ед/л")
    ast: Optional[float]          = Field(default=None, description="АСТ, Ед/л")

    # ЭКГ флагтары
    ecg_af: bool          = Field(default=False)
    ecg_tachycardia: bool = Field(default=False)
    ecg_blockade: bool    = Field(default=False)
    ecg_st_changes: bool  = Field(default=False)

    # Антропометрия
    height_cm: Optional[float] = Field(default=None)
    weight_kg: Optional[float] = Field(default=None)
    bmi: Optional[float]       = Field(default=None, description="Автоесептелген BMI")

    # ML нәтижесі (FastAPI есептеген)
    risk_group: Optional[str]  = Field(default=None, description="Норма A / B / C / C→D / D")
    risk_score: Optional[float]= Field(default=None, description="Балл 0.0–1.0")
    active_alerts: List[str]   = Field(default_factory=list, description="Белсенді алерт кодтары")

    # Пациент контексті
    age: Optional[int]    = Field(default=None)
    sex: Optional[str]    = Field(default=None)
    nyha_class: Optional[int] = Field(default=None, description="NYHA ФК 1–4")

    # Коморбидность
    has_diabetes: bool     = Field(default=False)
    has_hypertension: bool = Field(default=False)
    has_prior_mi: bool     = Field(default=False)
    has_ckd: bool          = Field(default=False)
    has_copd: bool         = Field(default=False)
    has_afib_history: bool = Field(default=False)


class DoctorDecision(BaseModel):
    """
    Дәрігердің шешімі — болашақта ML моделінің Y (target/label) болады.

    ТЗ §3.5: "Диагноз + назначенные препараты + рекомендации"
    ТЗ: "Все назначения врача сохраняются как обучающие данные"
    """
    # Негізгі шешім
    diagnosis: str = Field(
        ...,
        description="Клиникалық қорытынды / диагноз",
        json_schema_extra={"example": "ХСН ФК III, декомпенсация. Антикоагулянттар тиімсіз."}
    )
    medications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Тағайындалған препараттар: [{name, dose, frequency}]",
        json_schema_extra={"example": [
            {"name": "Бисопролол", "dose": "5мг", "frequency": "күніне 1 рет"},
            {"name": "Фуросемид",  "dose": "40мг", "frequency": "таңертең"},
        ]}
    )
    recommendations: Optional[str] = Field(
        default=None,
        description="Режим, диета, белсенділік шектеулері"
    )
    next_visit_days: Optional[int] = Field(
        default=None,
        description="Келесі визитке дейін күн саны",
        json_schema_extra={"example": 14}
    )

    # v3.0-да маңызды — неліктен өзгертті?
    change_reason: Optional[str] = Field(
        default=None,
        description=(
            "Шешім себебі (v3.0 үшін алтын деректер): "
            "dose_ineffective / side_effect / lab_deterioration / "
            "improvement / routine"
        ),
        json_schema_extra={"example": "lab_deterioration"}
    )


class TrainingDataRequest(BaseModel):
    """
    Spring Boot жіберетін сұраныс.
    Дәрігер тағайындауды сақтаған кезде бір рет шақырылады.
    """
    # Spring Boot ID-лары
    visit_id: int = Field(
        ...,
        description="Spring Boot visits кестесіндегі ID",
        json_schema_extra={"example": 142}
    )
    prescription_id: int = Field(
        ...,
        description="Spring Boot prescriptions кестесіндегі ID",
        json_schema_extra={"example": 87}
    )
    doctor_id: int = Field(
        ...,
        description="Дәрігер ID (users кестесі)",
        json_schema_extra={"example": 2}
    )
    patient_id: int = Field(
        ...,
        description="Пациент ID (patients кестесі)",
        json_schema_extra={"example": 34}
    )

    # ML деректері
    input_features: InputFeatures = Field(
        ...,
        description="Визит кезіндегі 8 клиникалық параметр + ML нәтижесі"
    )
    doctor_decision: DoctorDecision = Field(
        ...,
        description="Дәрігердің диагнозы + препараттары + ұсынымдары"
    )

    # Алмас дәрігері ме? (gold label үшін)
    is_lead_cardiologist: bool = Field(
        default=False,
        description="True болса — Алмастың шешімі (gold label v3.0 үшін)"
    )


class TrainingDataResponse(BaseModel):
    """FastAPI жауабы Spring Boot-қа."""
    success: bool
    record_id: str          = Field(description="Сақталған жазбаның ID-і")
    visit_id: int
    patient_id: int
    label_type: str         = Field(description="gold_label / secondary_label")
    total_records: int      = Field(description="Барлығы жиналған жазбалар саны")
    message: str


# ─────────────────────────────────────────────────────────────────────────────
#  ДЕРЕКТЕРДІ САҚТАУ (v1.0 — JSON файл)
# ─────────────────────────────────────────────────────────────────────────────

# JSON файл жолы — cardio-ml/ папкасында
_BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR      = os.path.join(_BASE_DIR, "training_data")
_DATA_FILE     = os.path.join(_DATA_DIR, "training_data.jsonl")
_STATS_FILE    = os.path.join(_DATA_DIR, "stats.json")


def _ensure_data_dir():
    """training_data/ папкасы жоқ болса — жасайды."""
    os.makedirs(_DATA_DIR, exist_ok=True)


def _save_record(record: dict) -> int:
    """
    JSONL форматына қосады (бір жол = бір жазба).
    JSONL — ML пайдалану үшін ең ыңғайлы формат:
      каждая строка = один training example
      pd.read_json('file.jsonl', lines=True) → DataFrame
    """
    _ensure_data_dir()
    with open(_DATA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _count_records() -> int:
    """Барлық жазбалар санын есептейді."""
    if not os.path.exists(_DATA_FILE):
        return 0
    with open(_DATA_FILE, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _update_stats(record: dict):
    """Статистика файлын жаңартады."""
    _ensure_data_dir()

    stats = {}
    if os.path.exists(_STATS_FILE):
        with open(_STATS_FILE, "r", encoding="utf-8") as f:
            stats = json.load(f)

    stats["total_records"]  = stats.get("total_records", 0) + 1
    stats["last_updated"]   = datetime.utcnow().isoformat()
    stats["gold_labels"]    = stats.get("gold_labels", 0) + (
        1 if record.get("label_type") == "gold_label" else 0
    )
    stats["secondary_labels"] = stats.get("secondary_labels", 0) + (
        1 if record.get("label_type") == "secondary_label" else 0
    )

    # Риск топтары бойынша тарату
    rg = record.get("input_features", {}).get("risk_group", "unknown")
    dist = stats.setdefault("risk_distribution", {})
    dist[rg] = dist.get(rg, 0) + 1

    with open(_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINT: POST /ml/save-training-data
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/save-training-data",
    response_model=TrainingDataResponse,
    summary="Сохранение обучающих данных v2.2",
    description=(
        "Сохраняет пару (визит → решение врача) в базу обучающих данных.\n\n"
        "**Когда вызывается:** Spring Boot вызывает этот endpoint сразу после "
        "сохранения назначения врача в таблицу `prescriptions`.\n\n"
        "**Что сохраняется:**\n"
        "- `input_features` — 8 клинических параметров визита + результат ML\n"
        "- `doctor_decision` — диагноз + препараты + рекомендации\n"
        "- `is_lead_cardiologist` — если True, запись помечается как "
        "`gold_label` (решения Алмаса для обучения v3.0)\n\n"
        "**v1.0:** данные сохраняются в JSONL файл "
        "(`training_data/training_data.jsonl`). "
        "При наличии PostgreSQL — мигрируются одной командой.\n\n"
        "**v3.0:** эти данные станут датасетом для обучения XGBoost модели."
    ),
    tags=["Training Data"],
)
async def save_training_data(req: TrainingDataRequest) -> TrainingDataResponse:
    try:
        # Метадеректер
        label_type = "gold_label" if req.is_lead_cardiologist else "secondary_label"
        record_id  = f"td_{req.visit_id}_{req.prescription_id}_{int(datetime.utcnow().timestamp())}"
        created_at = datetime.utcnow().isoformat()

        # Сақтау объектісі (ТЗ §4 training_data кестесімен сәйкес)
        record = {
            "record_id":             record_id,
            "visit_id":              req.visit_id,
            "prescription_id":       req.prescription_id,
            "patient_id":            req.patient_id,
            "doctor_id":             req.doctor_id,
            "label_type":            label_type,
            "is_lead_cardiologist":  req.is_lead_cardiologist,
            "input_features":        req.input_features.model_dump(),
            "doctor_decision":       req.doctor_decision.model_dump(),
            "created_at":            created_at,
            "model_version":         "rule_v2.2",  # кейін ML v3.0-да өзгереді
        }

        # JSONL-ға жазу
        _save_record(record)

        # Статистиканы жаңарту
        stats = _update_stats(record)
        total = stats.get("total_records", 1)

        return TrainingDataResponse(
            success=True,
            record_id=record_id,
            visit_id=req.visit_id,
            patient_id=req.patient_id,
            label_type=label_type,
            total_records=total,
            message=(
                f"Сақталды: {label_type}. "
                f"Барлығы {total} жазба жиналды. "
                f"v3.0 моделі үшін мақсат: 3000+ жазба."
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training data сақтау қатесі: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINT: GET /ml/training-data/stats
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/training-data/stats",
    summary="Статистика обучающих данных",
    description=(
        "Показывает сколько обучающих примеров накоплено.\n\n"
        "**Целевые показатели:**\n"
        "- 500 записей → достаточно для baseline XGBoost\n"
        "- 1000+ записей → стабильная модель\n"
        "- 3000+ записей → надёжная v3.0 модель\n\n"
        "**Gold labels** = решения Алмаса (главного кардиолога)\n"
        "**Secondary labels** = решения других врачей"
    ),
    tags=["Training Data"],
)
async def training_data_stats():
    try:
        _ensure_data_dir()

        if not os.path.exists(_STATS_FILE):
            return {
                "total_records":      0,
                "gold_labels":        0,
                "secondary_labels":   0,
                "risk_distribution":  {},
                "target_for_v3":      3000,
                "progress_pct":       0,
                "message":            "Деректер жоқ. Дәрігер тағайындаулар сақтала бастаса — жазбалар пайда болады.",
                "last_updated":       None,
            }

        with open(_STATS_FILE, "r", encoding="utf-8") as f:
            stats = json.load(f)

        total    = stats.get("total_records", 0)
        target   = 3000
        progress = round(total / target * 100, 1)

        stats["target_for_v3"] = target
        stats["progress_pct"]  = progress
        stats["message"] = (
            f"{total} жазба жиналды ({progress}% мақсатқа). "
            f"v3.0 моделі үшін {target - total} жазба жетіспейді."
            if total < target else
            f"✅ {total} жазба — v3.0 моделін оқытуға болады!"
        )

        return stats

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Статистика оқу қатесі: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINT: GET /ml/training-data/export
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/training-data/export",
    summary="Экспорт обучающих данных (последние N записей)",
    description=(
        "Возвращает последние N записей для проверки качества данных.\n\n"
        "Используется Бекзатом для анализа перед обучением модели v3.0."
    ),
    tags=["Training Data"],
)
async def export_training_data(limit: int = 10):
    try:
        if not os.path.exists(_DATA_FILE):
            return {"records": [], "total": 0}

        records = []
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # Соңғы N жазбаны қайтар
        return {
            "records": records[-limit:],
            "total":   len(records),
            "showing": min(limit, len(records)),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Экспорт қатесі: {str(e)}",
        )