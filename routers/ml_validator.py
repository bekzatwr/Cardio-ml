"""
routers/ml_validator.py — ML Training Data Validator HTTP роутері
CardioTracker ML v2.2

Endpoints:
  POST /ml/validator/analyze  — жазбаларды талдау (Spring жіберіп тексерген кезде)
  GET  /ml/validator/report   — жергілікті JSONL файлынан есеп (демо үшін)
"""

import json
import os
from fastapi import APIRouter, HTTPException

from services.ml_validator_service import analyze_collected_data

router = APIRouter()

_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_FILE = os.path.join(_BASE_DIR, "training_data", "training_data.jsonl")


@router.post(
    "/validator/analyze",
    summary="ML Validator — Training Data анализі",
    description=(
        "Spring Boot жіберген training_data жазбаларын талдайды.\n\n"
        "**Не жасайды:**\n"
        "- JSON жазбаларды Pandas DataFrame-ге айналдырады\n"
        "- Corrupted (мусорлық) жазбаларды анықтайды\n"
        "- Пропуск статистикасын есептейді\n"
        "- Класс дисбалансын тексереді\n"
        "- RandomForest оқытып, Feature Importance шығарады\n"
        "- v3.0 дайындық баллын есептейді (0–100)\n\n"
        "**Минимум:** 10 жарамды жазба болса RandomForest оқытылады."
    ),
    tags=["Training Data"],
)
async def analyze_training_data(payload: dict):
    try:
        records     = payload.get("records", [])
        top_n       = int(payload.get("top_n_features", 5))
        if not isinstance(records, list):
            raise HTTPException(status_code=400, detail="records тізім болуы керек.")
        result = analyze_collected_data(records, top_n_features=top_n)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validator қате: {str(e)}")


@router.get(
    "/validator/report",
    summary="ML Validator — Жергілікті есеп (JSONL файлынан)",
    description=(
        "Жергілікті `training_data/training_data.jsonl` файлынан оқып, "
        "толық сапа есебін шығарады.\n\n"
        "Бұл endpoint Spring Boot қажет емес — тікелей FastAPI-дан іске қосылады."
    ),
    tags=["Training Data"],
)
async def get_local_report(top_n: int = 5):
    try:
        if not os.path.exists(_DATA_FILE):
            return {
                "message": "training_data.jsonl табылмады. Алдымен дәрігер тағайындаулар сақталуы керек.",
                "total_records": 0,
                "readiness_score": 0.0,
                "readiness_label": "недостаточно",
            }
        records = []
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        result = analyze_collected_data(records, top_n_features=top_n)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Есеп жасау қатесі: {str(e)}")