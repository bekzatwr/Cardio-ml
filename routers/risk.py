"""
routers/risk.py — Риск классификациясы HTTP роутері
CardioTracker ML v2.2

Архитектура: thin router — барлық логика services/risk_service.py-да.

Түзетулер v2.2:
  ✅ medium_count AlertResponse-та жоқ болатын → schemas.py-де қосылды,
     роутер medium_count есептеп береді
  ✅ Версия description v2.2-ге жаңартылды
"""

from fastapi import APIRouter, HTTPException

from schemas import PatientData, RiskResponse
from services.risk_service import classify_risk

router = APIRouter()


@router.post(
    "/risk-classification",
    response_model=RiskResponse,
    summary="Классификация риска ACC/AHA v2.2",
    description=(
        "Оценивает группу риска пациента по ACC/AHA стадиям.\n\n"
        "**Группы риска:**\n"
        "- `Норма (A)` 🟢 — факторы риска, структурных изменений нет\n"
        "- `Стадия B` 🔵 — структурные изменения без симптомов\n"
        "- `Риск C` 🟡 — структурные изменения + симптомы ХСН\n"
        "- `C→D` 🟠 — тяжёлая ХСН, рефрактерная к терапии\n"
        "- `Стадия D` 🔴 — терминальная ХСН\n\n"
        "**Параметры классификации:**\n"
        "ФВ ЛЖ, NT-proBNP, Тест 6 мин (NYHA), "
        "Биохимия (Cr, Hb, Мочевина, Билирубин, АСТ/АЛТ), "
        "ЭКГ-флаги, BMI, Возраст, Коморбидность\n\n"
        "**Исправления v2.2:**\n"
        "- `_score_walk`: шкала NYHA по ТЗ (<150/150–425/426–550/>550м)\n"
        "- `_score_nt_probnp`: NT ≥125 → зона наблюдения (было >125)\n"
        "- Предупреждения: парадокс NT, кардиоренальный синдром, ИМ в анамнезе\n"
        "- `patient_id`: тип `str` (было `int`)"
    ),
    tags=["Классификация рисков"],
)
async def classify_risk_endpoint(patient: PatientData) -> RiskResponse:
    try:
        return classify_risk(patient)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка классификации риска: {str(e)}",
        )