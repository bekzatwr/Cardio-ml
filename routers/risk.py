"""
routers/risk.py — Риск классификациясы HTTP роутері
CardioTracker ML v2.2
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
        "**Группы риска (ТЗ 5.2: норма/C/C->D/D):**\n"
        "- `норма` — нет симптомов / структурных изменений без симптомов\n"
        "- `C` — структурные изменения + симптомы ХСН\n"
        "- `C->D` — тяжёлая ХСН, рефрактерная к терапии\n"
        "- `D` — терминальная ХСН\n\n"
        "**Параметры:** ФВ ЛЖ, NT-proBNP, Тест 6 мин, Биохимия, ЭКГ, BMI, Коморбидность"
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