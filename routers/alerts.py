"""
routers/alerts.py — Клиникалық алерттер HTTP роутері
CardioTracker ML v2.2

Архитектура: thin router — барлық логика services/alert_service.py-да.

Түзетулер v2.2:
  ✅ AlertResponse.medium_count есептеледі (schemas.py v2.2-де қосылды)
  ✅ Description v2.2 түзетулерін сипаттайды
"""

from fastapi import APIRouter, HTTPException

from schemas import (
    AlertCheckRequest,
    AlertResponse,
    NtProBnpParadoxRequest,
    NtProBnpParadoxResponse,
)
from services.alert_service import check_alerts, check_nt_paradox

router = APIRouter()


@router.post(
    "/check-alerts",
    response_model=AlertResponse,
    summary="Проверка клинических алертов v2.2",
    description=(
        "Проверяет все алерты для текущего визита пациента.\n\n"
        "**Типы алертов:**\n"
        "- `ALERT-01..06` — одиночные параметры "
        "(ФВ, Hb, Cr, NT-proBNP, ЭКГ)\n"
        "- `COMBO-01/05` — клинические синдромы "
        "(кардиоренальный, тройной риск)\n"
        "- `TREND-01..04` — тренды "
        "(требует ≥2 визитов в `visit_history`)\n\n"
        "**Исправления v2.2:**\n"
        "- ALERT-03: Hb <80 → CRITICAL, 80–100 → HIGH, 100–110 → MEDIUM\n"
        "- ALERT-04: только динамика >30% (абсолютный порог убран)\n"
        "- ALERT-05: NT >5000 → CRITICAL, 1800–5000 → HIGH, 900–1800 → MEDIUM\n"
        "- TREND-01: требует ≥2 визитов в истории\n\n"
        "**Приоритеты:** `CRITICAL` → `HIGH` → `MEDIUM` → `LOW`"
    ),
    tags=["Алерты и предупреждения"],
)
async def check_alerts_endpoint(req: AlertCheckRequest) -> AlertResponse:
    try:
        alert_list    = check_alerts(req)
        critical_list = [a for a in alert_list if a.priority == "CRITICAL"]
        high_list     = [a for a in alert_list if a.priority == "HIGH"]
        medium_list   = [a for a in alert_list if a.priority == "MEDIUM"]

        return AlertResponse(
            patient_id=req.current.patient_id,
            total_alerts=len(alert_list),
            critical_count=len(critical_list),
            high_count=len(high_list),
            medium_count=len(medium_list),          # FIX v2.2
            alerts=alert_list,
            requires_immediate_action=len(critical_list) > 0,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка проверки алертов: {str(e)}",
        )


@router.post(
    "/nt-probnp-paradox",
    response_model=NtProBnpParadoxResponse,
    summary="Проверка парадокса NT-proBNP",
    description=(
        "Определяет парадокс NT-proBNP: нормальный уровень маркера "
        "при тяжёлой систолической дисфункции.\n\n"
        "**Критерии:** NT-proBNP <125 пг/мл + ФВ <35% + симптомы ХСН.\n\n"
        "**Причины:** ожирение (дилюция NT-proBNP), "
        "терминальная стадия D (истощение нейрогуморального ответа)."
    ),
    tags=["Алерты и предупреждения"],
)
async def nt_probnp_paradox_endpoint(req: NtProBnpParadoxRequest):
    try:
        return check_nt_paradox(req)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка проверки парадокса NT-proBNP: {str(e)}",
        )