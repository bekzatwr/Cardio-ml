"""
routers/summary.py — ИИ-заключение HTTP роутері
CardioTracker ML v2.2

Архитектура: thin router — барлық логика services/summary_service.py-да.

Түзетулер v2.2:
  ✅ Description қазақша/орысша аралас → тазаланды
  ✅ summary_service v2.2: "бірінші визит" режимі, ДИНАМИКА бөлімі
"""

from fastapi import APIRouter, HTTPException

from schemas import SummaryRequest, SummaryResponse
from services.summary_service import generate_summary

router = APIRouter()


@router.post(
    "/ai-summary",
    response_model=SummaryResponse,
    summary="Генерация AI клинического заключения v2.2",
    description=(
        "Формирует полное клиническое заключение на казахском языке.\n\n"
        "**Структура заключения:**\n"
        "1. `ЖАЛПЫ ЖАҒДАЙ` — группа риска, балл, активные алерты\n"
        "2. `НЕГІЗГІ КӨРСЕТКІШТЕР` — ФВ/ЛЖ, NYHA, бүйрек, анемия, ЭКГ\n"
        "3. `ДИНАМИКА` — сравнение с предыдущим визитом "
        "(только если `visit_history` не пустой)\n"
        "4. `КЛИНИКАЛЫҚ ЕСКЕРТУЛЕР` — кардиоренальный синдром, "
        "гепатопатия и др.\n"
        "5. `НАЗАР АУДАРЫҢЫЗ` — зоны риска (ФВ <30%, NT >900, Hb <110)\n"
        "6. `ҰСЫНЫС` — рекомендация по дальнейшей тактике\n\n"
        "**ИИ НЕ ставит диагноз и НЕ назначает лечение** — "
        "только факты и отклонения (ТЗ v1.0).\n\n"
        "Если `visit_history = []` — режим первого визита, "
        "раздел ДИНАМИКА не генерируется."
    ),
    tags=["ИИ-заключение"],
)
async def generate_summary_endpoint(req: SummaryRequest) -> SummaryResponse:
    try:
        return generate_summary(req)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка генерации заключения: {str(e)}",
        )