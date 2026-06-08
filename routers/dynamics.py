"""
routers/dynamics.py — Динамика және тренд анализі HTTP роутері
CardioTracker ML v2.2

Архитектура: thin router — барлық логика services/dynamics_service.py-да.

Түзетулер v2.2:
  ✅ ValueError → HTTPException 400 (минимум 2 визит шарты)
  ✅ DynamicsResponse.error_message — schemas.py v2.2-де қосылды
  ✅ Description v2.2 функциялар тізімімен жаңартылды
"""

from fastapi import APIRouter, HTTPException

from schemas import DynamicsRequest, DynamicsResponse
from services.dynamics_service import analyze_dynamics

router = APIRouter()


@router.post(
    "/dynamics-analysis",
    response_model=DynamicsResponse,
    summary="Анализ динамики и трендов v2.2",
    description=(
        "Анализирует изменения ключевых показателей по нескольким визитам.\n\n"
        "**Требования:** минимум 2 визита в `visits` (от старого к новому).\n\n"
        "**Параметры трендов:**\n"
        "ФВ ЛЖ, NT-proBNP, Тест 6 мин, Креатинин, Гемоглобин, "
        "Вес, Мочевина, Билирубин\n\n"
        "**Возможности v2.2:**\n"
        "- Линейная регрессия `linear_regression(x, y)`\n"
        "- Прогноз на следующий визит (`forecast_next`)\n"
        "- Velocity в месяц (только при заполненном `visit_date`)\n"
        "- Severity: `normal` / `mild` / `moderate` / `severe`\n"
        "- Прогрессия группы риска (`risk_progression`)\n"
        "- Средний интервал (`avg_interval_days`) — при наличии дат\n"
        "- `build_day_axis`, `calc_avg_interval` — экспортированы для тестов"
    ),
    tags=["Динамика и тренды"],
)
async def analyze_dynamics_endpoint(req: DynamicsRequest) -> DynamicsResponse:
    try:
        return analyze_dynamics(req)
    except ValueError as e:
        # Pydantic validator немесе analyze_dynamics: <2 визит
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка анализа динамики: {str(e)}",
        )