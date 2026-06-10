"""
middleware/auth.py
CardioTracker ML v2.2 — API Key аутентификациясы

Мақсат: FastAPI сервисіне тек Spring Boot кіре алсын,
        басқа сұраныстар 403 Forbidden алсын.

Логика:
  - Әр сұраныста X-API-Key header тексереді
  - Дұрыс key болмаса → 403 Forbidden + JSON қате
  - Ашық эндпоинттер (key қажет емес):
      GET  /
      GET  /health
      GET  /docs
      GET  /openapi.json
      GET  /redoc
  - Key .env файлынан оқылады (ML_API_KEY айнымалысы)
  - .env жоқ болса — дефолт key қолданылады (тек локальды дев үшін)
"""

import os
import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

# .env-тен оқу — python-dotenv орнатылса автоматты, болмаса os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv жоқ болса — тек os.environ қолданылады

# API key — .env-тен немесе environment variable-дан
_API_KEY: str = os.environ.get("ML_API_KEY", "cardio-secret-2024")

# Key header атауы
_KEY_HEADER = "X-API-Key"

# Аутентификациясыз өтетін эндпоинттер (prefix немесе exact match)
_PUBLIC_PATHS: set[str] = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}


# ─────────────────────────────────────────────────────────────────────────────
#  MIDDLEWARE КЛАСЫ
# ─────────────────────────────────────────────────────────────────────────────

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Starlette BaseHTTPMiddleware негізіндегі API Key тексерушісі.

    Қолданылуы (main.py-де):
        from middleware.auth import APIKeyMiddleware
        app.add_middleware(APIKeyMiddleware)
    """

    def __init__(self, app: ASGIApp, api_key: Optional[str] = None) -> None:
        super().__init__(app)
        # main.py-ден key берілсе — соны қолдан, болмаса .env-ті
        self._api_key = api_key or _API_KEY
        logger.info(
            "APIKeyMiddleware іске қосылды. "
            "Ашық эндпоинттер: %s",
            sorted(_PUBLIC_PATHS),
        )

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # ── Ашық эндпоинттерге key тексерілмейді ─────────────────────────
        if self._is_public(path):
            return await call_next(request)

        # ── Key тексеру ───────────────────────────────────────────────────
        provided_key = request.headers.get(_KEY_HEADER)

        if not provided_key:
            logger.warning(
                "API Key жоқ: %s %s | IP: %s",
                request.method,
                path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error":   "Forbidden",
                    "detail":  f"API Key қажет. '{_KEY_HEADER}' header-ін жіберіңіз.",
                    "code":    "MISSING_API_KEY",
                },
            )

        if provided_key != self._api_key:
            logger.warning(
                "Жарамсыз API Key: %s %s | IP: %s",
                request.method,
                path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error":   "Forbidden",
                    "detail":  "API Key жарамсыз.",
                    "code":    "INVALID_API_KEY",
                },
            )

        # ── Key дұрыс — сұраныс өтеді ─────────────────────────────────────
        return await call_next(request)

    @staticmethod
    def _is_public(path: str) -> bool:
        """Эндпоинт ашық па — тексереді."""
        if path in _PUBLIC_PATHS:
            return True
        # /docs/... сияқты Swagger sub-path-тары
        if path.startswith("/docs/") or path.startswith("/redoc/"):
            return True
        return False