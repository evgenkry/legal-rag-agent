"""Авторизация для admin."""

from fastapi import Header, HTTPException, status

from src.core.config import get_settings


async def verify_admin_api_key(x_api_key: str | None = Header(None, alias="X-API-KEY")) -> None:
    """
    Проверяет X-API-KEY против ADMIN_API_KEY.
    - ADMIN_API_KEY не задан -> 503 Service Unavailable (admin выключен)
    - Ключ неверный или отсутствует -> 401 Unauthorized
    """
    settings = get_settings()
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API is disabled: ADMIN_API_KEY is not set",
        )
    if not x_api_key or x_api_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-KEY",
        )
