"""Redis-backed cache with in-memory fallback"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class CacheBackend:
    def __init__(self) -> None:
        self._redis = None
        self._memory: dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        settings = get_settings()
        if not settings.enable_redis_cache:
            return
        try:
            import redis.asyncio as redis  # type: ignore

            self._redis = redis.from_url(settings.redis_url, decode_responses=True)
            await self._redis.ping()
        except Exception as e:
            logger.warning("Redis unavailable, cache falls back to memory: %s", e)
            self._redis = None

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    async def get_json(self, key: str) -> dict[str, Any] | None:
        now = time.time()
        if self._redis is not None:
            raw = await self._redis.get(key)
            if not raw:
                return None
            try:
                return json.loads(raw)
            except Exception:
                return None
        async with self._lock:
            payload = self._memory.get(key)
            if not payload:
                return None
            exp, raw = payload
            if exp <= now:
                self._memory.pop(key, None)
                return None
            try:
                return json.loads(raw)
            except Exception:
                return None

    async def set_json(self, key: str, value: dict[str, Any], ttl_sec: int) -> None:
        raw = json.dumps(value, ensure_ascii=False)
        if self._redis is not None:
            await self._redis.set(key, raw, ex=ttl_sec)
            return
        async with self._lock:
            self._memory[key] = (time.time() + ttl_sec, raw)

    @staticmethod
    def make_key(prefix: str, *parts: str) -> str:
        basis = "|".join([prefix, *parts])
        return f"{prefix}:{hashlib.sha256(basis.encode('utf-8')).hexdigest()}"


cache_backend = CacheBackend()
