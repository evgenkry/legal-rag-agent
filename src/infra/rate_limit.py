"""Rate limiting backed by Redis with in-memory fallback."""

import asyncio
import logging
import time

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self) -> None:
        self._redis = None
        self._memory: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        settings = get_settings()
        if not settings.enable_rate_limit:
            return
        try:
            import redis.asyncio as redis  # type: ignore

            self._redis = redis.from_url(settings.redis_url, decode_responses=True)
            await self._redis.ping()
        except Exception as e:
            logger.warning("Redis unavailable, limiter falls back to memory: %s", e)
            self._redis = None

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    async def check_and_hit(self, user_key: str, *, limit: int, period_sec: int = 60) -> bool:
        if not user_key:
            return True
        bucket = f"ratelimit:{user_key}"
        now = int(time.time())
        if self._redis is not None:
            current = await self._redis.incr(bucket)
            if current == 1:
                await self._redis.expire(bucket, period_sec)
            return current <= limit

        async with self._lock:
            timestamps = self._memory.get(bucket, [])
            threshold = now - period_sec
            timestamps = [t for t in timestamps if t > threshold]
            if len(timestamps) >= limit:
                self._memory[bucket] = timestamps
                return False
            timestamps.append(float(now))
            self._memory[bucket] = timestamps
            return True


rate_limiter = RateLimiter()
