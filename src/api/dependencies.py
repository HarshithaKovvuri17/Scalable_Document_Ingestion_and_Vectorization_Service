"""
FastAPI dependency injection providers.
"""

import logging
from functools import lru_cache

import redis as redis_lib

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_redis_client() -> redis_lib.Redis:  # type: ignore[type-arg]
    """Return a cached Redis client (used for optional connectivity checks)."""
    cfg = get_settings()
    return redis_lib.Redis(
        host=cfg.REDIS_HOST,
        port=cfg.REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=2,
    )
