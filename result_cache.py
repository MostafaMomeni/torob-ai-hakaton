"""
Search-result cache keyed by result_id.

When a full search runs (LLM path), the ordered product_ids + metadata are stored
under a short-lived result_id (default TTL = 30 minutes). Subsequent page requests
with that result_id are served from cache without calling the LLM.

Storage: JSON files under data/result_cache/{result_id}.json
Expired entries are deleted on read (and opportunistically on write).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shopping_assistant.result_cache")

DEFAULT_TTL_SECONDS = 30 * 60  # 30 minutes
MAX_CACHED_PRODUCT_IDS = 100


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


class ResultCache:
    """Disk-backed cache of search result sets keyed by result_id."""

    def __init__(
        self,
        cache_dir: Path,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_product_ids: int = MAX_CACHED_PRODUCT_IDS,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = int(ttl_seconds)
        self.max_product_ids = int(max_product_ids)

    def _path(self, result_id: str) -> Path:
        # sanitize: only allow safe chars in filename
        safe = "".join(c for c in result_id if c.isalnum() or c in ("_", "-"))
        if not safe or safe != result_id:
            raise ValueError(f"invalid result_id: {result_id!r}")
        return self.cache_dir / f"{safe}.json"

    def _new_id(self) -> str:
        return "res_" + uuid.uuid4().hex[:16]

    def put(
        self,
        product_ids: List[str],
        *,
        total: Optional[int] = None,
        applied_filters: Optional[Dict[str, Any]] = None,
        search_snapshot: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        query_text: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Store a ranked result set. Returns the cache entry (includes result_id, expires_at)."""
        result_id = self._new_id()
        now = _utcnow()
        ttl = int(ttl_seconds if ttl_seconds is not None else self.ttl_seconds)
        expires = now + timedelta(seconds=ttl)

        ids = [str(pid) for pid in (product_ids or []) if pid][: self.max_product_ids]
        entry = {
            "result_id": result_id,
            "created_at": _iso(now),
            "expires_at": _iso(expires),
            "ttl_seconds": ttl,
            "product_ids": ids,
            "total": int(total if total is not None else len(ids)),
            "applied_filters": applied_filters or {},
            "search_snapshot": search_snapshot or {},
            "conversation_id": conversation_id,
            "query_text": query_text,
        }
        path = self._path(result_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, default=str)
        logger.info(
            "Cached result_id=%s products=%s total=%s expires=%s",
            result_id,
            len(ids),
            entry["total"],
            entry["expires_at"],
        )
        # opportunistic cleanup of a few expired files
        self.cleanup_expired(max_files=20)
        return entry

    def get(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Return cache entry if present and not expired; else None (and delete if expired)."""
        if not result_id or not isinstance(result_id, str):
            return None
        try:
            path = self._path(result_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                entry = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read result cache %s: %s", result_id, e)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

        expires = _parse_iso(entry.get("expires_at"))
        if expires is None or _utcnow() >= expires:
            logger.info("Result cache expired or invalid: %s", result_id)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        return entry

    def delete(self, result_id: str) -> bool:
        try:
            path = self._path(result_id)
        except ValueError:
            return False
        if path.exists():
            path.unlink(missing_ok=True)
            return True
        return False

    def cleanup_expired(self, max_files: int = 50) -> int:
        """Delete up to max_files expired cache entries. Returns number removed."""
        removed = 0
        try:
            files = list(self.cache_dir.glob("res_*.json"))
        except OSError:
            return 0
        now = _utcnow()
        for path in files[: max(1, max_files)]:
            try:
                with path.open("r", encoding="utf-8") as f:
                    entry = json.load(f)
                expires = _parse_iso(entry.get("expires_at"))
                if expires is None or now >= expires:
                    path.unlink(missing_ok=True)
                    removed += 1
            except (OSError, json.JSONDecodeError):
                try:
                    path.unlink(missing_ok=True)
                    removed += 1
                except OSError:
                    pass
        return removed
