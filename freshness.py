from datetime import datetime, timedelta
from typing import Optional


def compute_staleness(observed_at: Optional[datetime], reference: Optional[datetime] = None) -> Optional[int]:
    """Return number of days since observed_at relative to reference (UTC). If observed_at is None return None."""
    if observed_at is None:
        return None
    ref = reference or datetime.utcnow()
    delta = ref - observed_at
    return max(0, delta.days)


def is_stale(observed_at: Optional[datetime], days_threshold: int = 14) -> bool:
    if observed_at is None:
        return True
    return (datetime.utcnow() - observed_at) > timedelta(days=days_threshold)
