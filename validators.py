from datetime import datetime, timedelta
from typing import Optional


def is_valid_coordinate(lat: Optional[float], lon: Optional[float]) -> bool:
    if lat is None or lon is None:
        return False
    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return False
    return -90 <= lat <= 90 and -180 <= lon <= 180


def is_recent(dt: Optional[datetime], days: int = 14) -> bool:
    if dt is None:
        return False
    return dt >= datetime.utcnow() - timedelta(days=days)
