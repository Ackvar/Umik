from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

DB_TS_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_TIMEZONE = "Europe/Moscow"


def _load_app_timezone() -> tzinfo:
    tz_name = os.getenv("APP_TIMEZONE", DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(tz_name)
    except Exception:
        if tz_name == DEFAULT_TIMEZONE:
            return timezone(timedelta(hours=3), name="MSK")
        local_tz = datetime.now().astimezone().tzinfo
        return local_tz or timezone.utc


APP_TIMEZONE = _load_app_timezone()


def app_now() -> datetime:
    return datetime.now(APP_TIMEZONE)


def format_db_timestamp(dt: datetime | None = None) -> str:
    dt = dt or app_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=APP_TIMEZONE)
    else:
        dt = dt.astimezone(APP_TIMEZONE)
    return dt.strftime(DB_TS_FORMAT)


def parse_db_timestamp(ts_value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts_value)
    except ValueError:
        dt = datetime.strptime(ts_value, DB_TS_FORMAT)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=APP_TIMEZONE)
    return dt.astimezone(APP_TIMEZONE)


def window_start_db_timestamp(*, minutes: int = 0, seconds: int = 0) -> str:
    return format_db_timestamp(app_now() - timedelta(minutes=minutes, seconds=seconds))


def to_utc_iso(ts_value: str | datetime) -> str:
    if isinstance(ts_value, str):
        dt = parse_db_timestamp(ts_value)
    else:
        dt = ts_value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=APP_TIMEZONE)
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
