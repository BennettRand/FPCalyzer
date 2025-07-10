from datetime import datetime, timezone

def start_stopwatch() -> datetime:
    return datetime.now(timezone.utc)

def stop_stopwatch(start: datetime) -> float:
    return (datetime.now(timezone.utc) - start).total_seconds()
