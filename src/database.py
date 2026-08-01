"""Thread-safe asynchronous SQLite audit logging for PPE violations."""

from __future__ import annotations

import sqlite3
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class ViolationDatabase:
    """Own a SQLite database and submit short logging operations off the UI thread."""

    def __init__(self, database_path: str | Path = "violations.db") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppe-sqlite")
        self._initialise()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self._connection() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS violations (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, violation_type TEXT NOT NULL, confidence_score REAL NOT NULL, snapshot_path TEXT NOT NULL)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp)")

    def log_violation_async(self, violation_type: str, confidence_score: float, snapshot_path: str) -> Future[int]:
        """Queue a violation insert and return a future containing its row ID."""
        if not violation_type.strip():
            raise ValueError("violation_type must not be empty.")
        return self._executor.submit(self._insert, violation_type, confidence_score, snapshot_path)

    def _insert(self, violation_type: str, confidence_score: float, snapshot_path: str) -> int:
        with self._connection() as connection:
            cursor = connection.execute("INSERT INTO violations (timestamp, violation_type, confidence_score, snapshot_path) VALUES (?, ?, ?, ?)", (datetime.now(timezone.utc).isoformat(), violation_type, float(confidence_score), snapshot_path))
            return int(cursor.lastrowid)

    def query(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        """Return violations within an inclusive UTC date-time window, newest first."""
        clauses: list[str] = []
        params: list[str] = []
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start.astimezone(timezone.utc).isoformat())
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end.astimezone(timezone.utc).isoformat())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            return pd.read_sql_query(f"SELECT id, timestamp, violation_type, confidence_score, snapshot_path FROM violations{where} ORDER BY timestamp DESC", connection, params=params)

    def close(self) -> None:
        """Shut down the background logger once the process exits."""
        self._executor.shutdown(wait=False, cancel_futures=False)
