import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional


class Storage:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    request_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    style TEXT NOT NULL,
                    reference_audio_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_path TEXT,
                    error_message TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    user_id INTEGER,
                    job_id TEXT,
                    message TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_user(self, username: str, password_hash: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES(?,?,?)",
                (username, password_hash, now),
            )
            return int(cursor.lastrowid)

    def get_user_by_username(self, username: str) -> Optional[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()

    def create_session(self, token: str, user_id: int, ttl_hours: int = 24) -> None:
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=ttl_hours)
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions(token, user_id, created_at, expires_at) VALUES(?,?,?,?)",
                (token, user_id, now.isoformat(), expires.isoformat()),
            )

    def get_session(self, token: str) -> Optional[sqlite3.Row]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE token = ?",
                (token,),
            ).fetchone()
        if not row:
            return None
        expires = datetime.fromisoformat(row["expires_at"])
        if expires < datetime.now(timezone.utc):
            self.delete_session(token)
            return None
        return row

    def delete_session(self, token: str) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

    def create_job(self, job: Dict[str, str]) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    id, user_id, request_id, prompt, style, reference_audio_path, status,
                    output_path, error_message, metadata, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job["id"],
                    job["user_id"],
                    job["request_id"],
                    job["prompt"],
                    job["style"],
                    job["reference_audio_path"],
                    job["status"],
                    None,
                    None,
                    json.dumps(job.get("metadata", {})),
                    job["created_at"],
                    job["updated_at"],
                ),
            )

    def update_job_status(
        self,
        job_id: str,
        status: str,
        output_path: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, output_path = COALESCE(?, output_path),
                    error_message = COALESCE(?, error_message),
                    metadata = COALESCE(?, metadata), updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    output_path,
                    error_message,
                    json.dumps(metadata) if metadata is not None else None,
                    now,
                    job_id,
                ),
            )

    def get_job(self, job_id: str, user_id: int) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ? AND user_id = ?",
                (job_id, user_id),
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["metadata"] = json.loads(data["metadata"] or "{}")
        return data

    def list_jobs(self, user_id: int, limit: int = 30) -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        jobs: List[Dict] = []
        for row in rows:
            job = dict(row)
            job["metadata"] = json.loads(job["metadata"] or "{}")
            jobs.append(job)
        return jobs

    def recent_job_count(self, user_id: int, window_seconds: int) -> int:
        since = (datetime.now(timezone.utc) - timedelta(seconds=window_seconds)).isoformat()
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM jobs WHERE user_id = ? AND created_at >= ?",
                (user_id, since),
            ).fetchone()
        return int(row["cnt"])

    def record_telemetry(
        self,
        request_id: str,
        message: str,
        user_id: Optional[int] = None,
        job_id: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO telemetry(request_id, user_id, job_id, message, details, created_at) VALUES(?,?,?,?,?,?)",
                (
                    request_id,
                    user_id,
                    job_id,
                    message,
                    json.dumps(details or {}),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
