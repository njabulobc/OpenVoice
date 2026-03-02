import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .auth import AuthService
from .queue import JobQueue
from .storage import Storage
from .structured_logging import configure_logging


class OpenVoiceService:
    def __init__(
        self,
        db_path: str,
        output_dir: str,
        synthesize_fn: Callable[[str, str, str, str, str], Dict],
        max_jobs_per_window: int = 5,
        rate_window_seconds: int = 60,
    ) -> None:
        self.storage = Storage(db_path)
        self.auth = AuthService(self.storage)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.output_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.synthesize_fn = synthesize_fn
        self.max_jobs_per_window = max_jobs_per_window
        self.rate_window_seconds = rate_window_seconds
        self.logger = configure_logging()
        self.queue = JobQueue(worker=self._run_job, max_workers=1)

    def register_or_login(self, username: str, password: str, request_id: str) -> str:
        try:
            self.auth.register(username, password)
            self.logger.info("user_registered", extra={"request_id": request_id})
        except ValueError:
            pass
        token = self.auth.login(username, password)
        self.logger.info("user_authenticated", extra={"request_id": request_id})
        return token

    def submit_job(
        self,
        token: str,
        prompt: str,
        style: str,
        reference_audio_path: str,
        request_id: str,
    ) -> str:
        user = self.auth.require_user(token)
        if len(prompt) < 2 or len(prompt) > 200:
            raise ValueError("Prompt length must be between 2 and 200 characters")
        if self.storage.recent_job_count(user.user_id, self.rate_window_seconds) >= self.max_jobs_per_window:
            self.storage.record_telemetry(
                request_id=request_id,
                user_id=user.user_id,
                message="rate_limit_exceeded",
                details={"window_seconds": self.rate_window_seconds, "max_jobs": self.max_jobs_per_window},
            )
            raise ValueError("Rate limit exceeded. Please wait before submitting more jobs.")

        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        job = {
            "id": job_id,
            "user_id": user.user_id,
            "request_id": request_id,
            "prompt": prompt,
            "style": style,
            "reference_audio_path": reference_audio_path,
            "status": "queued",
            "metadata": {"username": user.username},
            "created_at": now,
            "updated_at": now,
        }
        self.storage.create_job(job)
        self.queue.enqueue(job_id)
        self.logger.info(
            "job_queued",
            extra={"request_id": request_id, "user_id": user.user_id, "job_id": job_id},
        )
        return job_id

    def _run_job(self, job_id: str) -> None:
        with self.storage.connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return

        user_id = int(row["user_id"])
        request_id = row["request_id"]
        self.storage.update_job_status(job_id, "running")
        self.logger.info(
            "job_running",
            extra={"request_id": request_id, "user_id": user_id, "job_id": job_id},
        )

        output_path = str(self.jobs_dir / f"{job_id}.wav")
        try:
            metadata = self.synthesize_fn(
                row["prompt"],
                row["style"],
                row["reference_audio_path"],
                output_path,
                request_id,
            )
            self.storage.update_job_status(job_id, "succeeded", output_path=output_path, metadata=metadata)
            self.logger.info(
                "job_succeeded",
                extra={"request_id": request_id, "user_id": user_id, "job_id": job_id},
            )
        except Exception as exc:
            self.storage.update_job_status(job_id, "failed", error_message=str(exc))
            self.storage.record_telemetry(
                request_id=request_id,
                user_id=user_id,
                job_id=job_id,
                message="job_failed",
                details={"error": str(exc)},
            )
            self.logger.exception(
                "job_failed",
                extra={"request_id": request_id, "user_id": user_id, "job_id": job_id},
            )

    def get_job(self, token: str, job_id: str) -> Optional[Dict]:
        user = self.auth.require_user(token)
        job = self.storage.get_job(job_id, user.user_id)
        if job and job["status"] == "queued" and self.queue.is_running(job_id):
            job["status"] = "running"
        return job

    def list_jobs(self, token: str) -> List[Dict]:
        user = self.auth.require_user(token)
        jobs = self.storage.list_jobs(user.user_id)
        for job in jobs:
            if job["status"] == "queued" and self.queue.is_running(job["id"]):
                job["status"] = "running"
        return jobs

    @staticmethod
    def new_request_id() -> str:
        return str(uuid.uuid4())
