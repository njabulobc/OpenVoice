from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Callable, Dict


class JobQueue:
    def __init__(self, worker: Callable[[str], None], max_workers: int = 1) -> None:
        self.worker = worker
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="openvoice-job")
        self._running: Dict[str, bool] = {}
        self._lock = Lock()

    def enqueue(self, job_id: str) -> None:
        with self._lock:
            self._running[job_id] = False
        self.executor.submit(self._run, job_id)

    def _run(self, job_id: str) -> None:
        with self._lock:
            self._running[job_id] = True
        try:
            self.worker(job_id)
        finally:
            with self._lock:
                self._running.pop(job_id, None)

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            return self._running.get(job_id, False)
