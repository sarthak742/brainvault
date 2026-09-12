"""
Celery worker for BrainVault.

This is a SEPARATE PROGRAM from the web server (app.py). You run it on its own:

    celery -A tasks.celery_app worker --loglevel=info --concurrency=1

- `broker`  = where jobs (tickets) are stored and handed out  -> Redis
- `backend` = where task state + results + progress are stored -> Redis

Because the ticket lives in Redis (not in the web server's memory), the web
server can crash and the job survives. And if THIS worker crashes mid-job, the
ticket goes back on the queue for the next worker.

--concurrency=1 means this worker runs one indexing job at a time, so two jobs
can never write the FAISS index file at the same moment.
"""
import os
import logging

from celery import Celery

import indexing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# One Redis URL for both the broker (job queue) and the backend (results/progress).
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("brainvault", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_track_started=True,      # report a STARTED state so the UI can show "working"
    result_expires=3600,          # forget finished results after an hour
)


@celery_app.task(bind=True)
def index_document_task(self, filename: str, user_id: str = "default") -> dict:
    """Index one user's uploaded document, reporting progress as it goes.

    `self.update_state(...)` writes progress into Redis. The web server reads it
    back through /api/index-status/<task_id> to drive a progress bar.
    """
    def progress(done: int, total: int, message: str) -> None:
        self.update_state(
            state="PROGRESS",
            meta={"done": done, "total": total, "message": message},
        )

    return indexing.index_single_document(filename, user_id=user_id, progress_cb=progress)


@celery_app.task(bind=True)
def full_rebuild_task(self) -> dict:
    """Full rebuild (used after a delete), with progress reporting."""
    def progress(done: int, total: int, message: str) -> None:
        self.update_state(
            state="PROGRESS",
            meta={"done": done, "total": total, "message": message},
        )

    return indexing.full_rebuild_index(progress_cb=progress)
