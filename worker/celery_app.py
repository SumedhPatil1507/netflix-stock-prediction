"""
Celery application factory.
Broker and result backend: Redis.

Set REDIS_URL in .env (default: redis://localhost:6379)
Run worker: celery -A worker.celery_app worker --loglevel=info
"""
import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

celery = Celery(
    "alpha_engine",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
