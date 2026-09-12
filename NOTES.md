# BrainVault — dev notes

## Ideas to explore later
- **Local-first architecture.** Instead of storing uploaded documents on the
  server and isolating them per user, run indexing + search on the user's own
  machine so files never leave the device. Things to keep in mind:
  - A browser can't read local files directly; the user still has to pick/drag them.
  - RAG doesn't use the file, it uses a vector index built from it — and that index
    has to live somewhere search can run. So "keep the file local" really means
    "keep the index local".
  - Two clean shapes: (1) a local-first / desktop app (one machine = one user, so
    no multi-tenancy is needed — BrainVault on localhost already behaves this way),
    or (2) a hosted web app that embeds in-browser and stores the index client-side
    (possible, but heavy). Trade-off: privacy vs. simplicity.
  - Good interview talking point: "where should the index live — server or device?"

## Remaining work
- **Frontend** (`static/index.html`) is stale — it still polls the removed
  `/api/rebuild-status` and doesn't use the async task flow. Rebuild it from the
  functional spec: poll `/api/index-status/<task_id>` after upload/delete, show a
  progress bar, render chat sources + the `grounded=false` refusal state, and a
  live metrics panel from `/api/metrics`.
- **Multi-tenancy** runs single-tenant by default (no `user_id` sent -> "default").
  To make it multi-user, pass a `user_id` (e.g. a localStorage id) on
  upload / chat / documents.

## How to run (three processes)
1. Redis:   `docker run -d -p 6379:6379 redis`   (or a local `redis-server`)
2. Worker:  `celery -A tasks.celery_app worker --loglevel=info --concurrency=1`
3. Web:     `python app.py`   (http://localhost:8000)
