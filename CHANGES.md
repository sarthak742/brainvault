# BrainVault — Self-Reflection Layer: what changed

This change adds a self-reflective RAG loop (CRAG retrieval grading + query
rewrite + answer faithfulness critique) on top of the existing pipeline.
It is fully backward-compatible: with `reflection_enabled: false` the system
behaves exactly as before, and all 7 original tests still pass.

## New files
- `reflection/__init__.py`        — package exports
- `reflection/_json.py`           — robust JSON extraction from LLM output
- `reflection/grader.py`          — RetrievalGrader (CRAG relevance grading)
- `reflection/query_rewriter.py`  — QueryRewriter (rewrite weak queries)
- `reflection/critic.py`          — AnswerCritic (faithfulness self-critique)
- `reflection/builder.py`         — build_answer_engine() factory (reads config)
- `tests/test_reflection.py`      — 15 unit tests (fake client, no network)
- `tests/test_answer_engine_reflection.py` — 4 loop integration tests

## Modified files
- `llm/answer_engine.py` — AnswerEngine now accepts optional grader / rewriter /
  critic and runs the reflection loop. `_build_prompt` gained a `system_prompt`
  argument (used for the stricter regeneration prompt). AnswerResult gained an
  optional `reflection` audit key. Old constructor calls still work unchanged.
- `config.yaml`  — added 6 `reflection_*` settings.
- `config.py`    — added 6 getter functions for those settings.
- `app.py`       — both engine construction sites now call build_answer_engine().
- `main.py`      — engine construction now calls build_answer_engine().
- `README.md`    — added the "Self-Reflection Layer" section.

## How to integrate (since your local copy was lost)
1. `git clone https://github.com/sarthak742/brainvault.git`
2. Copy the files in this folder over the cloned repo (same relative paths).
3. `pip install -r requirements.txt`
4. `python -m pytest tests/ -q`   # reflection tests need no network
5. Commit: `git add -A && git commit -m "Add self-reflective RAG layer (CRAG + answer critique)"`

## Verification done
- 26/26 tests pass (15 unit + 4 loop + 7 original).
- Every changed .py file passes py_compile.
- config.yaml parses.
