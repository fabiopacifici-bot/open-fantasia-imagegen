# Spec: Issue #13 — No tests for server endpoints, hot-swap logic, or resolve_size edge cases

- **Source:** GitHub issue `#13` (OPEN) — fetched from `fabiopacifici-bot/open-fantasia-imagegen`
- **State:** OPEN
- **Labels:** `agent-ready`
- **Author:** `fabiopacifici-bot` (Olly)
- **Created:** 2026-04-25T22:46:11Z
- **Updated:** 2026-04-25T22:46:11Z

## Problem

`tests/test_imagegen.py` only covers `enhance_prompt` and a single happy-path `generate`
call. There are **zero tests** for:

1. `src/server.py` — `/generate`, `/health`, `/edit` (503 path), and the **hot-swap**
   path (`_do_generate` with `req.model` different from `_model_id`).
2. `resolve_size` with an **invalid quality string** (should raise `ValueError`).
3. `is_gguf` with **edge inputs**.

The hot-swap path at `server.py:211-217` is particularly risky given no coverage exists
for its **VRAM management side-effects**.

## Fix

Add a `tests/test_server.py` using **FastAPI `TestClient`** with a **mocked `get_pipeline`**,
covering **at minimum**:

- `/health` — response shape
- `/generate` — returns **503** when the lock is held
- `/edit` — returns **503**
- **hot-swap** — triggers `_unload_flux_for_swap`

## Acceptance criteria

- New `tests/test_server.py` exists and runs green.
- The four minimum coverage cases above pass.
- Existing `tests/test_imagegen.py` continues to pass (no regressions).