# Repository Guidelines

## Project Structure & Module Organization
- Backend logic sits in `server.py`, the FastAPI app that wires chat endpoints and tool orchestration from `tools.py`.
- `tools.py` houses network/search helpers, subprocess execution, and note storage utilities; keep new tooling async-friendly.
- `index.html` is the single-page chat client; keep additional static assets alongside it.
- Developer utilities live in `scripts/` (notably `scripts/cli_chat.py`), and shell entrypoints sit in `start.sh` and `start_old.sh`.
- `macos_app.py` contains the optional macOS wrapper; keep platform-specific code isolated here.
- Dependencies belong in `requirements.txt`; add automated checks under a `tests/` directory (not yet present).

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate` — standardize on `.venv` for local work.
- `pip install -r requirements.txt` — sync backend dependencies.
- `./start.sh` — install dependencies then run `uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools`.
- `python scripts/cli_chat.py "Ping the agent"` — smoke-test the streaming API from the terminal.

## Coding Style & Naming Conventions
- Python modules follow PEP 8: four-space indentation, snake_case functions, CapitalizedClass names, and descriptive async coroutines.
- Preserve type hints; add docstrings only when behaviour is non-obvious.
- Keep HTTP handlers thin by delegating parsing and side-effects to `tools.py`; run `python -m black server.py tools.py` before large submissions.
- Front-end HTML/CSS uses two-space indentation and lowercase attributes; keep scripts inline unless shared.

## Testing Guidelines
- There is no committed suite yet; add `pytest` with `pytest-asyncio` when covering FastAPI routes.
- Place new cases under `tests/` (e.g., `test_chat_stream.py`), mirror endpoint names inside class-based groupings, and target ≥80% coverage using `pytest --maxfail=1 --disable-warnings -q`.

## Commit & Pull Request Guidelines
- This snapshot lacks Git history; adopt `type: short summary` commit subjects (e.g., `feat: stream notes tool events`) with bodies explaining context and testing.
- Keep PRs focused; include a short problem statement, before/after behaviour, local command outputs, and screenshots or terminal transcripts when UI changes apply.
- Link related issues and mention follow-up work explicitly so other agents can queue next steps.

## Environment & Configuration Tips
- Backend behaviour is driven by env vars in `server.py` (`OLLAMA_HOST`, `MODEL`, `DEFAULT_NUM_CTX`, `USER_MAX_CTX`); document overrides in PRs.
- Client utilities honour `CHAT_API` or `APP_BACKEND`; adjust these when running against non-default ports or hosts.
