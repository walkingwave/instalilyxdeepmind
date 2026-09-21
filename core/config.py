"""Environment loading.

Keys live in .env (gitignored). Every entry point calls load_env() before
touching a provider so behaviour is the same from a shell, a test, or a cron.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None) -> None:
    env_path = path or (_ROOT / ".env")
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if env_path.exists():
        load_dotenv(env_path, override=False)
