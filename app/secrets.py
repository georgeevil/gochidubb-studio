"""Secrets store — platform credentials kept OUT of config-user.json.

``GET /api/config`` is unauthenticated, so anything living on UserConfig is
readable by any local caller (and, with permissive CORS, any webpage). API
tokens for publishing (VK, later RuTube) therefore live here instead:
``secrets.json`` at the project root, written atomically with mode 0600,
and never echoed by any route — callers only ever see presence booleans
via :func:`secret_status`.

Precedence mirrors ``app.config._load_config``: environment variables win
over the file. Values are NEVER logged.

Usage:
    from app.secrets import get_secret, set_secret, secret_status

    get_secret("vk_access_token")   # "" if unset
    set_secret("vk_group_id", "123456")   # persists immediately
    secret_status()   # {"vk_access_token": True, "vk_group_id": False}
"""
import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger("gochidubb.secrets")

# Same BASE resolution as app/config.py — project root is the parent of app/.
BASE = Path(__file__).parent.parent.resolve()
SECRETS_FILE = BASE / "secrets.json"

# Known secret keys -> environment variable that overrides the file value.
KNOWN_SECRETS = {
    "vk_access_token": "VK_ACCESS_TOKEN",
    "vk_group_id": "VK_GROUP_ID",
    "linear_api_key": "LINEAR_API_KEY",
    "linear_team_id": "LINEAR_TEAM_ID",
    "linear_project_id": "LINEAR_PROJECT_ID",
    # SMTP, for the bug-report fallback when the issue tracker refuses the
    # write. smtp_password is the only true credential here; the rest live
    # alongside it because a half-configured mailer is worse than none, and
    # keeping the set together means one presence check covers it.
    "smtp_host": "SMTP_HOST",
    "smtp_port": "SMTP_PORT",
    "smtp_security": "SMTP_SECURITY",
    "smtp_username": "SMTP_USERNAME",
    "smtp_password": "SMTP_PASSWORD",
    "smtp_from": "SMTP_FROM",
    # Overrides app.bugreport.SUPPORT_EMAIL for anyone self-hosting.
    "bugreport_email": "BUGREPORT_EMAIL",
}


class SecretsStore:
    """File-backed secret storage with env-var override on read."""

    def __init__(self, path: Path = SECRETS_FILE):
        self._path = Path(path)
        self._values: dict = {}
        self._load()

    # ── Reading ────────────────────────────────────────────────────────
    def _load(self) -> None:
        data = {}
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except Exception as e:
                # Never include file contents in the log — just the reason.
                log.warning(f"[secrets] Could not read {self._path.name}: {e}")
        self._values = {k: str(data.get(k) or "") for k in KNOWN_SECRETS}

    def get(self, key: str) -> str:
        """Return the secret value ("" if unset). Env var wins over the file."""
        if key not in KNOWN_SECRETS:
            raise KeyError(f"Unknown secret key: {key!r}")
        env = os.getenv(KNOWN_SECRETS[key])
        if env is not None and env.strip():
            return env.strip()
        return self._values.get(key, "")

    def status(self) -> dict:
        """Presence booleans only — safe to expose over HTTP."""
        return {k: bool(self.get(k)) for k in KNOWN_SECRETS}

    # ── Writing ────────────────────────────────────────────────────────
    def set(self, key: str, value) -> None:
        """Set a secret and persist immediately (atomic write, chmod 600)."""
        if key not in KNOWN_SECRETS:
            raise KeyError(f"Unknown secret key: {key!r}")
        self._values[key] = str(value or "").strip()
        self._save()

    def _save(self) -> None:
        payload = {k: v for k, v in self._values.items() if v}
        try:
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), prefix=".secrets-", suffix=".tmp"
            )
            try:
                # Restrict the temp file BEFORE writing any secret bytes.
                try:
                    os.fchmod(fd, 0o600)
                except (AttributeError, OSError):
                    pass  # Windows / exotic filesystems
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self._path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass
        except Exception as e:
            log.warning(f"[secrets] Could not save {self._path.name}: {e}")


# Singleton — mirrors app.config.cfg. Import the module-level functions below.
_store = SecretsStore()


def get_secret(key: str) -> str:
    """Return the secret value for `key` ("" if unset). Raises KeyError on unknown keys."""
    return _store.get(key)


def set_secret(key: str, value) -> None:
    """Set and persist a secret. Raises KeyError on unknown keys."""
    _store.set(key, value)


def secret_status() -> dict:
    """Presence booleans per known key — never the values themselves."""
    return _store.status()
