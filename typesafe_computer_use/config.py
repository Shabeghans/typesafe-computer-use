"""Tunables, the site catalog, and environment loading."""

from __future__ import annotations

import os
from pathlib import Path

MIN_OCR_CONFIDENCE = 0.3
MAX_OPTIONS = 255  # TypeSafe Choice ceiling
ABORT_CORNER_PX = 4
DEFAULT_MIN_CONFIDENCE = 0.4
DEFAULT_STEPS = 100
DEFAULT_DELAY = 2.0
DEFAULT_WRITER_MODEL = "claude-haiku-4-5"
DEFAULT_ANSWER_MODEL = "claude-sonnet-5"  # runs once per run, on a screenshot: worth a stronger reader
DEFAULT_BROWSER = "Comet"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"  # serves TypeSafe's /v1/systemone and Anthropic's /v1/messages
# OpenRouter names the Claude models with a vendor prefix and dotted versions.
OPENROUTER_MODELS = {
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
}

# Sites the classifier can pick by name. Anything else goes through the writer.
SITES: dict[str, str] = {
    "github": "https://github.com/",
    "gmail": "https://mail.google.com/",
    "google_calendar": "https://calendar.google.com/",
    "launchdarkly": "https://app.launchdarkly.com/",
    "linear": "https://linear.app/",
    "notion": "https://www.notion.so/",
    "slack": "https://app.slack.com/",
    "typesafe_console": "https://console.typesafe.ai/",
}


def load_dotenv(path: Path) -> None:
    """Set KEY=VALUE lines from a .env file into the environment unless already set."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def decision_client() -> dict[str, str]:
    """TypeSafeClient arguments: OpenRouter when OPENROUTER_API_KEY is set, TypeSafe's own API otherwise."""
    key = os.environ.get("OPENROUTER_API_KEY")
    return {"api_key": key, "base_url": OPENROUTER_BASE_URL} if key else {}


def writer_client() -> dict[str, str]:
    """anthropic.Anthropic arguments: OpenRouter when OPENROUTER_API_KEY is set, Anthropic's own API otherwise.

    An explicit token stops the SDK reading ANTHROPIC_API_KEY, so that key never goes to OpenRouter.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    return {"auth_token": key, "base_url": OPENROUTER_BASE_URL} if key else {}


def _model(env: str, default: str) -> str:
    model = os.environ.get(env, default)
    return OPENROUTER_MODELS.get(model, model) if os.environ.get("OPENROUTER_API_KEY") else model


def browser() -> str:
    return os.environ.get("CLICKER_BROWSER", DEFAULT_BROWSER)


def writer_model() -> str:
    return _model("CLICKER_WRITER_MODEL", DEFAULT_WRITER_MODEL)


def answer_model() -> str:
    return _model("CLICKER_ANSWER_MODEL", DEFAULT_ANSWER_MODEL)


def email() -> str | None:
    return os.environ.get("CLICKER_EMAIL") or None
