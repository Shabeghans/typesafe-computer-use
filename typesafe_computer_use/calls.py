"""The call log: every request to the decision model and the writer, and what came back.

Both clients are wrapped rather than edited at each call site, so no call can skip the log.
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import msgspec
from PIL import Image

RULE = "=" * 78


class CallLog:
    """Appends one readable entry per call to a file in the run folder."""

    def __init__(self, path: Path):
        self.path = path
        self.count = 0
        self.step: str = "-"  # set by the loop, so each entry says which step made it

    def write(self, service: str, seconds: float, entry: dict) -> None:
        self.count += 1
        header = f"{RULE}\n#{self.count}  step {self.step}  {service}  {seconds:.2f}s  {time.strftime('%H:%M:%S')}\n{RULE}\n"
        with self.path.open("a") as f:
            f.write(header + json.dumps(entry, indent=2, ensure_ascii=False, default=str) + "\n\n")


class LoggedTypeSafe:
    """A TypeSafe client whose system_one calls land in the call log: state, questions, answers."""

    def __init__(self, client, log: CallLog):
        self._client = client
        self._log = log

    def system_one(self, state, questions, **kwargs):
        entry = {
            "input_state": state,
            "questions": {name: msgspec.to_builtins(q) for name, q in questions.items()},
        }
        started = time.perf_counter()
        try:
            response = self._client.system_one(state=state, questions=questions, **kwargs)
        except Exception as e:
            self._log.write("jev ERROR", time.perf_counter() - started, {**entry, "error": repr(e)})
            raise
        entry["output"] = {name: msgspec.to_builtins(a) for name, a in response.answers.items()}
        entry["model"] = getattr(response, "model", None)
        entry["usage"] = msgspec.to_builtins(getattr(response, "usage", None))
        self._log.write("jev", time.perf_counter() - started, entry)
        return response


class LoggedAnthropic:
    """An Anthropic client whose messages.create calls land in the call log: request and reply."""

    def __init__(self, client, log: CallLog):
        self.messages = _LoggedMessages(client.messages, log)


class _LoggedMessages:
    def __init__(self, messages, log: CallLog):
        self._messages = messages
        self._log = log

    def create(self, **request):
        entry = {"input": {**request, "messages": [_without_image_data(m) for m in request.get("messages", [])]}}
        started = time.perf_counter()
        try:
            response = self._messages.create(**request)
        except Exception as e:
            self._log.write("anthropic ERROR", time.perf_counter() - started, {**entry, "error": repr(e)})
            raise
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        usage = getattr(response, "usage", None)
        entry["output"] = {
            "text": parsed if parsed is not None else text,
            "stop_reason": getattr(response, "stop_reason", None),
            "model": getattr(response, "model", None),
            "usage": usage.model_dump(exclude_none=True) if hasattr(usage, "model_dump") else usage,
        }
        self._log.write("anthropic", time.perf_counter() - started, entry)
        return response


def _without_image_data(message: dict) -> dict:
    """The message with each base64 image replaced by a short description; the capture is in the run folder."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    return {**message, "content": [_describe_image(block) for block in content]}


def _describe_image(block):
    source = block.get("source") if isinstance(block, dict) else None
    if not (isinstance(source, dict) and source.get("type") == "base64"):
        return block
    raw = base64.b64decode(source["data"])
    try:
        width, height = Image.open(io.BytesIO(raw)).size
        size = f"{width}x{height}, "
    except OSError:
        size = ""
    return {
        "type": "image",
        "source": f"<{source.get('media_type')} {size}{len(raw):,} bytes; the capture is saved in this run folder>",
    }
