"""macOS speech recognition for voice mode, through the JevEars helper app.

macOS lets a process use speech recognition only if its app declares it, and a Python started from
a terminal counts as the terminal, which does not. So the recognizer runs in a small Swift app,
built on first use with the Xcode command line tools and launched with `open`, which makes it its
own app with its own permissions. Signals turn its microphone on and off, and it appends what
it hears to a file as JSON lines (see JevEars/JevEars.swift), which `AppleEars` follows.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

SOURCE = Path(__file__).parent / "JevEars"
APP = Path.home() / "Library" / "Application Support" / "typesafe-computer-use" / "JevEars.app"
READY_TIMEOUT = 120.0  # long enough to answer the permission prompts on first use

# Names the recognizer should expect: what people open, beyond the installed apps.
SITES = [
    "GitHub", "Gmail", "Google", "Google Docs", "Google Drive", "Google Calendar", "YouTube", "Slack",
    "Notion", "Figma", "Linear", "LinkedIn", "Reddit", "ChatGPT", "Claude", "Spotify", "WhatsApp",
    "Discord", "Zoom", "Jira", "Confluence", "Amazon", "Netflix", "Wikipedia", "Hacker News",
]  # fmt: skip
MAX_WORDS = 100  # contextual strings past this help little


class EarsError(RuntimeError):
    pass


def vocabulary() -> list[str]:
    """The sites above, then the installed app names."""
    names = list(SITES)
    for folder in (Path("/Applications"), Path("/System/Applications"), Path.home() / "Applications"):
        if folder.is_dir():
            names += sorted(app.stem for app in folder.glob("*.app"))
    return list(dict.fromkeys(names))[:MAX_WORDS]


def source_hash() -> str:
    digest = hashlib.sha256()
    for name in ("JevEars.swift", "Info.plist"):
        digest.update((SOURCE / name).read_bytes())
    return digest.hexdigest()


def build(app: Path = APP) -> Path:
    """Compile the helper app, unless the one there was built from the current source.

    Each rebuild changes its signature, so macOS asks for its permissions again.
    """
    stamp = app / "Contents" / "Resources" / "source.sha256"
    wanted = source_hash()
    if stamp.is_file() and stamp.read_text() == wanted:
        return app
    if shutil.which("xcrun") is None:
        raise EarsError("building JevEars needs the Xcode command line tools: xcode-select --install")
    print("building the JevEars speech helper (once)...")
    shutil.rmtree(app, ignore_errors=True)
    (app / "Contents" / "MacOS").mkdir(parents=True)
    stamp.parent.mkdir(parents=True)
    shutil.copy(SOURCE / "Info.plist", app / "Contents" / "Info.plist")
    binary = app / "Contents" / "MacOS" / "JevEars"
    steps = [
        ["xcrun", "swiftc", "-O", "-swift-version", "5", "-o", str(binary), str(SOURCE / "JevEars.swift")],
        ["codesign", "--force", "--sign", "-", str(app)],
    ]
    for step in steps:
        done = subprocess.run(step, capture_output=True, text=True)
        if done.returncode != 0:
            raise EarsError(f"building JevEars failed:\n{done.stderr.strip()}")
    stamp.write_text(wanted)
    return app


class AppleEars:
    """Runs JevEars. `hold()` turns its microphone on and `release()` off; each utterance heard in
    between goes to `on_utterance(text)` once the recognizer is done, and `on_partial(text)` gets it
    as it is heard."""

    def __init__(self, on_utterance: Callable[[str], None], on_partial: Callable[[str], None] = lambda text: None):
        self.on_utterance = on_utterance
        self.on_partial = on_partial
        self.on_device = False
        self._pid = 0
        self._folder = Path(tempfile.mkdtemp(prefix="jev-ears-"))
        self._log = self._folder / "heard.jsonl"
        self._log.touch()
        self._offset = 0
        self._partial = ""
        self._ready = threading.Event()
        self._error = ""

    def start(self) -> None:
        app = build()
        words = self._folder / "words.txt"
        words.write_text("\n".join(vocabulary()))
        args = ["--out", str(self._log), "--parent", str(os.getpid()), "--words", str(words)]
        # -n: a fresh instance; -g: in the background, so the app being driven keeps the focus.
        subprocess.run(["open", "-n", "-g", str(app), "--args", *args], check=True)
        started = time.monotonic()
        hinted = False
        while not self._ready.is_set():
            self.poll()
            if self._error:
                raise EarsError(self._error)
            waited = time.monotonic() - started
            if not hinted and waited > 3:
                print("allow JevEars to use the microphone and speech recognition if macOS asks...")
                hinted = True
            if waited > READY_TIMEOUT:
                raise EarsError(f"JevEars did not start; its output is in {self._log}")
            time.sleep(0.05)

    def hold(self) -> None:
        self._signal(signal.SIGUSR1)

    def release(self) -> None:
        self._signal(signal.SIGUSR2)

    def _signal(self, number: int) -> None:
        if self._pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(self._pid, number)

    def poll(self) -> None:
        """Read whatever JevEars has written since the last poll."""
        with self._log.open("rb") as f:
            f.seek(self._offset)
            chunk = f.read()
        self._offset += len(chunk)
        lines = (self._partial + chunk.decode("utf-8", "replace")).split("\n")
        self._partial = lines.pop()  # the unfinished end of a line still being written
        for line in lines:
            if line.strip():
                self.handle(json.loads(line))

    def handle(self, message: dict) -> None:
        if "pid" in message:
            self._pid = int(message["pid"])
        elif "ready" in message:
            self.on_device = bool(message.get("on_device"))
            self._ready.set()
        elif "error" in message:
            self._error = str(message["error"])
        elif "text" in message:
            text = str(message["text"]).strip()
            if message.get("final"):
                self.on_utterance(text)
            else:
                self.on_partial(text)

    def run(self, done: threading.Event) -> None:
        while not done.is_set():
            self.poll()
            if self._error:
                print(f"speech recognition stopped: {self._error}")
                done.set()
            time.sleep(0.05)

    def describe(self) -> str:
        return "macOS speech recognition" + (" (on this Mac)" if self.on_device else " (Apple's servers)")

    def close(self) -> None:
        self._signal(signal.SIGTERM)
        shutil.rmtree(self._folder, ignore_errors=True)
