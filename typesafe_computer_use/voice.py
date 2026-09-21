"""Voice mode: push to talk, and each thing said is done.

Holding the key turns the microphone on; letting go turns it off, and what was said in between is
one request. macOS's speech recognition hears it, through the JevEars helper app (apple_speech.py).
Each request goes on a queue, and a worker thread runs the ordinary step loop on it, so the key can
be held again for the next one while the last is still being done. Saying only "stop" or "cancel"
ends the request being done and drops the queue.

Requests end without the writer's final answer: the step loop says itself whether Jev judged the
request done, so no screenshot goes to a model at the end.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .runner import STOPPED, RunConfig, run

CANCEL = re.compile(r"(stop|cancel|never ?mind)( it| that)?", re.IGNORECASE)


def is_cancel(text: str) -> bool:
    return CANCEL.fullmatch(re.sub(r"[^\w\s]", "", text).strip()) is not None


def slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "request"


def verdict(outcome: str) -> str:
    """How a request's run ended, for the user."""
    if outcome == "done":
        return "done: Jev judged the request complete"
    if outcome == "dry run":
        return "dry run: showed the first step and did nothing (pass --act to drive the machine)"
    if outcome == "aborted (stopped by request)":
        return "cancelled"
    return f"not done: {STOPPED.get(outcome, outcome)}"


@dataclass
class Session:
    """The shared state of one voice session: the requests heard, and the stop flags."""

    out: Path
    requests: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)  # actions across every request, oldest first
    pending: queue.Queue = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)  # ends the running request
    done: threading.Event = field(default_factory=threading.Event)  # ends the session

    def heard(self, text: str, say: Callable[[str], None] = print) -> None:
        """One utterance, from key down to key up."""
        if not text:
            say("heard nothing")
        elif is_cancel(text):
            while not self.pending.empty():
                self.pending.get_nowait()
            self.cancel.set()
            say(f'heard: "{text}"  ->  cancelling')
        else:
            self.requests.append(text)
            self.pending.put(text)
            say(f'heard: "{text}"')


def acting_loop(session: Session, ctx_factory, make_config: Callable[[str, Path], RunConfig], say=print) -> None:
    """Run each queued request through the step loop, one at a time, sharing the action history.

    `ctx_factory(goal, typesafe, history)` builds the action Context for one request.
    """
    count = 0
    while not session.done.is_set():
        try:
            request = session.pending.get(timeout=0.2)
        except queue.Empty:
            continue
        count += 1
        session.cancel.clear()  # a cancel said while idle has already emptied the queue
        say(f'\n>>> request {count}: "{request}"')
        cfg = make_config(request, session.out / f"{count:02d}-{slug(request)}")
        cfg.stop = session.cancel
        cfg.answer = False
        state = run(cfg, lambda typesafe, history, goal=request: ctx_factory(goal, typesafe, history), history=session.history)
        say(f'<<< request {count}, "{request}": {verdict(state.outcome)}')
        if state.outcome.startswith("aborted") and "stopped by request" not in state.outcome:
            say("mouse in the corner: ending the voice session")
            session.done.set()


def start(session: Session, ctx_factory, make_config: Callable[[str, Path], RunConfig], ears, key) -> None:
    """Listen until Ctrl-C or the mouse-corner abort.

    `ears` has start(), hold(), release(), run(done), describe() and close(), and reports each
    utterance to session.heard. `key` calls ears.hold and ears.release; it has start() and stop().
    """
    session.out.mkdir(parents=True, exist_ok=True)
    try:
        ears.start()
        key.start()
        threads = [
            threading.Thread(target=ears.run, args=(session.done,), name="hear", daemon=True),
            threading.Thread(target=acting_loop, args=(session, ctx_factory, make_config), name="act", daemon=True),
        ]
        for thread in threads:
            thread.start()
        print(
            f"ready, with {ears.describe()}. hold {key.key.replace('_', ' ')} and speak a request, "
            "then let go. Ctrl-C or the mouse in the top-left corner ends the session."
        )
        try:
            while not session.done.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nending the voice session")
        finally:
            session.cancel.set()
            session.done.set()
        for thread in threads:
            thread.join(timeout=5)
    finally:
        key.stop()
        ears.close()
    print(f"session folder: {session.out}")
