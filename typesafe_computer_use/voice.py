"""Voice mode: act on spoken requests while the user is still talking.

A speech recognizer keeps a transcript: macOS's own, through the JevEars helper app
(apple_speech.py), or a local Whisper model (whisper_speech.py). Every time the transcript grows,
one Jev call asks whether the speech not yet acted on already holds a complete request, and after
which word it ends. Each request found goes on a queue, and a worker thread runs the ordinary step
loop on it while listening carries on. Everything said so far is never re-acted on: a count of
consumed words moves past each request.

Whisper and the audio library are imported only when that recognizer is chosen, so the base
install does without them.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from typesafe_sdk import Choice

from .runner import RunConfig, run

COMMIT_SILENCE = 0.9  # this much quiet closes the window, and its words become final
DEBOUNCE = 0.25  # after the transcript changes, wait this long for the next word before deciding
ENDPOINT_WORDS = 60  # most words offered as where a request ends; far under the Choice ceiling
DEFAULT_ACT_CONFIDENCE = 0.5  # acting early is cheap: the loop checks the screen before each step

HEARD = {
    "act_now": (
        "The speech not yet acted on already holds a complete request that can be started now, such "
        "as opening an app or a website, or doing something in the app that is open, even when the "
        "user is still talking about what comes after it. More words for the request just acted on, "
        "such as the rest of the text to type, count once the user has stopped speaking."
    ),
    "wait_for_more": (
        "The speech not yet acted on is empty, only filler, or a request whose target or details "
        "have not been said yet, such as 'open' or 'create a new'."
    ),
    "cancel": "The user is telling the computer to stop, cancel, or never mind what it is doing.",
}


# ------------------------------------------------------------------ transcript


class Transcript:
    """Words heard so far: the final ones, plus the open window the recognizer may still revise. Thread safe."""

    def __init__(self) -> None:
        self._final: list[str] = []
        self._window: list[str] = []
        self.version = 0
        self.changed = threading.Condition()

    def set_window(self, words: list[str]) -> None:
        with self.changed:
            if words != self._window:
                self._window = words
                self.version += 1
                self.changed.notify_all()

    def commit(self, words: list[str]) -> None:
        with self.changed:
            self._final += words
            self._window = []
            self.version += 1
            self.changed.notify_all()

    def snapshot(self) -> tuple[list[str], int]:
        with self.changed:
            return self._final + self._window, self.version


def words_of(text: str) -> list[str]:
    """The words of a transcription, without stray marks such as '//' or '...'."""
    return [word for word in text.split() if any(ch.isalnum() for ch in word)]


# ------------------------------------------------------------------ the listening decision


@dataclass(frozen=True)
class Heard:
    kind: str  # act_now, wait_for_more, or cancel
    confidence: float
    words: int  # how many pending words the request spans, for act_now


def endpoint_criteria(pending: list[str]) -> dict[str, str]:
    """Each place the request could end, as the speech up to that word. Keyed by word count."""
    offered = pending[:ENDPOINT_WORDS]
    return {str(n): " ".join(offered[:n]) for n in range(1, len(offered) + 1)}


def listen_state(words: list[str], consumed: int, requests: list[str], history: list[str], speaking: bool) -> dict:
    return {
        "speech_so_far": " ".join(words),
        "speech_not_yet_acted_on": " ".join(words[consumed:]),
        "requests_already_acted_on": requests[-8:],
        "last_actions_taken": history[-8:],
        "user_still_speaking": speaking,
    }


def listen_questions(pending: list[str]) -> dict[str, Choice]:
    return {
        "heard": Choice(
            instructions=(
                "The user is speaking requests to their computer, live, and the computer acts as soon "
                "as a request is complete, while the user keeps talking. Looking only at the speech not "
                "yet acted on: can a request be started now?"
            ),
            criteria=HEARD,
        ),
        "ends_after": Choice(
            instructions=(
                "If a request can be started now, it is the speech not yet acted on from its start up "
                "to which point? Pick the shortest span that holds the whole request, so later words are "
                "left for the next one."
            ),
            criteria=endpoint_criteria(pending),
        ),
    }


def listen(client, words: list[str], consumed: int, requests: list[str], history: list[str], speaking: bool) -> Heard:
    pending = words[consumed:]
    answers = client.system_one(
        state=listen_state(words, consumed, requests, history, speaking), questions=listen_questions(pending)
    ).answers
    heard, ends = answers["heard"], answers["ends_after"]
    return Heard(kind=heard.choice, confidence=heard.confidence, words=int(ends.choice))


# ------------------------------------------------------------------ the session


def slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "request"


@dataclass
class Session:
    """The shared state of one voice session: the transcript, the requests found, and the stop flags."""

    out: Path
    act_confidence: float = DEFAULT_ACT_CONFIDENCE
    transcript: Transcript = field(default_factory=Transcript)
    consumed: int = 0  # transcript words already turned into requests, or dropped by a cancel
    requests: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)  # actions across every request, oldest first
    pending: queue.Queue = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)  # ends the running request
    done: threading.Event = field(default_factory=threading.Event)  # ends the session

    def apply(self, heard: Heard, words: list[str]) -> str | None:
        """Act on one listening decision. Returns the request queued, if any."""
        if heard.confidence < self.act_confidence:
            return None
        if heard.kind == "cancel":
            self.consumed = len(words)
            while not self.pending.empty():
                self.pending.get_nowait()
            self.cancel.set()
            return None
        if heard.kind != "act_now":
            return None
        span = words[self.consumed : self.consumed + max(1, heard.words)]
        self.consumed += len(span)
        request = " ".join(span)
        self.requests.append(request)
        self.pending.put(request)
        return request


def listening_loop(session: Session, client, speaking: Callable[[], bool], say: Callable[[str], None] = print) -> None:
    """Decide on each new stretch of speech until the session ends."""
    seen = -1
    while not session.done.is_set():
        with session.transcript.changed:
            session.transcript.changed.wait_for(lambda seen=seen: session.transcript.version != seen, timeout=0.2)
        if session.transcript.version == seen:
            continue
        time.sleep(DEBOUNCE)
        words, seen = session.transcript.snapshot()
        if len(words) <= session.consumed:
            continue
        heard = listen(client, words, session.consumed, session.requests, session.history, speaking())
        pending = " ".join(words[session.consumed :])
        request = session.apply(heard, words)
        verdict = f"{heard.kind} ({heard.confidence:.2f})"
        if request:
            verdict += f': "{request}"'
        say(f'heard: "{pending}"  ->  {verdict}')
        if heard.kind == "cancel" and session.cancel.is_set():
            say("cancelling the current request")


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
        state = run(cfg, lambda typesafe, history, goal=request: ctx_factory(goal, typesafe, history), history=session.history)
        if state.outcome.startswith("aborted") and "stopped by request" not in state.outcome:
            say("mouse in the corner: ending the voice session")
            session.done.set()


def ears_for(name: str, transcript: Transcript, whisper_model: str | None = None):
    """The speech recognizer: "apple" (macOS's own) or "whisper" (a local model)."""
    if name == "apple":
        from .apple_speech import AppleEars

        return AppleEars(transcript)
    from .whisper_speech import DEFAULT_MODEL, WhisperEars

    return WhisperEars(transcript, whisper_model or DEFAULT_MODEL)


def start(session: Session, client, ctx_factory, make_config: Callable[[str, Path], RunConfig], ears) -> None:
    """Listen until Ctrl-C or the mouse-corner abort.

    `ears` has start(), run(done) to feed the transcript, speaking(), describe(), and close().
    """
    session.out.mkdir(parents=True, exist_ok=True)
    try:
        ears.start()
        threads = [
            threading.Thread(target=ears.run, args=(session.done,), name="hear", daemon=True),
            threading.Thread(target=listening_loop, args=(session, client, ears.speaking), name="listen", daemon=True),
            threading.Thread(target=acting_loop, args=(session, ctx_factory, make_config), name="act", daemon=True),
        ]
        for thread in threads:
            thread.start()
        print(
            f"listening with {ears.describe()}. speak your requests; Ctrl-C or the mouse in the top-left corner ends the session."
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
        ears.close()
    print(f"session folder: {session.out}")
