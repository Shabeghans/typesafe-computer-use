"""Voice mode: act on spoken requests while the user is still talking.

The microphone feeds a local Whisper model, which keeps a transcript. Every time the transcript
grows, one Jev call asks whether the speech not yet acted on already holds a complete request,
and after which word it ends. Each request found goes on a queue, and a worker thread runs the
ordinary step loop on it while listening carries on. Everything said so far is never re-acted on:
a count of consumed words moves past each request.

Whisper and the audio library are imported only when voice mode starts, so the base install
does without them.
"""

from __future__ import annotations

import contextlib
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from typesafe_sdk import Choice

from .runner import RunConfig, run

SAMPLE_RATE = 16_000  # what Whisper reads
BLOCK_SECONDS = 0.1  # one microphone callback
TRANSCRIBE_EVERY = 0.7  # seconds between passes over the open window
SILENCE_RMS = 0.01  # a block quieter than this is not speech
COMMIT_SILENCE = 0.9  # this much quiet closes the window, and its words become final
MAX_WINDOW = 20.0  # a window never grows past this, so a pass stays quick
IDLE_KEEP = 0.5  # seconds of quiet kept before speech starts, so its first syllable is not cut
FIRST_PASS_AFTER = 0.5  # seconds of speech before the first pass: Whisper invents words from a syllable
DEBOUNCE = 0.25  # after the transcript changes, wait this long for the next word before deciding
ENDPOINT_WORDS = 60  # most words offered as where a request ends; far under the Choice ceiling
DEFAULT_ACT_CONFIDENCE = 0.5  # acting early is cheap: the loop checks the screen before each step
DEFAULT_WHISPER_MODEL = "base.en"

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
    """Words heard so far: the final ones, plus the open window Whisper may still revise. Thread safe."""

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
    """The words of a transcription, without the stray marks Whisper emits, such as '//' or '...'."""
    return [word for word in text.split() if any(ch.isalnum() for ch in word)]


class Transcriber:
    """Turns audio blocks into transcript updates. `transcribe(samples) -> text` is the model."""

    def __init__(self, transcribe: Callable[[np.ndarray], str], transcript: Transcript):
        self._transcribe = transcribe
        self.transcript = transcript
        self._window = np.zeros(0, dtype=np.float32)
        self._voiced = False  # the window holds speech, not just the lead-in quiet
        self.last_voice = 0.0
        self._next_pass = 0.0

    def feed(self, samples: np.ndarray, now: float) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size and float(np.sqrt(np.mean(samples**2))) >= SILENCE_RMS:
            if not self._voiced:
                self._next_pass = now + FIRST_PASS_AFTER
            self._voiced = True
            self.last_voice = now
        self._window = np.concatenate([self._window, samples])
        if not self._voiced:
            self._window = self._window[-int(IDLE_KEEP * SAMPLE_RATE) :]

    def tick(self, now: float) -> None:
        """Re-read the open window when due, and close it after a pause or when it gets long."""
        if not self._voiced:
            return
        seconds = self._window.size / SAMPLE_RATE
        if now - self.last_voice >= COMMIT_SILENCE or seconds >= MAX_WINDOW:
            self.transcript.commit(words_of(self._transcribe(self._window)))
            self._window = np.zeros(0, dtype=np.float32)
            self._voiced = False
        elif now >= self._next_pass:
            self.transcript.set_window(words_of(self._transcribe(self._window)))
            self._next_pass = now + TRANSCRIBE_EVERY


def whisper(model_name: str) -> Callable[[np.ndarray], str]:
    """A local Whisper model as `samples -> text`. The first use downloads the model."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(samples: np.ndarray) -> str:
        segments, _ = model.transcribe(samples, language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False)
        return " ".join(segment.text.strip() for segment in segments).strip()

    return transcribe


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


def start(
    session: Session,
    client,
    ctx_factory,
    make_config: Callable[[str, Path], RunConfig],
    whisper_model: str = DEFAULT_WHISPER_MODEL,
) -> None:
    """Open the microphone and run until Ctrl-C or the mouse-corner abort."""
    import sounddevice as sd

    session.out.mkdir(parents=True, exist_ok=True)
    print(f"loading the {whisper_model} speech model (downloaded on first use)...")
    transcriber = Transcriber(whisper(whisper_model), session.transcript)
    blocks: queue.Queue = queue.Queue()

    def on_audio(indata, frames, when, status) -> None:
        blocks.put(indata[:, 0].copy())

    def transcribing() -> None:
        while not session.done.is_set():
            with contextlib.suppress(queue.Empty):
                transcriber.feed(blocks.get(timeout=0.1), time.monotonic())
            transcriber.tick(time.monotonic())

    def speaking() -> bool:
        return time.monotonic() - transcriber.last_voice < COMMIT_SILENCE

    threads = [
        threading.Thread(target=transcribing, name="transcribe", daemon=True),
        threading.Thread(target=listening_loop, args=(session, client, speaking), name="listen", daemon=True),
        threading.Thread(target=acting_loop, args=(session, ctx_factory, make_config), name="act", daemon=True),
    ]
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=int(BLOCK_SECONDS * SAMPLE_RATE), callback=on_audio
    )
    with stream:
        for thread in threads:
            thread.start()
        print("listening. speak your requests; Ctrl-C or the mouse in the top-left corner ends the session.")
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
    print(f"session folder: {session.out}")
