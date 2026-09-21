"""Local Whisper speech recognition for voice mode: `clicker-voice --ears whisper`.

The microphone feeds a Whisper model through sounddevice. The open window of speech is re-read
every so often while the user talks, and made final after a pause. Needs the voice extras.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from .voice import COMMIT_SILENCE, Transcript, words_of

SAMPLE_RATE = 16_000  # what Whisper reads
BLOCK_SECONDS = 0.1  # one microphone callback
TRANSCRIBE_EVERY = 0.7  # seconds between passes over the open window
SILENCE_RMS = 0.01  # a block quieter than this is not speech
MAX_WINDOW = 20.0  # a window never grows past this, so a pass stays quick
IDLE_KEEP = 0.5  # seconds of quiet kept before speech starts, so its first syllable is not cut
FIRST_PASS_AFTER = 0.5  # seconds of speech before the first pass: Whisper invents words from a syllable
DEFAULT_MODEL = "base.en"


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


class WhisperEars:
    """The microphone through a local Whisper model, feeding the transcript."""

    def __init__(self, transcript: Transcript, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.transcript = transcript
        self._blocks: queue.Queue = queue.Queue()
        self._transcriber: Transcriber | None = None
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        print(f"loading the {self.model_name} speech model (downloaded on first use)...")
        self._transcriber = Transcriber(whisper(self.model_name), self.transcript)

        def on_audio(indata, frames, when, status) -> None:
            self._blocks.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=int(BLOCK_SECONDS * SAMPLE_RATE), callback=on_audio
        )
        self._stream.start()

    def run(self, done: threading.Event) -> None:
        while not done.is_set():
            with contextlib.suppress(queue.Empty):
                self._transcriber.feed(self._blocks.get(timeout=0.1), time.monotonic())
            self._transcriber.tick(time.monotonic())

    def speaking(self) -> bool:
        return self._transcriber is not None and time.monotonic() - self._transcriber.last_voice < COMMIT_SILENCE

    def describe(self) -> str:
        return f"Whisper {self.model_name}"

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
