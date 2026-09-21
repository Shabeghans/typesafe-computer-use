import threading
from types import SimpleNamespace

import numpy as np
from typesafe_sdk import ChoiceAnswer

from typesafe_computer_use import voice
from typesafe_computer_use.runner import RunConfig, RunState
from typesafe_computer_use.voice import (
    COMMIT_SILENCE,
    SAMPLE_RATE,
    Heard,
    Session,
    Transcriber,
    Transcript,
    acting_loop,
    endpoint_criteria,
    listen,
    listen_questions,
)

WORDS = ["open", "notes", "and", "once", "you're", "there", "create", "a", "new", "note"]


def speech(seconds: float) -> np.ndarray:
    return np.full(int(seconds * SAMPLE_RATE), 0.2, dtype=np.float32)


def quiet(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_endpoints_are_the_pending_speech_up_to_each_word():
    assert endpoint_criteria(["open", "notes", "and"]) == {"1": "open", "2": "open notes", "3": "open notes and"}
    assert len(endpoint_criteria(["w"] * 500)) == voice.ENDPOINT_WORDS


def test_one_call_asks_whether_to_act_and_where_the_request_ends(tmp_path):
    seen = {}

    def system_one(state, questions):
        seen.update(state=state, questions=questions)
        return SimpleNamespace(
            answers={
                "heard": ChoiceAnswer(choice="act_now", confidence=0.9, probabilities={"act_now": 0.9}),
                "ends_after": ChoiceAnswer(choice="2", confidence=0.8, probabilities={"2": 0.8}),
            }
        )

    heard = listen(SimpleNamespace(system_one=system_one), WORDS, 0, [], [], speaking=True)
    assert heard == Heard(kind="act_now", confidence=0.9, words=2)
    assert seen["state"]["speech_not_yet_acted_on"] == " ".join(WORDS)
    assert set(seen["questions"]) == {"heard", "ends_after"}
    assert set(seen["questions"]["heard"].criteria) == {"act_now", "wait_for_more", "cancel"}


def test_questions_offer_only_the_speech_not_yet_acted_on():
    criteria = listen_questions(WORDS[2:])["ends_after"].criteria
    assert criteria["1"] == "and"


def test_a_request_consumes_its_words_and_queues_them(tmp_path):
    session = Session(out=tmp_path)
    assert session.apply(Heard("act_now", 0.9, 2), WORDS) == "open notes"
    assert session.consumed == 2
    assert session.apply(Heard("act_now", 0.9, 8), WORDS) == "and once you're there create a new note"
    assert session.consumed == len(WORDS)
    assert [session.pending.get_nowait(), session.pending.get_nowait()] == [
        "open notes",
        "and once you're there create a new note",
    ]


def test_waiting_or_an_unsure_answer_changes_nothing(tmp_path):
    session = Session(out=tmp_path, act_confidence=0.7)
    assert session.apply(Heard("wait_for_more", 0.95, 3), WORDS) is None
    assert session.apply(Heard("act_now", 0.5, 2), WORDS) is None
    assert session.consumed == 0 and session.pending.empty()


def test_cancel_drops_the_queue_and_what_was_said(tmp_path):
    session = Session(out=tmp_path)
    session.apply(Heard("act_now", 0.9, 2), WORDS)
    session.apply(Heard("cancel", 0.9, 1), WORDS)
    assert session.pending.empty() and session.cancel.is_set()
    assert session.consumed == len(WORDS)


def test_the_window_is_reread_while_speaking_and_closed_after_a_pause():
    passes = []

    def transcribe(samples):
        passes.append(samples.size)
        return "open notes"

    transcript = Transcript()
    t = Transcriber(transcribe, transcript)
    t.feed(quiet(2.0), now=0.0)
    t.tick(now=0.0)
    assert not passes  # quiet alone is never transcribed

    t.feed(speech(0.1), now=2.0)
    t.tick(now=2.0)
    assert not passes  # a syllable is too little to read
    t.feed(speech(0.5), now=2.5)
    t.tick(now=2.5)
    assert transcript.snapshot()[0] == ["open", "notes"]  # an open window, still revisable
    assert passes[0] == int((voice.IDLE_KEEP + 0.6) * SAMPLE_RATE)  # the lead-in quiet was trimmed

    t.feed(quiet(0.1), now=2.6 + COMMIT_SILENCE)
    t.tick(now=2.6 + COMMIT_SILENCE)
    words, _ = transcript.snapshot()
    assert words == ["open", "notes"]
    t.feed(speech(0.6), now=5.0)
    t.tick(now=5.0 + voice.FIRST_PASS_AFTER)
    assert transcript.snapshot()[0] == ["open", "notes", "open", "notes"]  # the first window became final


def test_requests_run_in_order_sharing_history_and_the_stop_flag(tmp_path, monkeypatch):
    session = Session(out=tmp_path)
    runs = []

    def fake_run(cfg, ctx_factory, history):
        ctx = ctx_factory("typesafe", history)
        runs.append((cfg.goal, ctx, cfg.stop))
        history.append(f"did {cfg.goal}")
        if len(runs) == 2:
            session.done.set()
        return RunState(outcome="done")

    monkeypatch.setattr(voice, "run", fake_run)
    session.pending.put("open notes")
    session.pending.put("create a new note")
    acting_loop(
        session,
        lambda goal, typesafe, history: (goal, len(history)),
        lambda request, out: RunConfig(goal=request, out=out),
        say=lambda _: None,
    )
    assert [goal for goal, _, _ in runs] == ["open notes", "create a new note"]
    assert runs[1][1] == ("create a new note", 1)  # the second request saw the first one's action
    assert all(stop is session.cancel for _, _, stop in runs)


def run_one(tmp_path, monkeypatch, outcome) -> Session:
    """Queue one request whose run ends with `outcome`, and give the loop a second to react."""
    session = Session(out=tmp_path)
    monkeypatch.setattr(voice, "run", lambda cfg, factory, history: RunState(outcome=outcome))
    session.pending.put("open notes")
    worker = threading.Thread(
        target=acting_loop,
        args=(session, lambda *a: None, lambda r, o: RunConfig(goal=r, out=o)),
        kwargs={"say": lambda _: None},
        daemon=True,
    )
    worker.start()
    worker.join(timeout=1.0)
    return session


def test_a_mouse_corner_abort_ends_the_session(tmp_path, monkeypatch):
    assert run_one(tmp_path, monkeypatch, "aborted (mouse in top-left corner)").done.is_set()


def test_a_spoken_cancel_ends_only_the_request(tmp_path, monkeypatch):
    session = run_one(tmp_path, monkeypatch, "aborted (stopped by request)")
    assert not session.done.is_set()
    session.done.set()


def test_stray_marks_are_not_words():
    assert voice.words_of("Open notes. And... // create") == ["Open", "notes.", "And...", "create"]


def test_slug():
    assert voice.slug("Open Notes, please!") == "open-notes-please"
    assert voice.slug("...") == "request"
