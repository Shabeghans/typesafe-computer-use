import threading

from typesafe_computer_use import voice
from typesafe_computer_use.runner import RunConfig, RunState
from typesafe_computer_use.voice import Session, acting_loop, is_cancel, verdict


def quiet(_: str) -> None:
    pass


def test_each_utterance_is_one_request(tmp_path):
    session = Session(out=tmp_path)
    session.heard("Open GitHub.", say=quiet)
    session.heard("Then search for typesafe.", say=quiet)
    assert [session.pending.get_nowait(), session.pending.get_nowait()] == ["Open GitHub.", "Then search for typesafe."]


def test_nothing_heard_queues_nothing(tmp_path):
    session = Session(out=tmp_path)
    session.heard("", say=quiet)
    assert session.pending.empty() and not session.requests


def test_a_lone_stop_cancels_and_drops_the_queue(tmp_path):
    session = Session(out=tmp_path)
    session.heard("Open GitHub.", say=quiet)
    session.heard("Stop!", say=quiet)
    assert session.pending.empty() and session.cancel.is_set()


def test_only_a_bare_cancel_counts():
    assert is_cancel("Cancel.") and is_cancel("never mind") and is_cancel("Stop that")
    assert not is_cancel("stop the timer") and not is_cancel("cancel my 3pm meeting")


def test_requests_run_in_order_sharing_history_without_the_final_answer(tmp_path, monkeypatch):
    session = Session(out=tmp_path)
    runs, said = [], []

    def fake_run(cfg, ctx_factory, history):
        ctx = ctx_factory("typesafe", history)
        runs.append((cfg.goal, ctx, cfg.stop, cfg.answer))
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
        say=said.append,
    )
    assert [goal for goal, *_ in runs] == ["open notes", "create a new note"]
    assert runs[1][1] == ("create a new note", 1)  # the second request saw the first one's action
    assert all(stop is session.cancel and answer is False for _, _, stop, answer in runs)
    assert any("done: Jev judged the request complete" in line for line in said)


def test_verdicts():
    assert verdict("done").startswith("done")
    assert verdict("aborted (stopped by request)") == "cancelled"
    assert verdict("stalled") == "not done: the last actions changed nothing"


def run_one(tmp_path, monkeypatch, outcome) -> Session:
    """Queue one request whose run ends with `outcome`, and give the loop a second to react."""
    session = Session(out=tmp_path)
    monkeypatch.setattr(voice, "run", lambda cfg, factory, history: RunState(outcome=outcome))
    session.pending.put("open notes")
    worker = threading.Thread(
        target=acting_loop,
        args=(session, lambda *a: None, lambda r, o: RunConfig(goal=r, out=o)),
        kwargs={"say": quiet},
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


def test_slug():
    assert voice.slug("Open Notes, please!") == "open-notes-please"
    assert voice.slug("...") == "request"
