import json

from typesafe_computer_use import apple_speech
from typesafe_computer_use.apple_speech import AppleEars


def listening() -> tuple[AppleEars, list, list]:
    utterances, partials = [], []
    return AppleEars(on_utterance=utterances.append, on_partial=partials.append), utterances, partials


def write(ears: AppleEars, *messages: dict, tail: str = "") -> None:
    with ears._log.open("a") as f:
        f.write("".join(json.dumps(m) + "\n" for m in messages) + tail)


def test_partials_show_progress_and_the_final_line_is_the_utterance():
    ears, utterances, partials = listening()
    write(ears, {"pid": 4321}, {"ready": True, "on_device": True}, {"text": "Open Git", "final": False})
    ears.poll()
    assert ears._pid == 4321 and ears.on_device
    assert partials == ["Open Git"] and utterances == []

    write(ears, {"text": "Open GitHub.", "final": True}, {"text": "", "final": True})
    ears.poll()
    assert utterances == ["Open GitHub.", ""]
    ears._pid = 0
    ears.close()


def test_a_line_still_being_written_waits_for_its_end():
    ears, utterances, _ = listening()
    write(ears, tail='{"text": "Open Gi')
    ears.poll()
    assert utterances == []
    write(ears, tail='tHub", "final": true}\n')
    ears.poll()
    assert utterances == ["Open GitHub"]
    ears.close()


def test_hold_and_release_signal_the_helper(monkeypatch):
    ears, _, _ = listening()
    sent = []
    monkeypatch.setattr(apple_speech.os, "kill", lambda pid, number: sent.append((pid, number)))
    ears.hold()
    assert sent == []  # no helper yet
    ears._pid = 99
    ears.hold()
    ears.release()
    assert sent == [(99, apple_speech.signal.SIGUSR1), (99, apple_speech.signal.SIGUSR2)]
    ears.close()


def test_an_error_line_is_kept_for_the_caller():
    ears, _, _ = listening()
    write(ears, {"error": "speech recognition is not allowed"})
    ears.poll()
    assert ears._error == "speech recognition is not allowed"
    ears.close()


def test_the_vocabulary_leads_with_sites_and_has_no_repeats(tmp_path, monkeypatch):
    apps = tmp_path / "Applications"
    for name in ("Slack", "Notes", "Zed"):
        (apps / f"{name}.app").mkdir(parents=True)
    monkeypatch.setattr(apple_speech.Path, "home", lambda: tmp_path)
    words = apple_speech.vocabulary()
    assert words[0] == "GitHub"
    assert len(words) == len(set(words)) <= apple_speech.MAX_WORDS
