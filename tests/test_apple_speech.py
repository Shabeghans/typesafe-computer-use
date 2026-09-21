import json

from typesafe_computer_use import apple_speech
from typesafe_computer_use.apple_speech import AppleEars
from typesafe_computer_use.voice import Transcript


def write(ears: AppleEars, *messages: dict, tail: str = "") -> None:
    with ears._log.open("a") as f:
        f.write("".join(json.dumps(m) + "\n" for m in messages) + tail)


def test_partial_lines_revise_the_window_and_final_lines_commit_it():
    transcript = Transcript()
    ears = AppleEars(transcript)
    write(ears, {"pid": 4321}, {"ready": True, "on_device": True}, {"text": "Open Kit", "final": False})
    ears.poll()
    assert ears._pid == 4321 and ears.on_device
    assert transcript.snapshot()[0] == ["Open", "Kit"]

    write(ears, {"text": "Open GitHub.", "final": True}, {"text": "and search", "final": False})
    ears.poll()
    assert transcript.snapshot()[0] == ["Open", "GitHub.", "and", "search"]
    assert ears.speaking()
    ears._pid = 0
    ears.close()


def test_a_line_still_being_written_waits_for_its_end():
    transcript = Transcript()
    ears = AppleEars(transcript)
    write(ears, tail='{"text": "Open Gi')
    ears.poll()
    assert transcript.snapshot()[0] == []
    write(ears, tail='tHub", "final": false}\n')
    ears.poll()
    assert transcript.snapshot()[0] == ["Open", "GitHub"]
    ears.close()


def test_an_error_line_is_kept_for_the_caller():
    ears = AppleEars(Transcript())
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
    assert "Zed" in words or len(words) == apple_speech.MAX_WORDS
