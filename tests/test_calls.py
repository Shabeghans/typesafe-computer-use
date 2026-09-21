import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer

from typesafe_computer_use.calls import CallLog, LoggedAnthropic, LoggedTypeSafe


class FakeTypeSafe:
    def __init__(self, answers=None, error=None):
        self.answers = answers or {}
        self.error = error

    def system_one(self, state, questions):
        if self.error:
            raise self.error
        return SimpleNamespace(model="jev-latest", usage=None, answers=self.answers)


def entries(path) -> list[dict]:
    """The JSON body of each entry, in order."""
    blocks = path.read_text().split("=" * 78 + "\n")
    return [json.loads(body) for body in blocks[2::2]]


def test_a_jev_call_logs_state_choices_and_answers(tmp_path):
    log = CallLog(tmp_path / "calls.log")
    log.step = "3"
    kind = ChoiceAnswer(choice="scroll_down", confidence=0.8, probabilities={"scroll_down": 0.8, "done": 0.2})
    ok = NoulAnswer(noul=0.93)
    client = LoggedTypeSafe(FakeTypeSafe({"kind": kind, "ok": ok}), log)

    questions = {
        "kind": Choice(instructions="which action?", criteria={"scroll_down": "Scroll down.", "done": "Goal met."}),
        "ok": Noul(instructions="did it work?"),
    }
    response = client.system_one(state={"goal": "g", "screen_items_in_reading_order": []}, questions=questions)

    assert response.answers["kind"] is kind  # the caller gets the real response back
    text = (tmp_path / "calls.log").read_text()
    assert "#1  step 3  jev" in text
    [entry] = entries(tmp_path / "calls.log")
    assert entry["input_state"]["goal"] == "g"
    assert entry["questions"]["kind"]["criteria"] == {"scroll_down": "Scroll down.", "done": "Goal met."}
    assert entry["output"]["kind"]["choice"] == "scroll_down"
    assert entry["output"]["kind"]["probabilities"] == {"scroll_down": 0.8, "done": 0.2}
    assert entry["output"]["ok"]["noul"] == 0.93


def test_a_failed_jev_call_is_logged_and_still_raises(tmp_path):
    log = CallLog(tmp_path / "calls.log")
    client = LoggedTypeSafe(FakeTypeSafe(error=RuntimeError("503")), log)
    with pytest.raises(RuntimeError):
        client.system_one(state={}, questions={"ok": Noul(instructions="?")})
    assert "jev ERROR" in (tmp_path / "calls.log").read_text()
    assert "503" in entries(tmp_path / "calls.log")[0]["error"]


def test_an_anthropic_call_logs_request_and_reply_without_the_image_bytes(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20)).save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode()
    reply = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"achieved": true, "answer": "done"}')],
        stop_reason="end_turn",
        model="anthropic/claude-sonnet-5",
        usage=None,
    )
    inner = SimpleNamespace(messages=SimpleNamespace(create=lambda **request: reply))
    client = LoggedAnthropic(inner, CallLog(tmp_path / "calls.log"))

    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
        {"type": "text", "text": '{"goal": "g"}'},
    ]
    assert client.messages.create(model="m", system="sys", messages=[{"role": "user", "content": content}]) is reply

    text = (tmp_path / "calls.log").read_text()
    assert data not in text
    [entry] = entries(tmp_path / "calls.log")
    image, prompt = entry["input"]["messages"][0]["content"]
    assert "image/png 40x20" in image["source"]
    assert prompt["text"] == '{"goal": "g"}'
    assert entry["input"]["system"] == "sys"
    assert entry["output"]["text"] == {"achieved": True, "answer": "done"}
    assert entry["output"]["stop_reason"] == "end_turn"
