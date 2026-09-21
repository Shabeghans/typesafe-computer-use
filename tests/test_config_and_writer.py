import os

from typesafe_computer_use.config import OPENROUTER_BASE_URL, decision_client, load_dotenv, writer_client, writer_model
from typesafe_computer_use.writer import make_writer, valid_url


def test_dotenv_sets_only_missing_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("CLICKER_TEST_PRESENT", "keep")
    monkeypatch.delenv("CLICKER_TEST_NEW", raising=False)
    (tmp_path / ".env").write_text('# comment\nCLICKER_TEST_PRESENT=override\nCLICKER_TEST_NEW="quoted value"\nbroken line\n')
    load_dotenv(tmp_path / ".env")
    assert os.environ["CLICKER_TEST_PRESENT"] == "keep"
    assert os.environ["CLICKER_TEST_NEW"] == "quoted value"


def test_dotenv_missing_file_is_fine(tmp_path):
    load_dotenv(tmp_path / "nope.env")


def test_decision_client_prefers_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert decision_client() == {"api_key": "sk-or-test", "base_url": OPENROUTER_BASE_URL}


def test_decision_client_falls_back_to_typesafe(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert decision_client() == {}


def test_writer_goes_through_openrouter_with_its_model_names(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.delenv("CLICKER_WRITER_MODEL", raising=False)
    assert writer_client() == {"auth_token": "sk-or-test", "base_url": OPENROUTER_BASE_URL}
    assert writer_model() == "anthropic/claude-haiku-4.5"
    client = make_writer()
    assert client.api_key is None and client.auth_token == "sk-or-test"  # the Anthropic key is never sent


def test_writer_falls_back_to_anthropic(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CLICKER_WRITER_MODEL", raising=False)
    assert writer_client() == {}
    assert writer_model() == "claude-haiku-4-5"


def test_valid_url():
    assert valid_url("https://www.cnn.com")
    assert valid_url("https://news.ycombinator.com/newest")
    assert not valid_url("http://www.cnn.com")
    assert not valid_url("https://localhost")
    assert not valid_url("https://www.cnn.com/a b")
    assert not valid_url("")
