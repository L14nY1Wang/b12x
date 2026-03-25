"""Tests for the HTTP API server.

No GPU required. Uses a mock engine to verify routing, request
parsing, and response format.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from serve.api.server import ServingApp
from serve.engine.serving import GenerationResult


class _FakeInputIds:
    """Mimics tokenizer(...).input_ids[0].tolist() chain."""
    def __init__(self):
        self.input_ids = self
    def __getitem__(self, idx):
        return self
    def tolist(self):
        return [1, 2, 3]


def _make_mock_engine():
    """Create a mock ServingEngine with enough interface for API tests."""
    engine = MagicMock()
    engine.model_path = "/data/models/test-model"
    engine.scheduler.stats = {
        "waiting": 0, "prefilling": 0, "running": 0,
        "finished": 5, "prefill_steps": 10, "decode_steps": 50,
        "preemptions": 0, "cache_pages": 100, "free_pages": 200,
    }
    engine.tokenizer = MagicMock()
    engine.tokenizer.return_value = _FakeInputIds()
    engine.tokenizer.apply_chat_template.return_value = "formatted prompt"

    # submit() returns a mock request that's immediately done.
    mock_req = MagicMock()
    mock_req.output_ids = [10, 20, 30]
    mock_req._done_event.wait.return_value = None
    engine.submit.return_value = mock_req
    engine._to_result.return_value = GenerationResult(
        request_id=0,
        prompt_ids=[1, 2, 3],
        generated_ids=[10, 20, 30],
        finished=True,
        finish_reason="stop",
        time_to_first_token_ms=50.0,
        total_time_ms=200.0,
    )
    engine.tokenizer.decode.return_value = "hello world"
    engine._ensure_stop_ids = MagicMock()
    engine.server_loop_health.return_value = {
        "running": True,
        "alive": True,
        "healthy": True,
        "last_error": None,
    }

    return engine


@pytest.fixture
def client():
    engine = _make_mock_engine()
    app = ServingApp(engine)
    test_client = TestClient(app.app)
    test_client.engine = engine
    return test_client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "test-model"
    assert "scheduler" in data
    assert data["server_loop"]["healthy"] is True
    assert data["scheduler"]["finished"] == 5


def test_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"


def test_completions(client):
    resp = client.post("/v1/completions", json={
        "prompt": "Hello",
        "max_tokens": 10,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) == 1
    assert data["choices"][0]["text"] == "hello world"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] == 3
    assert data["usage"]["completion_tokens"] == 3


def test_chat_completions(client):
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "hello world"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_completions_with_timeout(client):
    resp = client.post("/v1/completions", json={
        "prompt": "Hello",
        "max_tokens": 10,
        "timeout": 30.0,
    })
    assert resp.status_code == 200
    # Verify timeout was passed through.
    engine = client.app.extra.get("engine") or client.app
    # Just verify the response is valid.
    data = resp.json()
    assert "choices" in data


def test_completions_with_sampling_params(client):
    resp = client.post("/v1/completions", json={
        "prompt": "Hello",
        "max_tokens": 10,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data


def test_error_handling():
    """Engine exceptions return structured error response."""
    engine = _make_mock_engine()
    engine.submit.side_effect = RuntimeError("GPU on fire")
    app = ServingApp(engine)
    client = TestClient(app.app, raise_server_exceptions=False)

    resp = client.post("/v1/completions", json={
        "prompt": "Hello",
        "max_tokens": 10,
    })
    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data
    assert "GPU on fire" in data["error"]["message"]
    assert data["error"]["type"] == "RuntimeError"


def test_stop_strings_converted_to_ids(client):
    """Stop strings are converted to full token sequences."""
    client.engine.tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
        "\n": [13],
        "END": [42, 43, 44],
    }[text]

    resp = client.post("/v1/completions", json={
        "prompt": "Hello",
        "max_tokens": 10,
        "stop": ["\n", "END"],
    })
    assert resp.status_code == 200
    submit_args = client.engine.submit.call_args
    params = submit_args.args[1]
    assert params.stop_sequences == [[13], [42, 43, 44]]
    assert params.stop_token_ids is None
    data = resp.json()
    assert "choices" in data


def test_health_shows_scheduler_stats(client):
    """Health endpoint returns scheduler stats."""
    resp = client.get("/health")
    data = resp.json()
    sched = data["scheduler"]
    assert "waiting" in sched
    assert "running" in sched
    assert "prefilling" in sched
    assert "finished" in sched
    assert "prefill_steps" in sched
    assert "decode_steps" in sched
    assert "preemptions" in sched
    assert "cache_pages" in sched
    assert "free_pages" in sched


def test_health_reports_unhealthy_loop():
    engine = _make_mock_engine()
    engine.server_loop_health.return_value = {
        "running": False,
        "alive": False,
        "healthy": False,
        "last_error": "RuntimeError: boom",
    }
    app = ServingApp(engine)
    client = TestClient(app.app)

    resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "error"
    assert data["server_loop"]["last_error"] == "RuntimeError: boom"


def test_models_response_format(client):
    """Models endpoint matches OpenAI format."""
    resp = client.get("/v1/models")
    data = resp.json()
    assert data["object"] == "list"
    model = data["data"][0]
    assert "id" in model
    assert "object" in model
    assert model["object"] == "model"


def test_chat_response_has_usage(client):
    """Chat response includes token usage stats."""
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
    })
    data = resp.json()
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] == 3
    assert data["usage"]["completion_tokens"] == 3
    assert data["usage"]["total_tokens"] == 6


def test_completion_response_has_id(client):
    """Completion response has a unique ID."""
    resp = client.post("/v1/completions", json={
        "prompt": "Hello",
        "max_tokens": 5,
    })
    data = resp.json()
    assert data["id"].startswith("cmpl-")
    assert data["object"] == "text_completion"
    assert "created" in data
    assert "model" in data


def test_overload_returns_503():
    """Queue full returns 503 Service Unavailable."""
    engine = _make_mock_engine()
    engine.submit.side_effect = RuntimeError("request queue full (256 waiting). Try again later.")
    app = ServingApp(engine)
    client = TestClient(app.app, raise_server_exceptions=False)

    resp = client.post("/v1/completions", json={
        "prompt": "Hello",
        "max_tokens": 10,
    })
    assert resp.status_code == 503
    data = resp.json()
    assert "queue full" in data["error"]["message"]
