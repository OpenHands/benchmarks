"""Tests for LiteLLM proxy virtual key management."""

import threading
from unittest.mock import patch

import httpx
import pytest

from benchmarks.utils.litellm_proxy import (
    _get_config,
    create_virtual_key,
    delete_key,
    get_current_virtual_key,
    get_key_spend,
    set_current_virtual_key,
)


_DUMMY_REQUEST = httpx.Request("GET", "https://proxy.example.com")


def _response(status_code: int, json: dict) -> httpx.Response:
    """Create an httpx.Response that supports raise_for_status()."""
    return httpx.Response(status_code, json=json, request=_DUMMY_REQUEST)


# ---- _get_config --------------------------------------------------------------


class TestGetConfig:
    def test_returns_none_when_no_env(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_MASTER_KEY", raising=False)
        assert _get_config() is None

    def test_returns_none_when_no_api_key(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.delenv("LLM_API_MASTER_KEY", raising=False)
        assert _get_config() is None

    def test_returns_none_when_no_base_url(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-test")
        assert _get_config() is None

    def test_returns_tuple_when_configured(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-test")
        assert _get_config() == ("https://proxy.example.com", "sk-test")

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com/")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-test")
        result = _get_config()
        assert result is not None
        assert result[0] == "https://proxy.example.com"


# ---- create_virtual_key -------------------------------------------------------


class TestCreateVirtualKey:
    def test_returns_none_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_MASTER_KEY", raising=False)
        assert create_virtual_key("inst-1") is None

    def test_success(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-admin")

        mock_resp = _response(200, {"key": "sk-virtual-123"})
        with patch("benchmarks.utils.litellm_proxy.httpx.post", return_value=mock_resp):
            key = create_virtual_key("inst-1", run_id="run-42")
        assert key == "sk-virtual-123"

    def test_sends_correct_payload(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-admin")

        mock_resp = _response(200, {"key": "sk-v"})
        with patch(
            "benchmarks.utils.litellm_proxy.httpx.post", return_value=mock_resp
        ) as mock_post:
            create_virtual_key("inst-1", run_id="run-42", max_budget=10.0)

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["metadata"] == {
            "instance_id": "inst-1",
            "run_id": "run-42",
        }
        assert kwargs["json"]["max_budget"] == 10.0
        assert kwargs["headers"]["Authorization"] == "Bearer sk-admin"

    def test_raises_on_http_error(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-admin")

        mock_resp = _response(401, {"error": "unauthorized"})
        with patch("benchmarks.utils.litellm_proxy.httpx.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="virtual key creation failed"):
                create_virtual_key("inst-1")

    def test_raises_on_connection_error(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-admin")

        with patch(
            "benchmarks.utils.litellm_proxy.httpx.post",
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(RuntimeError, match="virtual key creation failed"):
                create_virtual_key("inst-1")

    def test_raises_on_timeout(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-admin")

        with patch(
            "benchmarks.utils.litellm_proxy.httpx.post",
            side_effect=httpx.ReadTimeout("timed out"),
        ):
            with pytest.raises(RuntimeError, match="virtual key creation failed"):
                create_virtual_key("inst-1")


# ---- get_key_spend -------------------------------------------------------------


class TestGetKeySpend:
    def test_returns_none_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_MASTER_KEY", raising=False)
        assert get_key_spend("sk-v") is None

    def test_success(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-admin")

        mock_resp = _response(200, {"info": {"spend": 0.014112}})
        with patch("benchmarks.utils.litellm_proxy.httpx.get", return_value=mock_resp):
            spend = get_key_spend("sk-virtual-123")
        assert spend == 0.014112

    def test_returns_none_on_http_500(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-admin")

        mock_resp = _response(500, {"error": "internal"})
        with patch("benchmarks.utils.litellm_proxy.httpx.get", return_value=mock_resp):
            assert get_key_spend("sk-v") is None

    def test_returns_none_on_connection_error(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-admin")

        with patch(
            "benchmarks.utils.litellm_proxy.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            assert get_key_spend("sk-v") is None


# ---- delete_key ----------------------------------------------------------------


class TestDeleteKey:
    def test_noop_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_MASTER_KEY", raising=False)
        # Should not raise
        delete_key("sk-v")

    def test_success(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-admin")

        mock_resp = _response(200, {})
        with patch(
            "benchmarks.utils.litellm_proxy.httpx.post", return_value=mock_resp
        ) as mock_post:
            delete_key("sk-virtual-123")

        _, kwargs = mock_post.call_args
        assert kwargs["json"] == {"keys": ["sk-virtual-123"]}

    def test_swallows_http_error(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-admin")

        mock_resp = _response(403, {"error": "forbidden"})
        with patch("benchmarks.utils.litellm_proxy.httpx.post", return_value=mock_resp):
            # Should not raise
            delete_key("sk-v")

    def test_swallows_connection_error(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("LLM_API_MASTER_KEY", "sk-admin")

        with patch(
            "benchmarks.utils.litellm_proxy.httpx.post",
            side_effect=httpx.ConnectError("refused"),
        ):
            delete_key("sk-v")


# ---- Thread-local virtual key --------------------------------------------------


class TestThreadLocalVirtualKey:
    def test_default_is_none(self):
        # Clear any leftover from other tests
        set_current_virtual_key(None)
        assert get_current_virtual_key() is None

    def test_set_and_get(self):
        set_current_virtual_key("sk-thread-key")
        assert get_current_virtual_key() == "sk-thread-key"
        set_current_virtual_key(None)

    def test_clear(self):
        set_current_virtual_key("sk-thread-key")
        set_current_virtual_key(None)
        assert get_current_virtual_key() is None

    def test_thread_isolation(self):
        """Different threads see different virtual keys."""
        results = {}
        barrier = threading.Barrier(2)

        def worker(name, key):
            set_current_virtual_key(key)
            barrier.wait()  # Ensure both threads have set their key
            results[name] = get_current_virtual_key()
            set_current_virtual_key(None)

        t1 = threading.Thread(target=worker, args=("t1", "key-A"))
        t2 = threading.Thread(target=worker, args=("t2", "key-B"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == "key-A"
        assert results["t2"] == "key-B"

    def test_main_thread_not_affected_by_worker(self):
        """Setting a key in a worker thread doesn't leak to the main thread."""
        set_current_virtual_key(None)

        def worker():
            set_current_virtual_key("worker-key")

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert get_current_virtual_key() is None
