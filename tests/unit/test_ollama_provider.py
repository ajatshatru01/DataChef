"""
Unit tests for OllamaProvider.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from datachef.providers.base import ProviderError
from datachef.providers.ollama import OllamaProvider


def test_ollama_provider_tool_call_normalization():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "gemma4",
        "message": {
            "role": "assistant",
            "content": "Let me calculate statistics for sales.",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_column_stats",
                        "arguments": {"columns": ["sales"]},
                    }
                }
            ],
        },
        "prompt_eval_count": 55,
        "eval_count": 22,
    }

    provider = OllamaProvider(host="http://localhost:11434", model_id="gemma4")

    with patch("requests.post", return_value=mock_response) as mock_post:
        resp = provider.chat_completion(
            messages=[{"role": "user", "content": "Analyze sales"}],
            tools=[{"type": "function", "function": {"name": "get_column_stats"}}],
        )

        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:11434/api/chat"

        assert resp.finish_reason == "tool_calls"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].tool_name == "get_column_stats"
        assert resp.tool_calls[0].arguments == {"columns": ["sales"]}
        assert resp.usage.total_tokens == 77


def test_ollama_provider_converts_openai_tool_history_to_native_schema():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "gemma4:cloud",
        "message": {"role": "assistant", "content": "The mean is 12."},
    }

    messages = [
        {"role": "user", "content": "What is the mean?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "get_column_stats",
                        "arguments": '{"columns": ["sales"]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"sales": {"mean": 12}}',
        },
    ]
    provider = OllamaProvider(host="http://localhost:11434", model_id="gemma4:cloud")

    with patch("requests.post", return_value=mock_response) as mock_post:
        provider.chat_completion(messages=messages, tools=[])

    payload = mock_post.call_args.kwargs["json"]
    native_call = payload["messages"][1]["tool_calls"][0]
    assert native_call == {
        "id": "call-1",
        "function": {
            "name": "get_column_stats",
            "arguments": {"columns": ["sales"]},
        },
    }
    assert payload["messages"][2]["tool_name"] == "get_column_stats"
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"columns": ["sales"]}'
    )


def test_ollama_provider_connection_error():
    import requests

    provider = OllamaProvider(host="http://localhost:11434", model_id="gemma4", max_retries=0)

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(ProviderError) as exc_info:
            provider.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                tools=[],
            )

        assert "Could not connect to Ollama" in str(exc_info.value)
