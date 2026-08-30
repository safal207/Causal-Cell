from __future__ import annotations

import json
import os
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from causal_cell import (
    ModelCall,
    ModelResultStatus,
    OllamaModelAdapter,
    digest_json,
)
from causal_cell.canonical import format_timestamp
from causal_cell.ollama import OLLAMA_MAX_RESPONSE_BYTES, _post_json


def model_call(*, expired: bool = False) -> ModelCall:
    payload = {"repository": {"branch": "main", "dirty": False}}
    deadline = datetime.now(UTC) + timedelta(seconds=-1 if expired else 30)
    return ModelCall(
        run_id="orun-test",
        organism_id="organism:repository-pilot",
        stage="observer",
        trace_id="trace-test",
        span_id="span-test",
        parent_span_id=None,
        intent_id="intent:test",
        parent_cause="user-request:test",
        payload=payload,
        payload_digest=digest_json(payload),
        deadline_at=format_timestamp(deadline),
        max_output_tokens=64,
        contains_secret=False,
        data_classification="internal",
    )


class OllamaAdapterTests(unittest.TestCase):
    def test_default_transport_explicitly_disables_environment_proxies(self) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers.items.return_value = []
        response.read.return_value = b"{}"
        opener = MagicMock()
        opener.open.return_value = response

        hostile_proxy_environment = {
            "HTTP_PROXY": "http://203.0.113.10:8080",
            "HTTPS_PROXY": "http://203.0.113.10:8080",
            "NO_PROXY": "",
        }
        with patch.dict(os.environ, hostile_proxy_environment, clear=False):
            with patch("causal_cell.ollama.build_opener", return_value=opener) as build:
                _post_json("http://127.0.0.1:11434/api/chat", b"{}", 1.0)

        proxy_handler = build.call_args.args[0]
        self.assertEqual({}, proxy_handler.proxies)
        opener.open.assert_called_once()

    def test_structured_local_call_returns_guarded_model_result(self) -> None:
        captured: dict[str, object] = {}
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        }

        def transport(url: str, body: bytes, timeout: float):
            captured.update(url=url, body=json.loads(body), timeout=timeout)
            return (
                200,
                {},
                json.dumps(
                    {
                        "message": {"content": '{"summary":"clean"}'},
                        "prompt_eval_count": 12,
                        "eval_count": 5,
                    }
                ).encode(),
            )

        adapter = OllamaModelAdapter(
            adapter_id="adapter:observer:ollama",
            model="qwen3:4b",
            output_schema=schema,
            system_prompt="Return bounded repository facts.",
            transport=transport,
        )
        schema["required"].clear()
        result = adapter.invoke(model_call())

        self.assertEqual(ModelResultStatus.RETURNED, result.status)
        self.assertEqual({"summary": "clean"}, result.output)
        self.assertEqual(12, result.input_tokens)
        self.assertEqual(5, result.output_tokens)
        self.assertEqual(0, result.cost_microunits)
        self.assertEqual("internal", result.data_classification)
        self.assertEqual("http://127.0.0.1:11434/api/chat", captured["url"])
        request = captured["body"]
        self.assertEqual("qwen3:4b", request["model"])
        self.assertFalse(request["stream"])
        self.assertFalse(request["think"])
        self.assertEqual(64, request["options"]["num_predict"])
        self.assertEqual(["summary"], request["format"]["required"])
        self.assertIn("untrusted data", request["messages"][1]["content"])

    def test_invalid_response_is_terminal_and_preserves_labels(self) -> None:
        adapter = OllamaModelAdapter(
            adapter_id="adapter:observer:ollama",
            model="qwen3:4b",
            output_schema={"type": "object"},
            system_prompt="Return JSON.",
            transport=lambda _url, _body, _timeout: (
                200,
                {},
                b'{"message":{"content":"not-json"}}',
            ),
        )
        result = adapter.invoke(model_call())
        self.assertEqual(ModelResultStatus.ERROR, result.status)
        self.assertEqual("OLLAMA_RESPONSE_INVALID", result.error_code)
        self.assertEqual("internal", result.data_classification)

    def test_oversized_response_is_terminal_before_json_parsing(self) -> None:
        adapter = OllamaModelAdapter(
            adapter_id="adapter:observer:ollama",
            model="qwen3:4b",
            output_schema={"type": "object"},
            system_prompt="Return JSON.",
            transport=lambda _url, _body, _timeout: (
                200,
                {},
                b"x" * (OLLAMA_MAX_RESPONSE_BYTES + 1),
            ),
        )
        result = adapter.invoke(model_call())
        self.assertEqual(ModelResultStatus.ERROR, result.status)
        self.assertEqual("OLLAMA_RESPONSE_TOO_LARGE", result.error_code)

    def test_expired_call_does_not_touch_transport(self) -> None:
        invoked = False

        def transport(_url: str, _body: bytes, _timeout: float):
            nonlocal invoked
            invoked = True
            raise AssertionError("transport must not run")

        adapter = OllamaModelAdapter(
            adapter_id="adapter:observer:ollama",
            model="qwen3:4b",
            output_schema={"type": "object"},
            system_prompt="Return JSON.",
            transport=transport,
        )
        result = adapter.invoke(model_call(expired=True))
        self.assertEqual(ModelResultStatus.ERROR, result.status)
        self.assertEqual("OLLAMA_DEADLINE_EXCEEDED", result.error_code)
        self.assertFalse(invoked)

    def test_non_loopback_or_hostname_http_endpoint_is_rejected(self) -> None:
        for endpoint in (
            "http://localhost:11434",
            "http://10.0.0.2:11434",
            "https://127.0.0.1:11434",
            "http://127.0.0.1:11434/api",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(ValueError, "literal loopback"):
                    OllamaModelAdapter(
                        adapter_id="adapter:observer:ollama",
                        model="qwen3:4b",
                        output_schema={"type": "object"},
                        system_prompt="Return JSON.",
                        base_url=endpoint,
                    )

    def test_prompt_and_output_schema_are_bound_into_adapter_identity(self) -> None:
        common = {
            "adapter_id": "adapter:observer:ollama",
            "model": "qwen3:4b",
            "system_prompt": "Return JSON.",
            "output_schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
            },
        }
        baseline = OllamaModelAdapter(**common)
        changed_prompt = OllamaModelAdapter(
            **{**common, "system_prompt": "Return different JSON."}
        )
        changed_schema = OllamaModelAdapter(
            **{
                **common,
                "output_schema": {
                    "type": "object",
                    "properties": {"risks": {"type": "array"}},
                },
            }
        )

        self.assertNotEqual(
            baseline.identity.identity_digest,
            changed_prompt.identity.identity_digest,
        )
        self.assertNotEqual(
            baseline.identity.identity_digest,
            changed_schema.identity.identity_digest,
        )


if __name__ == "__main__":
    unittest.main()
