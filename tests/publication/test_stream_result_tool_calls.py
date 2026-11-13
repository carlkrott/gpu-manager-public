"""Streaming clients get the same tool calls as buffered clients.

The broker consumes every member as an SSE stream and accumulates it into a
durable chat.completion; `stream_result` replays that durable result to
streaming callers. Live 2026-10-11: the replay carried only content and
finish_reason, so a streaming agent saw finish_reason "tool_calls" with no
tool calls. These tests run member SSE -> broker accumulator -> durable
result -> stream replay -> the same accumulator acting as the client.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gemma_broker.runtime import BufferedAiohttpTransport, _CandidateCompatAdapter  # noqa: E402


def _sse(chunks: list[dict]) -> bytes:
    body = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
    return body + b"data: [DONE]\n\n"


def _member_tool_stream() -> bytes:
    base = {"id": "chatcmpl-m", "created": 1, "model": "gpt-6-luna", "object": "chat.completion.chunk"}

    def chunk(delta, finish=None, **extra):
        return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

    return _sse(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"reasoning_content": "need both"}),
            chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                                   "function": {"name": "get_weather", "arguments": ""}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":"}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": "\"Paris\"}"}}]}),
            chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function",
                                   "function": {"name": "get_time", "arguments": "{\"city\":\"Tokyo\"}"}}]}),
            chunk({}, "tool_calls"),
            {**base, "choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}},
        ]
    )


class _Response:
    def __init__(self, raw: bytes, split: int = 7) -> None:
        self._parts = [raw[i:i + split] for i in range(0, len(raw), split)]

    @property
    def content(self):
        async def gen():
            for part in self._parts:
                yield part

        return gen()


class _Wait:
    def __init__(self, terminal: dict) -> None:
        self._terminal = terminal

    async def wait(self, job_id: str, *, timeout: float) -> dict:
        return self._terminal


def _transport() -> BufferedAiohttpTransport:
    return BufferedAiohttpTransport(session=SimpleNamespace())


def _replay(terminal: dict) -> bytes:
    adapter = _CandidateCompatAdapter(SimpleNamespace(), polling_wait=_Wait(terminal))

    async def collect() -> bytes:
        return b"".join([frame async for frame in adapter.stream_result("job1", timeout=1.0)])

    return asyncio.run(collect())


def _accumulate(raw: bytes) -> dict:
    return asyncio.run(_transport()._read_streamed_chat_response(_Response(raw)))


def test_streaming_client_gets_member_tool_calls():
    durable = _accumulate(_member_tool_stream())
    message = durable["choices"][0]["message"]
    assert [c["function"]["name"] for c in message["tool_calls"]] == ["get_weather", "get_time"]

    replay = _replay({"status": 200, "body": json.dumps(durable)})
    client = _accumulate(replay)

    choice = client["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == message["tool_calls"]
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
    assert choice["message"]["reasoning_content"] == "need both"
    assert client["usage"] == {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}


def test_tool_call_frames_have_index_id_type():
    durable = _accumulate(_member_tool_stream())
    replay = _replay({"status": 200, "body": durable})
    frames = [json.loads(line[6:]) for line in replay.decode().splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    calls = [c for f in frames for ch in f["choices"] for c in ch["delta"].get("tool_calls", [])]
    assert [(c["index"], c["id"], c["type"]) for c in calls] == [
        (0, "call_a", "function"),
        (1, "call_b", "function"),
    ]
    assert all(f["object"] == "chat.completion.chunk" for f in frames)
    assert replay.endswith(b"data: [DONE]\n\n")


def test_plain_text_replay_unchanged():
    durable = {
        "id": "chatcmpl-x", "created": 2, "model": "gemma", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    }
    replay = _replay({"status": 200, "body": json.dumps(durable)})
    frames = [json.loads(line[6:]) for line in replay.decode().splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert [f["choices"][0]["delta"] for f in frames] == [{"role": "assistant"}, {"content": "hello"}, {}]
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert "usage" not in frames[-1]


def test_null_content_with_tool_calls_is_not_sent_as_null():
    durable = {
        "id": "chatcmpl-y", "created": 3, "model": "gemma",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}}],
    }
    replay = _replay({"status": 200, "body": json.dumps(durable)})
    assert b'"content":null' not in replay
    client = _accumulate(replay)
    assert client["choices"][0]["message"]["tool_calls"][0]["function"] == {"name": "f", "arguments": "{}"}


def test_error_result_still_single_error_frame():
    replay = _replay({"status": 503, "body": json.dumps({"error": {"message": "no member"}})})
    assert replay == b'data: {"error":{"message":"no member"}}\n\ndata: [DONE]\n\n'
