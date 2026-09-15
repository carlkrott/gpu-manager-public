"""
Format Converter — multi-format input/output conversion layer for GPU Manager.

INPUT SIDE: Detects structured data in LLM prompts and converts to LEAN format
  (28% token savings vs JSON, highest LLM accuracy per benchmarks).

OUTPUT SIDE: Detects LLM response format and converts to guaranteed JSON.
  LEAN → JSON, JSON → passthrough, prose → wrapped JSON.

Usage:
    from format_converter import FormatConverter
    fc = FormatConverter()

    # Input: convert prompt messages to LEAN-optimized
    lean_messages = fc.convert_messages_to_lean(original_messages)

    # Output: ensure response is JSON
    json_response = fc.convert_response_to_json(raw_response_body)
"""

import json
import re
import logging

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False
    json_repair = None  # type: ignore

try:
    import xmltodict
    _HAS_XMLTODICT = True
except ImportError:
    _HAS_XMLTODICT = False

try:
    from lean_format import encode as lean_encode, decode as lean_decode, LeanParseError
    _HAS_LEAN = True
except ImportError:
    _HAS_LEAN = False
    logging.warning("lean_format not available — LEAN conversion disabled")

logger = logging.getLogger("format_converter")

# ─── Detection patterns ──────────────────────────────────────────────────

# Fenced code blocks: ```json ... ``` or ```yaml ... ```
_FENCE_RE = re.compile(
    r'```(\w+)?\s*\n(.*?)```',
    re.DOTALL
)

# Inline JSON objects/arrays (greedy from first { or [ to matching close)
_JSON_OBJ_START = re.compile(r'(\{|\[)')

# LEAN signatures: key:value pairs, tabular headers key[N]:fields
_LEAN_KV_RE = re.compile(r'^[\w][\w.-]*:.+', re.MULTILINE)
_LEAN_TABULAR_RE = re.compile(r'^[\w\[][\w.-]*\[\d+\]:', re.MULTILINE)

# YAML: key: value (but not JSON since JSON also has these)
_YAML_LINE_RE = re.compile(r'^[\w][\w-]*:\s*\S', re.MULTILINE)

# XML: <tag>...</tag>
_XML_RE = re.compile(r'<\?xml|<[a-zA-Z][^>]*>.*?</[a-zA-Z][^>]*>', re.DOTALL)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 5.2 — per-service format gate (single source of truth for body shapes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# When you add a new backend body shape to GPU Manager (e.g. a new sd-server
# flavour, a ComfyUI variant, or a custom music API), you MUST:
#
#   1. Add the new format name to BODY_FORMATS below.
#   2. Add the input side to _PASSTHROUGH_INPUT (if passthrough) OR add a
#      dedicated branch in convert_request_for_format() that calls into
#      FormatConverter() or does its own transformation.
#   3. Add the output side to _PASSTHROUGH_OUTPUT (if passthrough) OR add a
#      dedicated branch in convert_response_for_format().
#   4. Update services.json — set the affected service entry's
#      `input_body_format` and `output_body_format` to the new enum value.
#   5. Restart gpu-manager.
#
# No worker-loop change is required for known-format swaps — the dispatch
# table is the extension point. gpu-manager.py WorkerPool reads the gate
# from services.json and routes through these helpers.
#
# If you do NOT update all 5 places, jobs will silently pass through (or
# mis-convert) for the affected service. Phase 5.3 will add a
# _validate_service_config() hook that refuses to start services with
# inconsistent (routing_group_type, input_body_format) combinations.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Full enum of allowed body-format values (input + output). When a new
# shape is added, append it here. These are the values that may appear in
# services.json `input_body_format` and `output_body_format` fields.
BODY_FORMATS = (
    # ── Input side ──────────────────────────────────────────────────────
    "openai_chat",            # OpenAI chat-completions body (LEAN-in)
    "ollama",                 # Ollama-style {prompt: "..."} body (LEAN-in)
    "comfyui_workflow",       # ComfyUI /prompt workflow JSON (passthrough)
    "sd_server_image",        # sd-server /sdapi/v1/txt2img body (passthrough)
    "sd_server_video",        # sd-server /sdcpp/v1/vid_gen body (passthrough)
    "acestep_music",          # acestep /generate body (passthrough)
    "raw_json",               # generic JSON, no conversion
    "passthrough",            # explicit "do nothing" alias
    # ── Output side ─────────────────────────────────────────────────────
    "openai_chat_out",        # OpenAI-style {choices:[...]} → JSON
    "ollama_out",             # Ollama-style {response: "..."} → JSON
    "comfyui_ack",            # ComfyUI {prompt_id, number, ...} passthrough
    "sd_server_image_out",    # sd-server {images:[...], info:...} passthrough
    "sd_server_video_out",    # sd-server video frames passthrough
    "acestep_audio_out",      # acestep audio response passthrough
    "raw_bytes",              # return body verbatim (any non-JSON or raw bytes)
    "raw_json_out",           # generic JSON, no conversion
)

# Defaults derived from routing_group_type (services.json field). When
# input_body_format / output_body_format is absent on a service entry,
# gpu-manager falls back to these maps, then to raw_json / raw_bytes.
DEFAULT_INPUT_BY_RGT  = {
    "llm":          "openai_chat",
    "generation":   "raw_json",
    "orchestrator": "raw_json",
}
DEFAULT_OUTPUT_BY_RGT = {
    "llm":          "openai_chat_out",
    "generation":   "raw_bytes",
    "orchestrator": "raw_bytes",
}

# Formats that require NO conversion on the input side — the body is
# forwarded to the backend verbatim (after UTF-8 encoding). The 6
# generation-side shapes are all in this set on purpose: the previous
# global converter mis-encoded the `prompt` field of sd-server/acestep/
# sulphur-unet bodies as LEAN, which corrupted the caption. Adding them
# here is the fix for bug #3 in PHASE5.2_PLAN.md §0.
_PASSTHROUGH_INPUT = {
    "raw_json", "passthrough", "comfyui_workflow",
    "sd_server_image", "sd_server_video", "acestep_music",
}
_PASSTHROUGH_OUTPUT = {
    "raw_bytes", "raw_json_out", "comfyui_ack",
    "sd_server_image_out", "sd_server_video_out", "acestep_audio_out",
}


def resolve_input_format(service_config, routing_group_type):
    """Return the resolved input body format for a service.

    Resolution order: explicit service_config field → routing_group_type
    default → raw_json (final fallback). Never raises.
    """
    return (service_config or {}).get("input_body_format") \
        or DEFAULT_INPUT_BY_RGT.get(routing_group_type, "raw_json")


def resolve_output_format(service_config, routing_group_type):
    """Return the resolved output body format for a service.

    Resolution order: explicit service_config field → routing_group_type
    default → raw_bytes (final fallback). Never raises.
    """
    return (service_config or {}).get("output_body_format") \
        or DEFAULT_OUTPUT_BY_RGT.get(routing_group_type, "raw_bytes")


class FormatConverter:
    """
    Multi-format converter for LLM request/response transformation.

    Input pipeline: JSON/YAML/XML → Python dict → LEAN encoded string
    Output pipeline: LEAN/JSON/prose → JSON guaranteed
    """

    def __init__(self, enable_lean: bool = True, enable_yaml: bool = True,
                 enable_xml: bool = True):
        self.enable_lean = enable_lean and _HAS_LEAN
        self.enable_yaml = enable_yaml and _HAS_YAML
        self.enable_xml = enable_xml and _HAS_XMLTODICT
        self.stats = {"input_converted": 0, "output_converted": 0,
                       "lean_encoded": 0, "json_passthrough": 0,
                       "json_repaired": 0, "prose_wrapped": 0, "errors": 0}

    # ═══════════════════════════════════════════════════════════════════════
    # INPUT SIDE: Convert prompt messages to LEAN-optimized format
    # ═══════════════════════════════════════════════════════════════════════

    def convert_messages_to_lean(self, messages: list) -> list:
        """
        Process OpenAI-format messages array.
        Finds structured data blocks and converts them to LEAN.
        Non-structured text (instructions, prose) is left untouched.

        Args:
            messages: List of {"role": ..., "content": ...} dicts

        Returns:
            New messages list with structured data converted to LEAN
        """
        if not self.enable_lean:
            return messages

        result = []
        for msg in messages:
            new_msg = dict(msg)
            content = new_msg.get("content", "")
            if isinstance(content, str) and content:
                converted = self._convert_text_blocks_to_lean(content)
                if converted != content:
                    new_msg["content"] = converted
                    self.stats["lean_encoded"] += 1
            result.append(new_msg)
        return result

    def _convert_text_blocks_to_lean(self, text: str) -> str:
        """
        Find structured data blocks within text and convert to LEAN.
        Handles:
        1. Fenced code blocks (```json, ```yaml, ```xml)
        2. Large inline JSON objects (>200 chars)
        3. Leaves instructions and prose untouched
        """

        # Phase 1: Convert fenced code blocks
        def _replace_fence(match):
            lang = (match.group(1) or "").lower().strip()
            code = match.group(2).strip()
            converted = self._try_convert_block(code, lang)
            if converted is not None:
                self.stats["input_converted"] += 1
                return f"```lean\n{converted}\n```"
            return match.group(0)  # keep original if conversion fails

        text = _FENCE_RE.sub(_replace_fence, text)

        # Phase 2: Convert large inline JSON blocks (>200 chars)
        # Only convert if it's clearly structured data, not part of a sentence
        text = self._convert_inline_json(text)

        return text

    def _convert_inline_json(self, text: str) -> str:
        """Find and convert large inline JSON blocks to LEAN."""
        if not self.enable_lean:
            return text

        # Find potential JSON blocks (lines that start with { or [ and span multiple lines)
        lines = text.split("\n")
        result_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check if this line starts a JSON block
            if stripped.startswith("{") or stripped.startswith("["):
                # Try to accumulate a complete JSON object
                json_candidate = self._extract_json_block(lines, i)
                if json_candidate:
                    json_text, end_line = json_candidate
                    # Only convert if it's substantial (>200 chars)
                    if len(json_text) > 200:
                        converted = self._try_convert_block(json_text, "json")
                        if converted is not None:
                            indent = len(line) - len(line.lstrip())
                            pad = " " * indent
                            lean_lines = converted.split("\n")
                            for ll in lean_lines:
                                result_lines.append(f"{pad}{ll}")
                            self.stats["input_converted"] += 1
                            i = end_line + 1
                            continue
                    # Small JSON — include original lines
                    for j in range(i, end_line + 1):
                        result_lines.append(lines[j])
                    i = end_line + 1
                    continue

            result_lines.append(line)
            i += 1

        return "\n".join(result_lines)

    def _extract_json_block(self, lines: list, start: int):
        """Extract a balanced JSON block starting at line index."""
        text = "\n".join(lines[start:])
        stripped = text.lstrip()
        if not stripped or stripped[0] not in "{[":
            return None

        opener = stripped[0]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escape_next = False
        end_char_idx = -1

        for idx, ch in enumerate(stripped):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    end_char_idx = idx
                    break

        if end_char_idx == -1:
            return None

        json_text = stripped[:end_char_idx + 1]

        # Map back to line index
        chars_consumed = len(text) - len(stripped) + end_char_idx + 1
        line_idx = start
        remaining = chars_consumed
        while remaining > 0 and line_idx < len(lines):
            line_len = len(lines[line_idx]) + 1  # +1 for \n
            if remaining <= line_len:
                break
            remaining -= line_len
            line_idx += 1

        return (json_text, line_idx)

    def _try_convert_block(self, code: str, lang_hint: str) -> str | None:
        """
        Try to parse code block and convert to LEAN.
        Returns LEAN string on success, None on failure.
        """
        if not self.enable_lean:
            return None

        try:
            data = self._detect_and_parse(code, lang_hint)
            if data is not None:
                return lean_encode(data)
        except Exception as e:
            logger.debug(f"LEAN conversion failed for {lang_hint} block: {e}")
        return None

    def _detect_and_parse(self, text: str, lang_hint: str = ""):
        """
        Detect format and parse to Python object.
        Returns dict/list/scalar on success, None on failure.
        """
        text = text.strip()
        if not text:
            return None

        # If language hint is explicit, try that first
        if lang_hint:
            parsed = self._parse_as_format(text, lang_hint)
            if parsed is not None:
                return parsed

        # Auto-detection order: JSON → YAML → XML
        # JSON is most common and unambiguous
        parsed = self._parse_as_format(text, "json")
        if parsed is not None:
            return parsed

        parsed = self._parse_as_format(text, "yaml")
        if parsed is not None:
            return parsed

        parsed = self._parse_as_format(text, "xml")
        if parsed is not None:
            return parsed

        return None

    def _parse_as_format(self, text: str, fmt: str):
        """Try parsing text as specific format. Returns None on failure."""
        try:
            if fmt == "json":
                data = json.loads(text)
                # Don't convert simple scalars or tiny objects
                if isinstance(data, (dict, list)):
                    return data
                return None

            elif fmt == "yaml" and self.enable_yaml:
                data = yaml.safe_load(text)
                # yaml.safe_load can parse JSON too, so verify it's not just
                # picking up the same thing
                if isinstance(data, (dict, list)) and data:
                    return data
                return None

            elif fmt == "xml" and self.enable_xml:
                # Quick check that it looks like XML
                if not _XML_RE.search(text):
                    return None
                data = xmltodict.parse(text)
                # Unwrap root element if single key
                if len(data) == 1:
                    root_key = list(data.keys())[0]
                    inner = data[root_key]
                    if isinstance(inner, (dict, list)):
                        return inner
                return data if data else None

        except Exception:
            return None
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # OUTPUT SIDE: Convert LLM response to guaranteed JSON
    # ═══════════════════════════════════════════════════════════════════════

    def convert_response_to_json(self, response_body: str) -> str:
        """
        Take raw LLM response body and ensure output is JSON.

        Handles three cases:
        1. Response is OpenAI format with JSON content → passthrough
        2. Response content is LEAN → decode to JSON
        3. Response content is prose → wrap in JSON envelope

        Args:
            response_body: Raw HTTP response body string from LLM backend

        Returns:
            JSON string (guaranteed parseable by json.loads)
        """
        # First, is the entire response body even JSON?
        try:
            resp = json.loads(response_body)
        except (json.JSONDecodeError, TypeError):
            # The whole body isn't JSON — wrap it entirely
            self.stats["prose_wrapped"] += 1
            return json.dumps({
                "content": response_body,
                "type": "prose",
                "format": "raw"
            })

        # Check if it's OpenAI chat completion format
        if isinstance(resp, dict) and "choices" in resp:
            return self._convert_openai_response(resp)

        # Check if it's Ollama format
        if isinstance(resp, dict) and "response" in resp and "model" in resp:
            return self._convert_ollama_response(resp)

        # It's JSON but not a known format — passthrough
        self.stats["json_passthrough"] += 1
        return response_body

    def _convert_openai_response(self, resp: dict) -> str:
        """Convert OpenAI-format response, ensuring content is JSON."""
        for choice in resp.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content", "")

            if not isinstance(content, str) or not content:
                continue

            converted = self._ensure_content_is_json(content)
            if converted is not None:
                msg["content"] = converted

        self.stats["output_converted"] += 1
        return json.dumps(resp)

    def _convert_ollama_response(self, resp: dict) -> str:
        """Convert Ollama-format response, ensuring content is JSON."""
        content = resp.get("response", "")
        if isinstance(content, str) and content:
            converted = self._ensure_content_is_json(content)
            if converted is not None:
                resp["response"] = converted
        return json.dumps(resp)

    def _ensure_content_is_json(self, content: str) -> str | None:
        """
        Ensure a content string is valid JSON.
        Returns JSON string, or None if no conversion needed.

        Strategy:
        1. Already valid JSON → return as-is (passthrough)
        2. Contains a JSON code block → extract and return it
        3. LEAN format → decode to JSON
        4. Plain prose → wrap in {"content": "...", "type": "prose"}
        """
        content = content.strip()

        # Case 1: Already valid JSON
        try:
            parsed = json.loads(content)
            # It's valid JSON — but ensure it's a string representation
            # that the caller can use directly
            self.stats["json_passthrough"] += 1
            return json.dumps(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        # Case 2: Contains fenced JSON block
        json_block = self._extract_json_from_text(content)
        if json_block is not None:
            self.stats["output_converted"] += 1
            return json_block

        # Case 2.5: Try json-repair on malformed JSON (LLM produced JSON with
        # syntax errors: extra braces, truncated at token limit, unquoted keys,
        # trailing commas, etc.). json-repair returns a dict for repairable
        # input, or empty string for pure prose.
        if _HAS_JSON_REPAIR:
            try:
                repaired = json_repair.loads(content)
                if repaired and isinstance(repaired, (dict, list)):
                    self.stats["json_repaired"] += 1
                    return json.dumps(repaired)
            except Exception:
                pass

        # Case 3: Try LEAN decode
        if self.enable_lean and self._looks_like_lean(content):
            try:
                data = lean_decode(content)
                self.stats["output_converted"] += 1
                return json.dumps(data)
            except LeanParseError:
                pass
            except Exception:
                pass

        # Case 4: Prose — wrap in JSON envelope
        self.stats["prose_wrapped"] += 1
        return json.dumps({
            "content": content,
            "type": "prose"
        })

    def _extract_json_from_text(self, text: str) -> str | None:
        """
        Try to extract valid JSON from text that contains other content.
        Looks for:
        1. Fenced JSON code blocks
        2. Inline JSON objects/arrays
        """
        # Look for ```json ... ``` blocks
        for match in _FENCE_RE.finditer(text):
            lang = (match.group(1) or "").lower().strip()
            code = match.group(2).strip()
            if lang in ("json", ""):
                try:
                    parsed = json.loads(code)
                    return json.dumps(parsed)
                except json.JSONDecodeError:
                    continue

        # Look for inline JSON (starts with { or [)
        for match in _JSON_OBJ_START.finditer(text):
            start = match.start()
            candidate = text[start:]
            # Try progressively shorter substrings until we find valid JSON
            for end_marker in ["]", "}", ")\n"]:
                last_pos = candidate.rfind(end_marker)
                while last_pos > 0:
                    try:
                        parsed = json.loads(candidate[:last_pos + 1])
                        if isinstance(parsed, (dict, list)):
                            return json.dumps(parsed)
                    except json.JSONDecodeError:
                        pass
                    last_pos = candidate.rfind(end_marker, 0, last_pos)

        return None

    def _looks_like_lean(self, text: str) -> bool:
        """Heuristic: does this text look like LEAN format?"""
        text = text.strip()
        if not text:
            return False

        # LEAN uses key:value with no space after colon
        kv_matches = _LEAN_KV_RE.findall(text)
        tabular_matches = _LEAN_TABULAR_RE.findall(text)

        # At least 2 key:value lines or 1 tabular header
        if len(kv_matches) >= 2 or len(tabular_matches) >= 1:
            # But not JSON (JSON has spaces after colons and uses quotes)
            json_indicators = text.count('": ') + text.count('":')
            lean_indicators = text.count(":") - json_indicators
            return lean_indicators > 0

        return False

    # ═══════════════════════════════════════════════════════════════════════
    # Request body conversion (for the proxy handler)
    # ═══════════════════════════════════════════════════════════════════════

    def convert_request_body(self, body: bytes | str) -> bytes:
        """
        Convert an HTTP request body to LEAN-optimized format.

        Parses the body as OpenAI/Ollama format, converts message content
        to LEAN where structured data is detected.

        Returns bytes ready for forwarding to the LLM backend.
        """
        if isinstance(body, bytes):
            body_str = body.decode("utf-8", errors="replace")
        else:
            body_str = body

        try:
            req = json.loads(body_str)
        except (json.JSONDecodeError, TypeError):
            # Not JSON — can't convert, return as-is
            return body if isinstance(body, bytes) else body.encode("utf-8")

        # OpenAI format
        if "messages" in req:
            req["messages"] = self.convert_messages_to_lean(req["messages"])
            return json.dumps(req).encode("utf-8")

        # Ollama format
        if "prompt" in req:
            converted = self._convert_text_blocks_to_lean(req["prompt"])
            if converted != req["prompt"]:
                req["prompt"] = converted
            return json.dumps(req).encode("utf-8")

        # Unknown format — return as-is
        return body if isinstance(body, bytes) else body.encode("utf-8")

    # ═══════════════════════════════════════════════════════════════════════
    # Utility
    # ═══════════════════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """Return conversion statistics."""
        return dict(self.stats)

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {k: 0 for k in self.stats}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 5.2 — module-level dispatch helpers (per-service format gate)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# These are the SINGLE entry points used by gpu-manager.WorkerPool for body
# conversion. They replace the previous global `_fmt_converter` singleton
# + `_FMT_ENABLED` boolean. Each call constructs a fresh FormatConverter()
# instance — verified safe (no class state, no locks, only `self.stats`).
#
# gpu-manager.py imports:
#   - convert_request_for_format(input_format, body)  → bytes
#   - convert_response_for_format(output_format, body) → str
#   - resolve_input_format(service_config, rgt)        → str
#   - resolve_output_format(service_config, rgt)       → str
#
# See the "REGISTER NEW FORMAT HERE" block above BODY_FORMATS for the
# extension contract. New formats need a branch in convert_request_for_format
# (if not passthrough) and a branch in convert_response_for_format.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _to_bytes(body):
    """Encode body to bytes if it isn't already. Centralised for dispatch."""
    return body if isinstance(body, bytes) else body.encode("utf-8")


def convert_request_for_format(input_format, body):
    """Convert an inbound request body per the resolved input_format enum.

    Returns bytes (always) so gpu-manager can pass them straight to
    aiohttp's data= parameter.

    - Passthrough formats (raw_json, comfyui_workflow, sd_server_*, acestep_*,
      passthrough) → return body unchanged (UTF-8 encoded).
    - openai_chat → FormatConverter().convert_request_body() (LEAN-in for
      OpenAI messages).
    - ollama → force the LEAN-prompt path even though the body has a string
      `prompt` field (the previous global converter over-detected this
      for sd-server bodies; with the per-service gate we only hit it when
      the service is declared as `ollama`).
    - Any unknown format → passthrough (raw bytes).
    """
    if input_format in _PASSTHROUGH_INPUT:
        return _to_bytes(body)
    if input_format == "openai_chat":
        return FormatConverter().convert_request_body(body)
    if input_format == "ollama":
        try:
            req = json.loads(body if isinstance(body, str) else body.decode("utf-8", errors="replace"))
        except Exception:
            return _to_bytes(body)
        if isinstance(req, dict) and isinstance(req.get("prompt"), str):
            fc = FormatConverter()
            converted = fc._convert_text_blocks_to_lean(req["prompt"])
            if converted != req["prompt"]:
                req["prompt"] = converted
            return json.dumps(req).encode("utf-8")
        return _to_bytes(body)
    # Unknown format — passthrough rather than crash. WorkerPool will log
    # a debug line and forward the body unchanged.
    return _to_bytes(body)


def convert_response_for_format(output_format, response_body):
    """Convert an outbound response body per the resolved output_format enum.

    Returns a str so gpu-manager can use it directly in the OOM substring
    heuristic and as the response body in the proxy reply.

    - Passthrough formats (raw_bytes, comfyui_ack, sd_server_*_out,
      acestep_audio_out, raw_json_out) → return response_body unchanged.
    - openai_chat_out, ollama_out, raw_json_out (when not in passthrough)
      → FormatConverter().convert_response_to_json() (LEAN/prose → JSON
      guaranteed).
    - Any unknown format → passthrough.
    """
    if output_format in _PASSTHROUGH_OUTPUT:
        return response_body
    if output_format in ("openai_chat_out", "ollama_out"):
        return FormatConverter().convert_response_to_json(response_body)
    # Unknown format — passthrough.
    return response_body
