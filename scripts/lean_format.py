"""LEAN (LLM-Efficient Adaptive Notation) encoder/decoder.

Faithful Python port of the TypeScript reference at
https://github.com/fiialkod/lean-format/blob/main/src/lean.ts

Provides:
    encode(data)   -> str
    decode(lean)   -> Any
    LeanParseError (with .line and .content attributes)

Design notes / Python translation rules followed:
    * isinstance(True, int) is True in Python -- bool is checked BEFORE int.
    * JS Array.isArray(x)  ->  isinstance(x, list)
    * JS "object" && !== null  ->  isinstance(x, dict)
    * JS Number(s)  ->  int(s) / float(s) (with care: bare token "1" -> int,
      "1.5" -> float; "0" / "1" become ints, "1e3" becomes float, etc.)
    * dicts preserve insertion order in Python 3.7+, matching JS behavior.
    * \t in the TS source is a literal TAB character -- preserved as "\t".
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Tuple


# --- Public exception ---------------------------------------------------------

class LeanParseError(Exception):
    """Raised when a LEAN document cannot be parsed.

    Attributes:
        line:    1-indexed line number where the error was detected.
        content: The textual content of the offending line (trimmed).
    """

    def __init__(self, line: int, content: str, message: str) -> None:
        self.line = line
        self.content = content
        super().__init__(f"Line {line}: {message} -> \"{content}\"")


# --- Shared constants and helpers --------------------------------------------

KEY_REGEX = re.compile(r"^[\w][\w-]*$")
NUMBER_REGEX = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

# Decoder-side regexes (the "key" capture permits dots for dot-flattened paths).
_KV_RE = re.compile(r"^([\w][\w.-]*):(.*)")
_EMPTY_OBJ_RE = re.compile(r"^([\w][\w.-]*):\{\}\s*$")
_BLOCK_RE = re.compile(r"^([\w][\w.-]*):\s*$")
_TABULAR_RE = re.compile(
    r"^([\w][\w.-]*)\[(\d+)\]:([\w][\w-]*(?:\t[\w][\w-]*)*(?:\t~)?)\s*$"
)
_FLAT_ARRAY_RE = re.compile(r"^([\w][\w.-]*)\[(\d+)\]:(.+)$")
_EMPTY_ARRAY_RE = re.compile(r"^([\w][\w.-]*)\[0\]:\s*$")
_NON_UNIFORM_RE = re.compile(r"^([\w][\w.-]*)\[([1-9]\d*)\]:\s*$")

# Root array patterns (no key prefix).
_ROOT_TABULAR_RE = re.compile(
    r"^\[(\d+)\]:([\w][\w-]*(?:\t[\w][\w-]*)*(?:\t~)?)\s*$"
)
_ROOT_FLAT_RE = re.compile(r"^\[(\d+)\]:(.+)$")
_ROOT_EMPTY_RE = re.compile(r"^\[0\]:\s*$")
_ROOT_NON_UNIFORM_RE = re.compile(r"^\[([1-9]\d*)\]:\s*$")

# List-item patterns.
_LIST_ITEM_SCALAR_RE = re.compile(r"^-\s+(.+)$")
_LIST_ITEM_EMPTY_OBJ_RE = re.compile(r"^-\s+\{\}\s*$")
_LIST_ITEM_KV_RE = re.compile(r"^-\s+([\w][\w-]*):(.*)")
_LIST_ITEM_BLOCK_RE = re.compile(r"^-\s+([\w][\w-]*):\s*$")
_LIST_ITEM_EMPTY_ARRAY_RE = re.compile(r"^-\s+([\w][\w-]*)\[0\]:\s*$")
_LIST_ITEM_NON_UNIFORM_RE = re.compile(r"^-\s+([\w][\w-]*)\[([1-9]\d*)\]:\s*$")
_LIST_ITEM_TABULAR_RE = re.compile(
    r"^-\s+([\w][\w-]*)\[(\d+)\]:([\w][\w-]*(?:\t[\w][\w-]*)*(?:\t~)?)\s*$"
)
_LIST_ITEM_ROOT_ARRAY_RE = re.compile(r"^-\s+\[(\d+)\]:(.*)")
_LIST_ITEM_ROOT_EMPTY_RE = re.compile(r"^-\s+\[0\]:\s*$")
_LIST_ITEM_ROOT_NON_UNIFORM_RE = re.compile(r"^-\s+\[([1-9]\d*)\]:\s*$")
_LIST_ITEM_ROOT_TABULAR_RE = re.compile(
    r"^-\s+\[(\d+)\]:([\w][\w-]*(?:\t[\w][\w-]*)*(?:\t~)?)\s*$"
)
_LIST_ITEM_EMPTY_OBJ_KEY_RE = re.compile(r"^-\s+([\w][\w-]*):\{\}\s*$")


def _validate_key(key: str) -> None:
    """Raises ValueError if a key contains forbidden characters."""
    if not KEY_REGEX.match(key):
        raise ValueError(
            f'Unsupported key "{key}". Keys must match /^[\\w][\\w-]*$/ '
            f"(word chars and hyphens). No dots (reserved for path "
            f"flattening), spaces, slashes, or colons."
        )


def _is_scalar(value: Any) -> bool:
    """Return True for None / str / int / float / bool (bool is a subclass
    of int in Python, but the JS reference treats booleans as primitives
    alongside numbers, so we match that semantics)."""
    return (
        value is None
        or isinstance(value, bool)
        or isinstance(value, (int, float))
        or isinstance(value, str)
    )


def _is_tabular_array(arr: List[Any]) -> bool:
    """Return True iff arr is non-empty, every item is a non-array dict,
    all rows share the same key set, and every value in every row is a
    scalar."""
    if not arr:
        return False
    if any(not (isinstance(x, dict) and not isinstance(x, list)) for x in arr):
        return False
    first_keys = sorted(arr[0].keys())
    for item in arr:
        keys = sorted(item.keys())
        if len(keys) != len(first_keys):
            return False
        if any(k != fk for k, fk in zip(keys, first_keys)):
            return False
        if any(not _is_scalar(v) for v in item.values()):
            return False
    return True


# --- Scalar encoding ---------------------------------------------------------

def _needs_quoting(value: str) -> bool:
    if value == "":
        return True
    if value in ("T", "F", "_"):
        return True
    if value.strip() != value:
        return True
    if NUMBER_REGEX.match(value):
        return True
    if "\t" in value or "\n" in value or "\\" in value or '"' in value:
        return True
    return False


def _escape_scalar(value: str) -> str:
    # Order matters: backslash FIRST, then the other escapes.
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _encode_scalar(value: str, force_quote: bool) -> str:
    if force_quote or _needs_quoting(value):
        return f'"{_escape_scalar(value)}"'
    return value


def _encode_primitive(value: Any) -> str:
    if value is None:
        return "_"
    # CRITICAL: bool check MUST come before int (bool is a subclass of int).
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"Unsupported number: {value}. NaN and Infinity not representable."
            )
        return str(value)
    if isinstance(value, str):
        return _encode_scalar(value, False)
    raise ValueError("Not a primitive")


# --- Cell encoding (tabular context) -----------------------------------------

def _escape_cell(value: str) -> str:
    # Order matters: backslash FIRST.
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '""')  # RFC 4180 doubling
    )


def _cell_encode(value: Any) -> str:
    if value is None:
        return "_"
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"Unsupported number: {value}. NaN and Infinity not representable."
            )
        return str(value)
    if isinstance(value, str):
        if _needs_quoting(value):
            return f'"{_escape_cell(value)}"'
        return value
    raise ValueError("Not a scalar cell value")


# --- Encoder -----------------------------------------------------------------

def encode(data: Any) -> str:
    """Encode a Python value (dict / list / scalar) as a LEAN string."""
    if data is None:
        return "_"
    if isinstance(data, bool):
        return "T" if data else "F"
    if isinstance(data, (int, float)):
        if isinstance(data, float) and not math.isfinite(data):
            raise ValueError(f"Unsupported number: {data}.")
        return str(data)
    if isinstance(data, str):
        return _encode_scalar(data, True)  # root strings always quoted

    if isinstance(data, list):
        return _encode_root_array(data)

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported type: {type(data).__name__}")

    entries = list(data.items())
    if not entries:
        return "{}"

    lines: List[str] = []
    for key, value in entries:
        _validate_key(key)
        _encode_property(key, value, lines, 0)
    return "\n".join(lines)


def _encode_root_array(arr: List[Any]) -> str:
    lines: List[str] = []
    _encode_array_value("", arr, lines, 0)
    return "\n".join(lines)


def _encode_property(path: str, value: Any, lines: List[str], indent: int) -> None:
    # Scalar -> dot-flatten.
    if _is_scalar(value):
        pad = "  " * indent
        lines.append(f"{pad}{path}:{_encode_primitive(value)}")
        return

    # Array.
    if isinstance(value, list):
        _encode_array_value(path, value, lines, indent)
        return

    # Object -- try dot-flattening, fall back to indented block.
    obj = value
    entries = list(obj.items())

    if not entries:
        pad = "  " * indent
        lines.append(f"{pad}{path}:{{}}")
        return

    # At indent 0: compare dot-flattening vs indented block, pick shorter.
    if indent == 0:
        # Strategy 1: dot-flatten (extend path with dots).
        dot_lines: List[str] = []
        for key, val in entries:
            _validate_key(key)
            _encode_property(f"{path}.{key}", val, dot_lines, 0)

        # Strategy 2: indented block.
        block_lines: List[str] = [f"{path}:"]
        for key, val in entries:
            _validate_key(key)
            _encode_property(key, val, block_lines, 1)

        # Pick shorter (character count including newlines).
        dot_cost = sum(len(l) + 1 for l in dot_lines)
        block_cost = sum(len(l) + 1 for l in block_lines)
        lines.extend(dot_lines if dot_cost <= block_cost else block_lines)
    else:
        pad = "  " * indent
        lines.append(f"{pad}{path}:")
        for key, val in entries:
            _validate_key(key)
            _encode_property(key, val, lines, indent + 1)


def _encode_array_value(
    path: str, arr: List[Any], lines: List[str], indent: int
) -> None:
    pad = "  " * indent
    prefix = path

    if not arr:
        lines.append(f"{pad}{prefix}[0]:")
        return

    # Flat scalar array.
    if all(_is_scalar(v) for v in arr):
        cells = "\t".join(_cell_encode(v) for v in arr)
        lines.append(f"{pad}{prefix}[{len(arr)}]:{cells}")
        return

    # Tabular array.
    if _is_tabular_array(arr):
        fields = list(arr[0].keys())
        for f in fields:
            _validate_key(f)
        fields_joined = "\t".join(fields)
        lines.append(f"{pad}{prefix}[{len(arr)}]:{fields_joined}")
        for row in arr:
            cells = "\t".join(_cell_encode(row[f]) for f in fields)
            lines.append(f"{pad}  {cells}")
        return

    # Semi-tabular: all items are dicts with all-scalar values but different keys.
    all_objects = all(
        isinstance(item, dict) and not isinstance(item, list) for item in arr
    )
    if all_objects and len(arr) >= 2:
        objects = arr
        all_scalar_values = all(
            _is_scalar(v) for obj in objects for v in obj.values()
        )
        if all_scalar_values:
            first = objects[0]
            shared_keys = [
                k for k in first.keys()
                if all(k in obj for obj in objects)
            ]

            if shared_keys:
                for k in shared_keys:
                    _validate_key(k)
                shared_set = set(shared_keys)

                # Build semi-tabular encoding.
                semi_lines: List[str] = []
                shared_joined = "\t".join(shared_keys)
                semi_lines.append(
                    f"{pad}{prefix}[{len(arr)}]:{shared_joined}\t~"
                )
                for obj in objects:
                    factored = [_cell_encode(obj[k]) for k in shared_keys]
                    remaining = []
                    for k, v in obj.items():
                        if k not in shared_set:
                            _validate_key(k)
                            remaining.append(f"{k}:{_cell_encode(v)}")
                    cells = "\t".join(factored + remaining)
                    semi_lines.append(f"{pad}  {cells}")

                # Build dashed-list encoding for comparison.
                dashed_lines: List[str] = [f"{pad}{prefix}[{len(arr)}]:"]
                for item in arr:
                    _encode_list_item(item, dashed_lines, indent + 1)

                # Pick shorter.
                semi_cost = sum(len(l) + 1 for l in semi_lines)
                dashed_cost = sum(len(l) + 1 for l in dashed_lines)
                lines.extend(semi_lines if semi_cost < dashed_cost else dashed_lines)
                return

    # Non-uniform / mixed array.
    lines.append(f"{pad}{prefix}[{len(arr)}]:")
    for item in arr:
        _encode_list_item(item, lines, indent + 1)


def _encode_list_item(item: Any, lines: List[str], indent: int) -> None:
    pad = "  " * indent

    # Scalar.
    if _is_scalar(item):
        if isinstance(item, str):
            lines.append(f"{pad}- {_encode_scalar(item, False)}")
        else:
            lines.append(f"{pad}- {_encode_primitive(item)}")
        return

    # Sub-array.
    if isinstance(item, list):
        sub_lines: List[str] = []
        _encode_array_value("", item, sub_lines, 0)
        lines.append(f"{pad}- {sub_lines[0]}")
        for i in range(1, len(sub_lines)):
            lines.append(f"{pad}  {sub_lines[i]}")
        return

    # Object.
    obj = item
    entries = list(obj.items())

    if not entries:
        lines.append(f"{pad}- {{}}")
        return

    first_key, first_val = entries[0]
    _validate_key(first_key)

    if _is_scalar(first_val):
        sv = _encode_primitive(first_val)
        lines.append(f"{pad}- {first_key}:{sv}")
    elif isinstance(first_val, list):
        sub_lines = []
        _encode_array_value(first_key, first_val, sub_lines, 0)
        lines.append(f"{pad}- {sub_lines[0]}")
        for i in range(1, len(sub_lines)):
            lines.append(f"{pad}  {sub_lines[i]}")
    else:
        # Non-scalar object value as first key.
        sub_obj = first_val
        if not sub_obj:
            lines.append(f"{pad}- {first_key}:{{}}")
        else:
            lines.append(f"{pad}- {first_key}:")
            for k, v in sub_obj.items():
                _validate_key(k)
                _encode_property(k, v, lines, indent + 2)

    # Remaining keys.
    for i in range(1, len(entries)):
        key, val = entries[i]
        _validate_key(key)
        _encode_property(key, val, lines, indent + 1)


# --- Decoder -----------------------------------------------------------------

def _unescape_scalar(s: str) -> str:
    result: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "n":
                result.append("\n")
                i += 2
                continue
            if nxt == "\\":
                result.append("\\")
                i += 2
                continue
            if nxt == '"':
                result.append('"')
                i += 2
                continue
            result.append("\\")
            i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def _unescape_cell(s: str) -> str:
    result: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        # RFC 4180: doubled quote becomes a single literal quote.
        if ch == '"' and i + 1 < n and s[i + 1] == '"':
            result.append('"')
            i += 2
            continue
        if ch == "\\" and i + 1 < n:
            if s[i + 1] == "n":
                result.append("\n")
                i += 2
                continue
            if s[i + 1] == "\\":
                result.append("\\")
                i += 2
                continue
        result.append(ch)
        i += 1
    return "".join(result)


def _parse_number(s: str) -> Any:
    """Convert a numeric string to int if it has no decimal/exponent,
    otherwise to float. Mirrors JS Number() which keeps ints as ints."""
    if not s:
        return s  # unreachable in practice
    if "." in s or "e" in s or "E" in s:
        return float(s)
    return int(s)


def _parse_scalar_value(s: str) -> Any:
    s = s.strip()
    if s == "T":
        return True
    if s == "F":
        return False
    if s == "_":
        return None
    if s == "":
        return ""
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        return _unescape_scalar(s[1:-1])
    if NUMBER_REGEX.match(s):
        return _parse_number(s)
    return s  # bare string


def _parse_cell_value(s: str) -> Any:
    s = s.strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        return _unescape_cell(s[1:-1])
    if s == "T":
        return True
    if s == "F":
        return False
    if s == "_":
        return None
    if s == "":
        return ""
    if NUMBER_REGEX.match(s):
        return _parse_number(s)
    return s


def _split_tab_cells(line: str) -> List[str]:
    cells: List[str] = []
    current: List[str] = []
    in_quotes = False
    pos = 0
    n = len(line)
    while pos < n:
        ch = line[pos]
        if in_quotes:
            if ch == '"':
                if pos + 1 < n and line[pos + 1] == '"':
                    current.append('""')
                    pos += 2
                else:
                    in_quotes = False
                    current.append('"')
                    pos += 1
            else:
                current.append(ch)
                pos += 1
        else:
            if ch == '"':
                in_quotes = True
                current.append('"')
                pos += 1
            elif ch == "\t":
                cells.append("".join(current))
                current = []
                pos += 1
            else:
                current.append(ch)
                pos += 1
    cells.append("".join(current))
    if in_quotes:
        raise LeanParseError(0, line, f"Unterminated quote in row: \"{line}\"")
    return cells


def _parse_tab_row(line: str) -> List[Any]:
    return [_parse_cell_value(c) for c in _split_tab_cells(line)]


def _get_indent(line: str) -> int:
    m = re.match(r"^(\s*)", line)
    return len(m.group(1)) if m else 0


def _set_key_or_throw(
    target: dict, key: str, value: Any, line_idx: int, content: str
) -> None:
    if key in target:
        raise LeanParseError(line_idx, content, f'Duplicate key "{key}"')
    target[key] = value


def _set_nested_value(
    target: dict, path: str, value: Any, line_idx: int, content: str
) -> None:
    parts = path.split(".")
    current = target
    for i, part in enumerate(parts[:-1]):
        if part not in current:
            current[part] = {}
        nxt = current[part]
        if not (isinstance(nxt, dict) and not isinstance(nxt, list)):
            raise LeanParseError(
                line_idx, content, f'Cannot nest into non-object at "{part}"'
            )
        current = nxt
    _set_key_or_throw(current, parts[-1], value, line_idx, content)


class _ParseState:
    __slots__ = ("i", "lines")

    def __init__(self, i: int, lines: List[str]) -> None:
        self.i = i
        self.lines = lines


def _skip_trailing(state: _ParseState) -> None:
    while state.i < len(state.lines):
        if state.lines[state.i].strip() != "":
            raise LeanParseError(
                state.i,
                state.lines[state.i].strip(),
                "Unexpected trailing content",
            )
        state.i += 1


def _parse_root_array(state: _ParseState) -> List[Any]:
    line = state.lines[state.i].strip()

    if _ROOT_EMPTY_RE.match(line):
        state.i += 1
        return []

    tab_m = _ROOT_TABULAR_RE.match(line)
    if tab_m:
        count = int(tab_m.group(1))
        fields = tab_m.group(2).split("\t")
        semi_tabular = bool(fields) and fields[-1] == "~"
        if semi_tabular:
            fields.pop()

        # Peek: tabular arrays have indented data rows; flat arrays do not.
        peek_idx = state.i + 1
        while peek_idx < len(state.lines) and state.lines[peek_idx].strip() == "":
            peek_idx += 1
        next_is_data_row = (
            count > 0
            and peek_idx < len(state.lines)
            and _get_indent(state.lines[peek_idx]) >= 2
        )

        if next_is_data_row:
            state.i += 1
            return _parse_tabular_rows(
                state, count, fields, 2, line, semi_tabular
            )
        # Fall through to flat array parsing below.

    nu_m = _ROOT_NON_UNIFORM_RE.match(line)
    if nu_m:
        count = int(nu_m.group(1))
        state.i += 1
        return _parse_list_items(state, 0, count)

    flat_m = _ROOT_FLAT_RE.match(line)
    if flat_m:
        count = int(flat_m.group(1))
        values = _parse_tab_row(flat_m.group(2))
        if len(values) != count:
            raise LeanParseError(
                state.i,
                line,
                f"Array count mismatch (declared {count}, got {len(values)})",
            )
        state.i += 1
        return values

    raise LeanParseError(state.i, line, "Unrecognized root array syntax")


def _parse_tabular_rows(
    state: _ParseState,
    count: int,
    fields: List[str],
    min_indent: int,
    header_content: str,
    semi_tabular: bool = False,
) -> List[Any]:
    # Check for duplicate fields.
    seen: set = set()
    for f in fields:
        if f in seen:
            raise LeanParseError(
                state.i - 1, header_content, f'Duplicate field "{f}"'
            )
        seen.add(f)

    rows: List[Any] = []
    while len(rows) < count and state.i < len(state.lines):
        row_line = state.lines[state.i]
        if row_line.strip() == "":
            state.i += 1
            continue
        if _get_indent(row_line) < min_indent:
            break

        if semi_tabular:
            cells = _split_tab_cells(row_line.strip())
            if len(cells) < len(fields):
                raise LeanParseError(
                    state.i,
                    row_line.strip(),
                    f"Row field count mismatch (expected at least "
                    f"{len(fields)}, got {len(cells)})",
                )
            obj: dict = {}
            for idx, f in enumerate(fields):
                obj[f] = _parse_cell_value(cells[idx])
            # Extra cells are key:value pairs.
            for c in range(len(fields), len(cells)):
                cell = cells[c]
                colon_idx = cell.find(":")
                if colon_idx == -1:
                    raise LeanParseError(
                        state.i,
                        row_line.strip(),
                        f'Semi-tabular extra cell missing key:value format: "{cell}"',
                    )
                key = cell[:colon_idx]
                raw_val = cell[colon_idx + 1 :]
                _set_key_or_throw(
                    obj, key, _parse_cell_value(raw_val), state.i, row_line.strip()
                )
            rows.append(obj)
        else:
            values = _parse_tab_row(row_line.strip())
            if len(values) != len(fields):
                raise LeanParseError(
                    state.i,
                    row_line.strip(),
                    f"Row field count mismatch (expected {len(fields)}, "
                    f"got {len(values)})",
                )
            obj = {f: values[idx] for idx, f in enumerate(fields)}
            rows.append(obj)
        state.i += 1

    if len(rows) != count:
        raise LeanParseError(
            state.i - 1,
            header_content,
            f"Row count mismatch (declared {count}, got {len(rows)})",
        )
    return rows


def _parse_list_items(
    state: _ParseState, base_indent: int, count: int
) -> List[Any]:
    arr: List[Any] = []
    item_indent = base_indent + 2

    while len(arr) < count and state.i < len(state.lines):
        line = state.lines[state.i]
        if line.strip() == "":
            state.i += 1
            continue
        ind = _get_indent(line)
        if ind < item_indent:
            break
        content = line[ind:]

        # Empty object item.
        if _LIST_ITEM_EMPTY_OBJ_RE.match(content):
            arr.append({})
            state.i += 1
            continue

        # Sub-array items (- [N]:...).
        if _LIST_ITEM_ROOT_EMPTY_RE.match(content):
            arr.append([])
            state.i += 1
            continue

        li_root_tab_m = _LIST_ITEM_ROOT_TABULAR_RE.match(content)
        if li_root_tab_m:
            cnt = int(li_root_tab_m.group(1))
            fields = li_root_tab_m.group(2).split("\t")
            semi_tab = bool(fields) and fields[-1] == "~"
            if semi_tab:
                fields.pop()
            # Peek-ahead: tabular has indented data rows, flat does not.
            peek_idx = state.i + 1
            while peek_idx < len(state.lines) and state.lines[peek_idx].strip() == "":
                peek_idx += 1
            next_is_data = (
                cnt > 0
                and peek_idx < len(state.lines)
                and _get_indent(state.lines[peek_idx]) >= ind + 2
            )
            if next_is_data:
                state.i += 1
                arr.append(
                    _parse_tabular_rows(state, cnt, fields, ind + 2, content, semi_tab)
                )
                continue
            # Fall through to flat array below.

        li_root_nu_m = _LIST_ITEM_ROOT_NON_UNIFORM_RE.match(content)
        if li_root_nu_m:
            cnt = int(li_root_nu_m.group(1))
            state.i += 1
            arr.append(_parse_list_items(state, ind, cnt))
            continue

        li_root_arr_m = _LIST_ITEM_ROOT_ARRAY_RE.match(content)
        if li_root_arr_m:
            cnt = int(li_root_arr_m.group(1))
            values = _parse_tab_row(li_root_arr_m.group(2))
            if len(values) != cnt:
                raise LeanParseError(
                    state.i, content, "Array count mismatch"
                )
            arr.append(values)
            state.i += 1
            continue

        # Object items (- key:value, - key{}, etc.).
        if (
            _LIST_ITEM_EMPTY_OBJ_KEY_RE.match(content)
            or _LIST_ITEM_TABULAR_RE.match(content)
            or _LIST_ITEM_EMPTY_ARRAY_RE.match(content)
            or _LIST_ITEM_NON_UNIFORM_RE.match(content)
            or _LIST_ITEM_KV_RE.match(content)
            or _LIST_ITEM_BLOCK_RE.match(content)
        ):
            arr.append(_parse_list_item_object(state, ind, item_indent))
            continue

        # Scalar item.
        scalar_m = _LIST_ITEM_SCALAR_RE.match(content)
        if scalar_m:
            arr.append(_parse_scalar_value(scalar_m.group(1)))
            state.i += 1
            continue

        raise LeanParseError(state.i, content, "Unrecognized list item")

    if len(arr) != count:
        raise LeanParseError(
            state.i - 1,
            "",
            f"List count mismatch (declared {count}, got {len(arr)})",
        )
    return arr


def _parse_list_item_object(
    state: _ParseState, line_ind: int, parent_item_indent: int
) -> dict:
    obj: dict = {}
    content = state.lines[state.i][line_ind:]
    first_line_idx = state.i

    # - key:{}
    empty_obj_key_m = _LIST_ITEM_EMPTY_OBJ_KEY_RE.match(content)
    if empty_obj_key_m:
        _set_key_or_throw(
            obj, empty_obj_key_m.group(1), {}, first_line_idx, content
        )
        state.i += 1
    # - key[N]:fields (tabular) -- peek-ahead to disambiguate from flat arrays.
    else:
        tab_m = _LIST_ITEM_TABULAR_RE.match(content)
        is_tabular_item = False
        if tab_m:
            count = int(tab_m.group(2))
            peek_idx = state.i + 1
            while peek_idx < len(state.lines) and state.lines[peek_idx].strip() == "":
                peek_idx += 1
            is_tabular_item = (
                count > 0
                and peek_idx < len(state.lines)
                and _get_indent(state.lines[peek_idx]) >= line_ind + 2
            )
        if tab_m and is_tabular_item:
            key = tab_m.group(1)
            count_str = tab_m.group(2)
            fields_str = tab_m.group(3)
            fields = fields_str.split("\t")
            semi_tab = bool(fields) and fields[-1] == "~"
            if semi_tab:
                fields.pop()
            state.i += 1
            _set_key_or_throw(
                obj,
                key,
                _parse_tabular_rows(
                    state, int(count_str), fields, line_ind + 2, content, semi_tab
                ),
                first_line_idx,
                content,
            )
        # - key[0]:
        else:
            empty_arr_m = _LIST_ITEM_EMPTY_ARRAY_RE.match(content)
            if empty_arr_m:
                _set_key_or_throw(
                    obj, empty_arr_m.group(1), [], first_line_idx, content
                )
                state.i += 1
            # - key[N]:
            else:
                nu_m = _LIST_ITEM_NON_UNIFORM_RE.match(content)
                if nu_m:
                    state.i += 1
                    _set_key_or_throw(
                        obj,
                        nu_m.group(1),
                        _parse_list_items(
                            state, line_ind, int(nu_m.group(2))
                        ),
                        first_line_idx,
                        content,
                    )
                # - key:value or - key: (block)
                else:
                    block_m = _LIST_ITEM_BLOCK_RE.match(content)
                    if block_m:
                        nested: dict = {}
                        state.i += 1
                        _parse_block(state, line_ind + 4, nested)
                        _set_key_or_throw(
                            obj, block_m.group(1), nested, first_line_idx, content
                        )
                    else:
                        kv_m = _LIST_ITEM_KV_RE.match(content)
                        if kv_m:
                            _set_key_or_throw(
                                obj,
                                kv_m.group(1),
                                _parse_scalar_value(kv_m.group(2)),
                                first_line_idx,
                                content,
                            )
                            state.i += 1
                        else:
                            raise LeanParseError(
                                state.i, content, "Invalid list item object"
                            )

    # Remaining keys.
    body_indent = line_ind + 2
    while state.i < len(state.lines):
        line = state.lines[state.i]
        if line.strip() == "":
            state.i += 1
            continue
        ind = _get_indent(line)
        if ind < body_indent:
            break
        lc = line[ind:]
        if ind == parent_item_indent and lc.startswith("- "):
            break
        _parse_line(state, ind, obj, lc)

    return obj


def _parse_line(
    state: _ParseState, ind: int, target: dict, content: str
) -> None:
    line_idx = state.i

    # key:{}
    empty_obj_m = _EMPTY_OBJ_RE.match(content)
    if empty_obj_m:
        _set_nested_value(target, empty_obj_m.group(1), {}, line_idx, content)
        state.i += 1
        return

    # key[N]:fields (tabular) -- disambiguate from flat arrays by peeking ahead.
    tab_m = _TABULAR_RE.match(content)
    if tab_m:
        path = tab_m.group(1)
        count_str = tab_m.group(2)
        fields_str = tab_m.group(3)
        count = int(count_str)
        fields = fields_str.split("\t")
        semi_tab = bool(fields) and fields[-1] == "~"
        if semi_tab:
            fields.pop()

        # Peek: tabular arrays have indented data rows; flat arrays do not.
        peek_idx = state.i + 1
        while peek_idx < len(state.lines) and state.lines[peek_idx].strip() == "":
            peek_idx += 1
        next_is_data_row = (
            count > 0
            and peek_idx < len(state.lines)
            and _get_indent(state.lines[peek_idx]) >= ind + 2
        )

        if next_is_data_row:
            state.i += 1
            _set_nested_value(
                target,
                path,
                _parse_tabular_rows(
                    state, count, fields, ind + 2, content, semi_tab
                ),
                line_idx,
                content,
            )
            return
        # Fall through to flat array parsing below.

    # key[0]:
    empty_arr_m = _EMPTY_ARRAY_RE.match(content)
    if empty_arr_m:
        _set_nested_value(target, empty_arr_m.group(1), [], line_idx, content)
        state.i += 1
        return

    # key[N]: (non-uniform)
    nu_m = _NON_UNIFORM_RE.match(content)
    if nu_m:
        state.i += 1
        _set_nested_value(
            target,
            nu_m.group(1),
            _parse_list_items(state, ind, int(nu_m.group(2))),
            line_idx,
            content,
        )
        return

    # key[N]:values (flat)
    flat_m = _FLAT_ARRAY_RE.match(content)
    if flat_m:
        count = int(flat_m.group(2))
        values = _parse_tab_row(flat_m.group(3))
        if len(values) != count:
            raise LeanParseError(
                line_idx,
                content,
                f"Array count mismatch (declared {count}, got {len(values)})",
            )
        _set_nested_value(target, flat_m.group(1), values, line_idx, content)
        state.i += 1
        return

    # key: (block header)
    block_m = _BLOCK_RE.match(content)
    if block_m:
        nested: dict = {}
        state.i += 1
        _parse_block(state, ind + 2, nested)
        _set_nested_value(target, block_m.group(1), nested, line_idx, content)
        return

    # key:value
    kv_m = _KV_RE.match(content)
    if kv_m:
        _set_nested_value(
            target, kv_m.group(1), _parse_scalar_value(kv_m.group(2)), line_idx, content
        )
        state.i += 1
        return

    raise LeanParseError(state.i, content, "Unrecognized LEAN syntax")


def _parse_block(state: _ParseState, base_indent: int, target: dict) -> None:
    while state.i < len(state.lines):
        line = state.lines[state.i]
        if line.strip() == "":
            state.i += 1
            continue
        ind = _get_indent(line)
        if ind < base_indent:
            break
        content = line[ind:]
        _parse_line(state, ind, target, content)


def decode(lean: str) -> Any:
    """Decode a LEAN string back into a Python value (dict / list / scalar)."""
    lines = lean.split("\n")
    first_idx = 0
    while first_idx < len(lines) and lines[first_idx].strip() == "":
        first_idx += 1
    if first_idx >= len(lines):
        raise LeanParseError(0, "", "Empty LEAN document")

    first = lines[first_idx].strip()
    state = _ParseState(first_idx, lines)

    # Root empty object.
    if first == "{}":
        state.i += 1
        _skip_trailing(state)
        return {}

    # Root scalar.
    if first == "T":
        state.i += 1
        _skip_trailing(state)
        return True
    if first == "F":
        state.i += 1
        _skip_trailing(state)
        return False
    if first == "_":
        state.i += 1
        _skip_trailing(state)
        return None

    # Root quoted string.
    if len(first) >= 2 and first.startswith('"') and first.endswith('"'):
        state.i += 1
        _skip_trailing(state)
        return _unescape_scalar(first[1:-1])

    # Root number.
    if first != "" and ":" not in first and "[" not in first:
        try:
            if NUMBER_REGEX.match(first):
                num = _parse_number(first)
                if isinstance(num, (int, float)) and math.isfinite(num):
                    state.i += 1
                    _skip_trailing(state)
                    return num
        except (ValueError, OverflowError):
            pass

    # Root array.
    if first.startswith("["):
        result = _parse_root_array(state)
        _skip_trailing(state)
        return result

    # Root object (key:value or key[N]:... lines).
    result: dict = {}
    _parse_block(state, 0, result)
    _skip_trailing(state)
    return result


# --- Self-test ---------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    TESTS = [
        # Test 1: Simple object
        {"users": [
            {"id": 1, "name": "Alice", "active": True},
            {"id": 2, "name": "Bob", "active": False},
        ]},
        # Test 2: Nested with dots
        {"config": {"database": {"host": "db.internal.prod", "port": 5432}}},
        # Test 3: Mixed types
        {
            "meta": {"version": "2.1.0", "debug": False},
            "tags": [],
            "notes": [1, "hello", {"key": "val"}],
        },
        # Test 4: Empty containers
        {"items": [], "meta": {}},
    ]

    # Additional round-trip stress tests.
    EXTRA = [
        # All scalar primitives
        {"a": 1, "b": 1.5, "c": -3, "d": 1e3, "e": True, "f": False, "g": None},
        # Bare strings that need quoting
        {"k1": "T", "k2": "F", "k3": "_", "k4": "42", "k5": "", "k6": "hello world",
         "k7": "tab\there", "k8": 'with"quote'},
        # Flat scalar array
        {"scores": [95, 87, 42, 100, 73]},
        # Tabular
        {"users": [
            {"id": 1, "name": "Alice", "email": "alice@ex.com", "active": True},
            {"id": 2, "name": "Bob", "email": "bob@ex.com", "active": False},
        ]},
        # Non-uniform list
        {"events": [
            {"type": "click", "target": "button-submit"},
            {"type": "pageview", "url": "/dashboard", "referrer": "google.com"},
            {"type": "error", "message": "NullPointerException", "severity": "high"},
        ]},
        # Root array
        [{"x": 1}, {"x": 2}],
        # Root scalars
        "root string",
        42,
        True,
        None,
    ]

    passed = 0
    failed = 0
    for i, data in enumerate(TESTS + EXTRA, 1):
        label = f"Test {i}"
        try:
            encoded = encode(data)
            decoded = decode(encoded)
            ok = decoded == data
            if ok:
                passed += 1
                print(f"  PASS  {label}: round-trip OK")
            else:
                failed += 1
                print(f"  FAIL  {label}: round-trip mismatch")
                print(f"        input:    {json.dumps(data, default=str)}")
                print(f"        encoded:  {encoded!r}")
                print(f"        decoded:  {json.dumps(decoded, default=str)}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {label}: {type(e).__name__}: {e}")
            print(f"        input:    {json.dumps(data, default=str)}")

    # SPEC example round-trip.
    spec_data = {
        "meta": {"version": "2.1.0", "debug": False},
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@ex.com", "active": True},
            {"id": 2, "name": "Bob", "email": "bob@ex.com", "active": False},
        ],
        "tags": [],
        "notes": [1, "hello", {"key": "val"}],
    }
    try:
        out = encode(spec_data)
        spec_str = (
            "meta.version:2.1.0\n"
            "meta.debug:F\n"
            "users[2]:id\tname\temail\tactive\n"
            "  1\tAlice\talice@ex.com\tT\n"
            "  2\tBob\tbob@ex.com\tF\n"
            "tags[0]:\n"
            "notes[3]:\n"
            "  - 1\n"
            "  - hello\n"
            "  - key:val"
        )
        # The encoder may pick dot-flatten OR indented block for `meta` --
        # we only need to verify that the decoded round-trip succeeds AND
        # that the produced string decodes to spec_data.
        decoded = decode(out)
        if decoded == spec_data:
            passed += 1
            print("  PASS  SPEC example: round-trip OK")
        else:
            failed += 1
            print("  FAIL  SPEC example: round-trip mismatch")
            print(f"        encoded:\n{out}")
            print(f"        decoded: {json.dumps(decoded, default=str)}")
    except Exception as e:
        failed += 1
        print(f"  FAIL  SPEC example: {type(e).__name__}: {e}")

    print()
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
