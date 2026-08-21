#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
import json
import re
import sys
from pathlib import Path

FILE = Path(".divan_diff.json")
FILE.touch(exist_ok=True)
NEG_PAD = 6
SEPARATOR = "│"
EXCLUDE_HEADERS = ["fastest", "slowest", "samples"]
MIN_PERCENT = 5 / 100

header_pattern = re.compile(
    r"^(?P<bench>\w+) *|(?P<header>\w+) *(?P<separator>│)?", flags=re.MULTILINE
)
line_pattern = re.compile(
    r"(?:^(?P<indent>[ ─╰│├]+))?(?P<name>[a-zA-Z]\w*)? +?(?P<value>[0-9\.]+)? ?(?P<unit>\w+)?(?:(?P<colend> *)│|$)",
    flags=re.MULTILINE,
)


to_ms = {"ns": 1 / 1e6, "µs": 1 / 1000, "ms": 1, "s": 1000, "m": 1000 * 60}


def parse_value(val: str, unit: str):
    unit = unit.strip()
    if len(unit) == 0:
        return int(val)
    else:
        return float(val) * to_ms[unit]


def to_unit(val: float, unit: str) -> float:
    unit = unit.strip()
    return val / to_ms[unit]


diff_template = " {sign}{percent:.2%} {sign}{delta:.3}  "
diff_template_len = len(diff_template.format(sign="+", delta=234.5678, percent=0.98899))
in_divan = False
benchg = []
headers = []
col_ranges = []
line_len = 0
last_indent_l = 0


def dim(s: str) -> str:
    return f"\x1b[2;37m{s}\x1b[0m"


def green(s: str) -> str:
    return f"\x1b[32m{s}\x1b[0m"


def red(s: str) -> str:
    return f"\x1b[31m{s}\x1b[0m"


def buffer_line(line: str) -> str:
    offset = 0
    for start, end in col_ranges:
        prev_len = len(line)
        line = line[: offset + start] + " " * diff_template_len + line[offset + end :]
        offset += len(line) - prev_len
    return line


for line in sys.stdin:
    if len(benchg) == 0 and SEPARATOR in line:
        first, *head = header_pattern.finditer(line)
        benchg.append(first.group("bench"))
        for h in head:
            header = h.group("header")
            headers.append(header)
            sep = h.start("separator")
            if sep > 0 and all(excl not in header for excl in EXCLUDE_HEADERS):
                col_ranges.append((sep - 1 - NEG_PAD, sep - 1))
        last_indent_l = 1
        line_len = len(line)
        print(buffer_line(line), end="")
        continue

    if len(benchg) > 0:
        if SEPARATOR not in line or not all(
            line[sep + 1] == SEPARATOR for (_, sep) in col_ranges
        ):
            in_divan = False
            benchg.clear()
            headers.clear()
            col_ranges.clear()
            print(line, end="")
            continue

        values = {}
        indent = ""
        matches = list(line_pattern.finditer(line))
        for i, col in enumerate(matches):
            if i == 0:
                indent = col.group("indent")
                indent_l = len(indent)
                if indent_l < last_indent_l:
                    benchg.pop()
                elif indent_l == last_indent_l:
                    benchg.pop()
                    benchg.append(col.group("name"))
                elif indent_l > last_indent_l:
                    benchg.append(col.group("name"))
                last_indent_l = indent_l
            if headers[i] in EXCLUDE_HEADERS:
                continue
            val = col.group("value") or ""
            if len(val.strip()) > 0:
                unit = col.group("unit") or ""
                if len(unit) > 0:
                    values[headers[i]] = (
                        parse_value(val, unit),
                        unit,
                        (col.end("colend") - NEG_PAD, col.end("colend")),
                    )
        with FILE.open("r") as f:
            try:
                data: dict = json.load(f)
            except Exception as e:
                print(e)
                data = {}
        datakey = ".".join(benchg)
        vals = data.get(datakey, {})
        col_diffs = {}
        offset = 0
        for k, (v, unit, (start, end)) in values.items():
            prev = vals.get(k)
            if prev is None:
                diff = " " * diff_template_len
            else:
                delta = v - prev
                p_diff = (v - prev) / prev
                diff = diff_template.format(
                    sign="" if delta < 0 else "+",
                    delta=to_unit(delta, unit),
                    percent=p_diff,
                ).ljust(diff_template_len)
                if abs(p_diff) < MIN_PERCENT:
                    diff = dim(diff)
                elif delta < 0:
                    diff = green(diff)
                elif delta > 0:
                    diff = red(diff)
            vals[k] = v
            len_pre = len(line)
            line = line[: offset + start] + diff + line[offset + end :]
            offset += len(line) - len_pre
        if len(values) == 0:
            line = buffer_line(line)
        data[datakey] = vals
        with open(".divan_diff.json", "w+") as f:
            json.dump(data, f)
    print(line, end="")
