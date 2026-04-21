#!/usr/bin/env python3
"""
Collect AI responses through a stdio MCP server and write an enriched prompts CSV.

This is intentionally schema-configurable for local or custom MCP servers.
For the official Peec MCP integration, use peec_mcp_export.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROMPT_COLUMNS = ("prompt", "raw_prompt", "query", "question", "text")


def main() -> int:
    args = parse_args()
    prompts = pd.read_csv(args.prompts)
    prompt_col = pick_column(prompts, PROMPT_COLUMNS)
    prompts = prompts.copy()
    if "prompt_id" not in prompts.columns:
        prompts["prompt_id"] = [f"prompt_{i:03d}" for i in range(1, len(prompts) + 1)]

    base_tool_args = parse_tool_args(args.tool_arg)
    command = split_command(args.mcp_command)

    with StdioMcpClient(command) as client:
        client.initialize()
        if args.list_tools:
            tools = client.list_tools()
            print(json.dumps(tools, indent=2, ensure_ascii=False))
            return 0

        responses: list[str] = []
        raw_results: list[str] = []
        for _, row in prompts.iterrows():
            prompt = str(row[prompt_col])
            tool_args = render_tool_args(
                base_tool_args,
                prompt=prompt,
                brand=args.brand,
                engine=args.engine,
            )
            tool_args[args.prompt_arg] = prompt
            if args.brand and args.brand_arg:
                tool_args[args.brand_arg] = args.brand
            if args.engine and args.engine_arg:
                tool_args[args.engine_arg] = args.engine

            result = client.call_tool(args.tool, tool_args)
            raw_results.append(json.dumps(result, ensure_ascii=False))
            responses.append(extract_response(result, args.response_path))

    prompts["response"] = responses
    prompts["peec_mcp_tool"] = args.tool
    prompts["peec_mcp_raw_result"] = raw_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompts.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote Peec-enriched prompts to {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect prompt responses from a stdio MCP server."
    )
    parser.add_argument("--prompts", required=True, type=Path, help="Input prompts CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output enriched CSV.")
    parser.add_argument(
        "--mcp-command",
        required=True,
        help="Command used to start the Peec MCP server, for example 'npx peec-mcp'.",
    )
    parser.add_argument(
        "--tool",
        required=True,
        help="MCP tool name to call for each prompt.",
    )
    parser.add_argument(
        "--prompt-arg",
        default="prompt",
        help="Tool argument name that receives the prompt text.",
    )
    parser.add_argument("--brand", default="", help="Target brand, if the Peec tool accepts it.")
    parser.add_argument(
        "--brand-arg",
        default="brand",
        help="Tool argument name that receives --brand. Use '' to omit.",
    )
    parser.add_argument("--engine", default="", help="Engine/platform, if the Peec tool accepts it.")
    parser.add_argument(
        "--engine-arg",
        default="engine",
        help="Tool argument name that receives --engine. Use '' to omit.",
    )
    parser.add_argument(
        "--tool-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra static tool argument. Values may include {prompt}, {brand}, {engine}.",
    )
    parser.add_argument(
        "--response-path",
        default="",
        help="Dot path to response text inside tool result, e.g. structuredContent.answer.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Initialize the MCP server and print available tools instead of collecting.",
    )
    return parser.parse_args()


def split_command(command: str) -> list[str]:
    import shlex

    return shlex.split(command)


def pick_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    by_lower = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    raise ValueError(
        f"Could not find prompt column. Expected one of {', '.join(candidates)}. "
        f"Found: {', '.join(frame.columns)}"
    )


def parse_tool_args(items: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--tool-arg must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def render_tool_args(args: dict[str, str], *, prompt: str, brand: str, engine: str) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in args.items():
        rendered[key] = value.format(prompt=prompt, brand=brand, engine=engine)
    return rendered


def extract_response(result: dict[str, Any], response_path: str) -> str:
    if response_path:
        value = get_path(result, response_path)
        return stringify(value)

    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        for key in ("response", "answer", "text", "output", "content"):
            if key in structured:
                return stringify(structured[key])

    content = result.get("content")
    if isinstance(content, list):
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if texts:
            return "\n".join(texts)

    return stringify(result)


def get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(f"Cannot read {part!r} from {type(current).__name__}")
    return current


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


class StdioMcpClient:
    def __init__(self, command: Sequence[str]) -> None:
        self.command = list(command)
        self.process: subprocess.Popen[bytes] | None = None
        self.next_id = 1

    def __enter__(self) -> "StdioMcpClient":
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr.buffer,
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "feature-visibility-mvp", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized", {})

    def list_tools(self) -> dict[str, Any]:
        return self.request("tools/list", {})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        self.write_message(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        while True:
            message = self.read_message()
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(f"MCP {method} failed: {message['error']}")
            return message.get("result", {})

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.write_message({"jsonrpc": "2.0", "method": method, "params": params})

    def write_message(self, message: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP process is not running.")
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.process.stdin.write(header + body)
        self.process.stdin.flush()

    def read_message(self) -> dict[str, Any]:
        if not self.process or not self.process.stdout:
            raise RuntimeError("MCP process is not running.")

        headers: dict[str, str] = {}
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed stdout.")
            if line in (b"\r\n", b"\n"):
                break
            key, value = line.decode("ascii").split(":", 1)
            headers[key.lower()] = value.strip()

        length = int(headers["content-length"])
        body = self.process.stdout.read(length)
        return json.loads(body.decode("utf-8"))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
