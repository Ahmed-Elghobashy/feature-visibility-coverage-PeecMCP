#!/usr/bin/env python3
"""
Export tracked Peec AI MCP chats into the CSV shape consumed by visibility_mvp.py.

Peec MCP is a read-only Streamable HTTP MCP server:
  https://api.peec.ai/mcp

This exporter uses the official MCP tools:
  - list_projects
  - list_prompts
  - list_models
  - list_chats
  - get_chat

First run requires OAuth authorization in the browser/terminal. Tokens are stored
locally in .peec_mcp_tokens.json, which is ignored by git.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


PEEC_MCP_URL = "https://api.peec.ai/mcp"
TOKEN_PATH = Path(".peec_mcp_tokens.json")
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 33418
CALLBACK_FUTURE: asyncio.Future[tuple[str, str | None]] | None = None
CALLBACK_SERVER: asyncio.AbstractServer | None = None


@dataclass
class ExportConfig:
    project: str | None
    project_id: str | None
    start_date: str
    end_date: str
    output: Path
    server_url: str
    token_path: Path
    limit: int
    topic_id: str | None
    tag_id: str | None
    model_id: str | None
    brand_id: str | None
    list_projects: bool
    list_tools: bool
    connect_timeout: float


def main() -> int:
    config = parse_args()
    asyncio.run(export(config))
    return 0


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(
        description="Export Peec MCP chats into a coverage-pipeline CSV."
    )
    parser.add_argument("--project", help="Peec project name or substring.")
    parser.add_argument("--project-id", help="Peec project ID. Overrides --project.")
    parser.add_argument("--start-date", required=False, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end-date", required=False, help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/peec_chats.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--server-url",
        default=PEEC_MCP_URL,
        help="Peec MCP Streamable HTTP URL.",
    )
    parser.add_argument(
        "--token-path",
        type=Path,
        default=TOKEN_PATH,
        help="Local OAuth token storage path.",
    )
    parser.add_argument("--limit", type=int, default=10000, help="Max chats to export.")
    parser.add_argument("--topic-id", help="Only prompts from this topic ID.")
    parser.add_argument("--tag-id", help="Only prompts with this tag ID.")
    parser.add_argument("--model-id", help="Only chats from this Peec model ID.")
    parser.add_argument("--brand-id", help="Only chats mentioning this Peec brand ID.")
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List accessible Peec projects and exit.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List Peec MCP tools and exit.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=60.0,
        help="Connection/read timeout in seconds.",
    )
    args = parser.parse_args()

    if not args.list_projects and not args.list_tools:
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date are required unless listing.")
        if not args.project and not args.project_id:
            parser.error("--project or --project-id is required unless listing.")

    return ExportConfig(
        project=args.project,
        project_id=args.project_id,
        start_date=args.start_date or "",
        end_date=args.end_date or "",
        output=args.output,
        server_url=args.server_url,
        token_path=args.token_path,
        limit=args.limit,
        topic_id=args.topic_id,
        tag_id=args.tag_id,
        model_id=args.model_id,
        brand_id=args.brand_id,
        list_projects=args.list_projects,
        list_tools=args.list_tools,
        connect_timeout=args.connect_timeout,
    )


async def export(config: ExportConfig) -> None:
    print(f"Connecting to Peec MCP at {config.server_url} ...", flush=True)
    access_token = await get_access_token(config)

    async with streamablehttp_client(
        config.server_url,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=config.connect_timeout,
        sse_read_timeout=300,
    ) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            print("Initializing MCP session ...", flush=True)
            await session.initialize()

            if config.list_tools:
                result = await session.list_tools()
                print(result.model_dump_json(indent=2))
                return

            projects = table_from_result(await call_tool(session, "list_projects"))
            if config.list_projects:
                print(projects.to_string(index=False))
                return

            project_id = resolve_project_id(projects, config)
            prompts = await fetch_prompts(session, project_id, config)
            models = table_from_result(await call_tool(session, "list_models", {"project_id": project_id}))
            model_names = dict(zip(models.get("id", []), models.get("name", [])))

            chat_rows = await fetch_chats(session, project_id, config)
            prompt_by_id = {
                str(row["id"]): row
                for _, row in prompts.iterrows()
            }

            records: list[dict[str, Any]] = []
            for _, chat in chat_rows.iterrows():
                chat_id = str(chat["id"])
                chat_detail = await call_tool(
                    session,
                    "get_chat",
                    {"project_id": project_id, "chat_id": chat_id},
                )
                records.append(flatten_chat(chat, chat_detail, prompt_by_id, model_names))

            config.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_records(records).to_csv(
                config.output,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            print(f"Wrote {len(records)} Peec chat rows to {config.output}")


async def get_access_token(config: ExportConfig) -> str:
    storage = JsonTokenStorage(config.token_path)
    tokens = storage.get_tokens()
    if tokens and tokens.get("access_token"):
        return str(tokens["access_token"])

    server_url = config.server_url.rstrip("/")
    redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback"
    client_info = storage.get_client_info()
    if not client_info:
        client_info = register_oauth_client(server_url, redirect_uri)
        storage.set_client_info(client_info)

    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).decode("ascii").rstrip("=")
    state = secrets.token_urlsafe(32)

    await ensure_callback_server()
    authorization_url = (
        f"{server_url}/authorize?"
        + urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": client_info["client_id"],
                "redirect_uri": redirect_uri,
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "resource": server_url,
            }
        )
    )

    print("\nAuthorize Peec MCP in your browser:")
    print(authorization_url, flush=True)
    try:
        webbrowser.open(authorization_url)
    except Exception:
        pass

    assert CALLBACK_FUTURE is not None
    print(f"Waiting for OAuth callback on {redirect_uri} ...", flush=True)
    code, returned_state = await CALLBACK_FUTURE
    if returned_state != state:
        raise RuntimeError("OAuth state mismatch.")

    tokens = exchange_oauth_token(
        server_url=server_url,
        client_id=client_info["client_id"],
        code=code,
        verifier=verifier,
        redirect_uri=redirect_uri,
    )
    storage.set_tokens(tokens)
    return str(tokens["access_token"])


def register_oauth_client(server_url: str, redirect_uri: str) -> dict[str, Any]:
    payload = {
        "redirect_uris": [redirect_uri],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "client_name": "Feature Visibility Coverage MVP",
        "software_id": "feature-visibility-coverage-mvp",
        "software_version": "0.1.0",
    }
    response = post_json(f"{server_url}/register", payload)
    if "client_id" not in response:
        raise RuntimeError(f"OAuth registration response missing client_id: {response}")
    return response


def exchange_oauth_token(
    *,
    server_url: str,
    client_id: str,
    code: str,
    verifier: str,
    redirect_uri: str,
) -> dict[str, Any]:
    return post_form(
        f"{server_url}/token",
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
            "resource": server_url,
        },
    )


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return read_json_response(request)


def post_form(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    return read_json_response(request)


def read_json_response(request: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {request.full_url}: {body}") from exc


async def ensure_callback_server() -> None:
    global CALLBACK_FUTURE, CALLBACK_SERVER
    if CALLBACK_FUTURE is None:
        loop = asyncio.get_running_loop()
        CALLBACK_FUTURE = loop.create_future()
    if CALLBACK_SERVER is None:
        CALLBACK_SERVER = await asyncio.start_server(
            handle_callback_connection,
            CALLBACK_HOST,
            CALLBACK_PORT,
        )


async def handle_callback_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    global CALLBACK_FUTURE
    request_line = await reader.readline()
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break

    target = request_line.decode("utf-8", errors="replace").split(" ")[1]
    parsed = urllib.parse.urlparse(target)
    query = urllib.parse.parse_qs(parsed.query)
    code = query.get("code", [""])[0]
    state = query.get("state", [None])[0]
    error = query.get("error", [""])[0]

    if CALLBACK_FUTURE and not CALLBACK_FUTURE.done():
        if code:
            CALLBACK_FUTURE.set_result((code, state))
            body = "Peec MCP authorization complete. You can return to the terminal."
            status = "200 OK"
        else:
            CALLBACK_FUTURE.set_exception(
                RuntimeError(f"OAuth callback missing code: {error or target}")
            )
            body = "Peec MCP authorization failed. You can return to the terminal."
            status = "400 Bad Request"
    else:
        body = "Peec MCP authorization callback already handled."
        status = "200 OK"

    response = (
        f"HTTP/1.1 {status}\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(body.encode('utf-8'))}\r\n"
        "Connection: close\r\n"
        "\r\n"
        f"{body}"
    )
    writer.write(response.encode("utf-8"))
    await writer.drain()
    writer.close()
    await writer.wait_closed()


class JsonTokenStorage:
    def __init__(self, path: Path) -> None:
        self.path = path

    def get_tokens(self) -> dict[str, Any] | None:
        data = self._read()
        return data.get("tokens")

    def set_tokens(self, tokens: dict[str, Any]) -> None:
        data = self._read()
        data["tokens"] = tokens
        self._write(data)

    def get_client_info(self) -> dict[str, Any] | None:
        data = self._read()
        return data.get("client_info")

    def set_client_info(self, client_info: dict[str, Any]) -> None:
        data = self._read()
        data["client_info"] = client_info
        self._write(data)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


async def call_tool(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = await session.call_tool(name, arguments or {})
    if result.isError:
        raise RuntimeError(f"Peec MCP tool {name} failed: {result}")
    if result.structuredContent is not None:
        return result.structuredContent
    texts = [
        item.text
        for item in result.content
        if getattr(item, "type", None) == "text" and hasattr(item, "text")
    ]
    if texts:
        try:
            return json.loads("\n".join(texts))
        except json.JSONDecodeError:
            return {"text": "\n".join(texts)}
    return result.model_dump(mode="json")


def table_from_result(result: dict[str, Any]) -> pd.DataFrame:
    if "columns" in result and "rows" in result:
        return pd.DataFrame(result["rows"], columns=result["columns"])
    if "data" in result and isinstance(result["data"], list):
        return pd.DataFrame(result["data"])
    if "items" in result and isinstance(result["items"], list):
        return pd.DataFrame(result["items"])
    if isinstance(result, dict):
        return pd.DataFrame([result])
    raise ValueError(f"Unsupported Peec table response: {result}")


def resolve_project_id(projects: pd.DataFrame, config: ExportConfig) -> str:
    if config.project_id:
        return config.project_id
    assert config.project
    if projects.empty:
        raise RuntimeError("Peec MCP returned no projects.")
    if "id" not in projects.columns or "name" not in projects.columns:
        raise RuntimeError(f"Unexpected list_projects columns: {list(projects.columns)}")

    needle = config.project.casefold()
    matches = projects[projects["name"].astype(str).str.casefold().str.contains(needle, regex=False)]
    if matches.empty:
        raise RuntimeError(
            f"No project matched {config.project!r}. Run --list-projects to see available projects."
        )
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple projects matched. Use --project-id. Matches:\n"
            + matches[["id", "name"]].to_string(index=False)
        )
    return str(matches.iloc[0]["id"])


async def fetch_prompts(
    session: ClientSession,
    project_id: str,
    config: ExportConfig,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    offset = 0
    page_size = min(config.limit, 1000)
    while True:
        args: dict[str, Any] = {
            "project_id": project_id,
            "limit": page_size,
            "offset": offset,
        }
        if config.topic_id:
            args["topic_id"] = config.topic_id
        if config.tag_id:
            args["tag_id"] = config.tag_id
        frame = table_from_result(await call_tool(session, "list_prompts", args))
        if frame.empty:
            break
        records.append(frame)
        if len(frame) < page_size:
            break
        offset += len(frame)
        if offset >= config.limit:
            break
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


async def fetch_chats(
    session: ClientSession,
    project_id: str,
    config: ExportConfig,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    offset = 0
    page_size = min(config.limit, 10000)
    while True:
        args: dict[str, Any] = {
            "project_id": project_id,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "limit": page_size,
            "offset": offset,
        }
        if config.model_id:
            args["model_id"] = config.model_id
        if config.brand_id:
            args["brand_id"] = config.brand_id
        frame = table_from_result(await call_tool(session, "list_chats", args))
        if frame.empty:
            break
        records.append(frame)
        if len(frame) < page_size:
            break
        offset += len(frame)
        if offset >= config.limit:
            break
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def flatten_chat(
    chat_row: pd.Series,
    chat: dict[str, Any],
    prompt_by_id: dict[str, pd.Series],
    model_names: dict[Any, Any],
) -> dict[str, Any]:
    prompt_obj = chat.get("prompt") if isinstance(chat.get("prompt"), dict) else {}
    model_obj = chat.get("model") if isinstance(chat.get("model"), dict) else {}
    prompt_id = str(prompt_obj.get("id") or chat_row.get("prompt_id") or "")
    model_id = str(model_obj.get("id") or chat_row.get("model_id") or "")
    prompt_row = prompt_by_id.get(prompt_id)

    prompt_text = (
        prompt_obj.get("text")
        or (prompt_row.get("text") if prompt_row is not None and "text" in prompt_row else "")
        or extract_user_prompt(chat.get("messages"))
    )
    response_text = extract_assistant_response(chat.get("messages"))

    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "chat_id": chat.get("id") or chat_row.get("id"),
        "date": chat.get("date") or chat_row.get("date"),
        "model_id": model_id,
        "engine": model_obj.get("name") or model_names.get(model_id, model_id),
        "response": response_text,
        "brands_mentioned": json.dumps(chat.get("brands_mentioned", []), ensure_ascii=False),
        "sources": json.dumps(chat.get("sources", []), ensure_ascii=False),
        "queries": json.dumps(chat.get("queries", []), ensure_ascii=False),
        "products": json.dumps(chat.get("products", []), ensure_ascii=False),
    }


def extract_user_prompt(messages: Any) -> str:
    for message in iter_messages(messages):
        role = str(message.get("role", "")).lower()
        if role in {"user", "human"}:
            return stringify_content(message)
    return ""


def extract_assistant_response(messages: Any) -> str:
    chunks: list[str] = []
    for message in iter_messages(messages):
        role = str(message.get("role", "")).lower()
        if role in {"assistant", "ai", "model"}:
            chunks.append(stringify_content(message))
    return "\n".join(chunk for chunk in chunks if chunk)


def iter_messages(messages: Any) -> Iterable[dict[str, Any]]:
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                yield message


def stringify_content(message: dict[str, Any]) -> str:
    content = message.get("content") or message.get("text") or message.get("message") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return json.dumps(content, ensure_ascii=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
