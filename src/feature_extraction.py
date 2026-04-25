#!/usr/bin/env python3
"""
Feature extraction helpers for uploaded PDFs and free-form feature docs.

Modes:
- mock: deterministic heuristic extractor for fast local iteration
- openai: LLM judge/extractor that converts product docs into feature rows
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


DEFAULT_MODEL = "gpt-4.1-mini"


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def extract_pdf_text_from_path(path: str | Path) -> str:
    return extract_pdf_text(Path(path).read_bytes())


def features_from_text(text: str, mode: str = "mock", model: str = DEFAULT_MODEL) -> pd.DataFrame:
    cleaned = normalize_text(text)
    if not cleaned:
        return empty_features()
    if mode == "openai":
        return extract_features_openai(cleaned, model)
    return extract_features_mock(cleaned)


def extract_features_from_pdf(
    pdf_bytes: bytes,
    mode: str = "mock",
    model: str = DEFAULT_MODEL,
) -> tuple[pd.DataFrame, str]:
    text = extract_pdf_text(pdf_bytes)
    frame = features_from_text(text, mode=mode, model=model)
    return frame, text


def empty_features() -> pd.DataFrame:
    return pd.DataFrame(columns=["feature_id", "feature_name", "description"])


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_features_mock(text: str) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    sections = split_heading_sections(text)
    if not sections:
        sections = split_candidate_sections(text)
    for section in sections:
        name, description = parse_section_mock(section)
        if not name or not description:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "feature_id": f"fx_{len(records) + 1:03d}",
                "feature_name": name,
                "description": description,
            }
        )
    return pd.DataFrame.from_records(records or [], columns=["feature_id", "feature_name", "description"])


def split_heading_sections(text: str) -> list[str]:
    pattern = re.compile(r"(?P<name>[A-Z][A-Za-z0-9&/\- ]{2,80}):")
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return []

    sections: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if block:
            sections.append(block)
    return sections


def split_candidate_sections(text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n", text)
    candidates: list[str] = []
    for chunk in chunks:
        lines = [line.strip(" -•\t") for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) == 1 and len(lines[0].split()) <= 8:
            continue
        candidates.append("\n".join(lines))
    return candidates


def parse_section_mock(section: str) -> tuple[str, str]:
    lines = [line.strip(" -•\t") for line in section.splitlines() if line.strip()]
    if not lines:
        return "", ""

    first = lines[0]
    if ":" in first and len(first.split(":")[0].split()) <= 8:
        lhs, rhs = first.split(":", 1)
        name = clean_feature_name(lhs)
        description_lines = [rhs.strip()] + lines[1:]
    else:
        name = clean_feature_name(first)
        description_lines = lines[1:]

    description = " ".join(part.strip() for part in description_lines if part.strip())
    description = clean_description(description)
    if not name or len(name) < 3:
        return "", ""
    if not description:
        return "", ""
    if looks_like_metadata(name, description):
        return "", ""
    return name, description


def clean_feature_name(value: str) -> str:
    value = re.sub(r"^[0-9]+[.)]\s*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" .:-")
    words = value.split()
    if len(words) > 8:
        value = " ".join(words[:8])
    return value.strip()


def clean_description(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value


def looks_like_metadata(name: str, description: str) -> bool:
    combined = f"{name} {description}".casefold()
    banned = [
        "table of contents",
        "overview",
        "agenda",
        "appendix",
        "introduction",
        "contact",
        "copyright",
    ]
    return any(token in combined for token in banned)


def extract_features_openai(text: str, model: str) -> pd.DataFrame:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Feature extraction mode openai requires OPENAI_API_KEY.")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract product features from the provided document text. "
                    "Return only JSON with this shape: "
                    "{\"features\": [{\"feature_name\": \"...\", \"description\": \"...\"}]}. "
                    "Each feature must be specific and usable for semantic matching against user queries. "
                    "Ignore general company background, team notes, pricing, and appendix text."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"document_text": text[:50000]}, ensure_ascii=False),
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI feature extraction call failed: {exc}") from exc

    content = body["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI feature extraction returned invalid JSON: {content}") from exc

    features = parsed.get("features", [])
    records: list[dict[str, str]] = []
    for idx, item in enumerate(features, start=1):
        name = clean_feature_name(str(item.get("feature_name", "")))
        description = clean_description(str(item.get("description", "")))
        if not name or not description:
            continue
        records.append(
            {
                "feature_id": f"fx_{idx:03d}",
                "feature_name": name,
                "description": description,
            }
        )
    return pd.DataFrame.from_records(records, columns=["feature_id", "feature_name", "description"])
