#!/usr/bin/env python3
"""
MCP wrapper around the feature visibility coverage pipeline.

This server exposes the repo's existing capabilities as MCP tools:

- validate_csv_inputs
- run_visibility_coverage
- summarize_feature_gaps
- export_peec_chats

The implementation intentionally reuses the existing CLI scripts rather than
duplicating pipeline logic in a second codepath.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP


ROOT = Path(__file__).resolve().parent.parent
VISIBILITY_SCRIPT = ROOT / "src" / "visibility_mvp.py"
PEEC_EXPORT_SCRIPT = ROOT / "src" / "peec_mcp_export.py"

PROMPT_COLUMNS = ("prompt", "raw_prompt", "query", "question", "text")
RESPONSE_COLUMNS = ("response", "answer", "ai_response", "model_response", "output")
FEATURE_NAME_COLUMNS = ("feature", "feature_name", "name", "title")
FEATURE_DESCRIPTION_COLUMNS = ("description", "feature_description", "desc")
BRAND_NAME_COLUMNS = ("brand", "brand_name", "name")


mcp = FastMCP(
    name="Feature Visibility Coverage",
    instructions=(
        "Run feature visibility coverage jobs over prompt/response data, feature "
        "descriptions, and tracked brands. Use validate_csv_inputs before a run "
        "when the CSV shape is uncertain."
    ),
    log_level="WARNING",
)


def _run_command(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "command": args,
    }


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _find_column(columns: list[str], options: tuple[str, ...]) -> str | None:
    lowered = {column.casefold(): column for column in columns}
    for option in options:
        if option.casefold() in lowered:
            return lowered[option.casefold()]
    return None


def _preview_overview(path: Path, limit: int = 10) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    preferred = [
        "mapped_feature_name",
        "cluster_label",
        "visibility_share",
        "gap_severity",
        "signal",
        "top_competitor_brand_name",
    ]
    columns = [column for column in preferred if column in frame.columns]
    return frame[columns].head(limit).to_dict(orient="records")


@mcp.tool()
def validate_csv_inputs(
    prompts_csv: str,
    features_csv: str,
    brands_csv: str | None = None,
) -> dict[str, Any]:
    """
    Validate the expected CSV contract for prompts, features, and brands.
    """
    prompts_path = Path(prompts_csv).expanduser().resolve()
    features_path = Path(features_csv).expanduser().resolve()
    brands_path = Path(brands_csv).expanduser().resolve() if brands_csv else None

    prompts_columns = _read_header(prompts_path)
    features_columns = _read_header(features_path)
    brands_columns = _read_header(brands_path) if brands_path else []

    prompt_column = _find_column(prompts_columns, PROMPT_COLUMNS)
    response_column = _find_column(prompts_columns, RESPONSE_COLUMNS)
    feature_name_column = _find_column(features_columns, FEATURE_NAME_COLUMNS)
    feature_description_column = _find_column(features_columns, FEATURE_DESCRIPTION_COLUMNS)
    brand_name_column = _find_column(brands_columns, BRAND_NAME_COLUMNS) if brands_path else None

    issues: list[str] = []
    if not prompt_column:
        issues.append("Prompts CSV is missing a prompt column.")
    if not feature_name_column:
        issues.append("Features CSV is missing a feature name column.")
    if not feature_description_column:
        issues.append("Features CSV is missing a feature description column.")
    if brands_path and not brand_name_column:
        issues.append("Brands CSV is missing a brand name column.")

    return {
        "ok": not issues,
        "issues": issues,
        "prompts_csv": str(prompts_path),
        "features_csv": str(features_path),
        "brands_csv": str(brands_path) if brands_path else None,
        "prompt_column": prompt_column,
        "response_column": response_column,
        "feature_name_column": feature_name_column,
        "feature_description_column": feature_description_column,
        "brand_name_column": brand_name_column,
        "prompt_columns": prompts_columns,
        "feature_columns": features_columns,
        "brand_columns": brands_columns,
    }


@mcp.tool()
def run_visibility_coverage(
    prompts_csv: str,
    features_csv: str,
    output_dir: str,
    brands_csv: str | None = None,
    brand: str | None = None,
    target_brand: str | None = None,
    target_brand_id: str | None = None,
    embedding_backend: str = "hash",
    normalizer: str = "openai_mock",
    brand_detector: str = "openai_mock",
    cluster_threshold: float = 0.58,
    feature_threshold: float = 0.2,
    min_cluster_size: int = 2,
    min_coverage_n: int = 1,
    aggregation_mode: str = "response",
) -> dict[str, Any]:
    """
    Run the coverage pipeline and return output paths plus a preview of feature gaps.

    Defaults are tuned for fast local iteration. For higher-fidelity runs, use
    `embedding_backend="bge-m3"` and switch `normalizer` / `brand_detector`
    from the mock modes to real ones.
    """
    if not brands_csv and not brand:
        raise ValueError("Provide either brands_csv or brand.")

    prompts_path = Path(prompts_csv).expanduser().resolve()
    features_path = Path(features_csv).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    args = [
        sys.executable,
        str(VISIBILITY_SCRIPT),
        "--prompts",
        str(prompts_path),
        "--features",
        str(features_path),
        "--output-dir",
        str(output_path),
        "--embedding-backend",
        embedding_backend,
        "--normalizer",
        normalizer,
        "--brand-detector",
        brand_detector,
        "--cluster-threshold",
        str(cluster_threshold),
        "--feature-threshold",
        str(feature_threshold),
        "--min-cluster-size",
        str(min_cluster_size),
        "--min-coverage-n",
        str(min_coverage_n),
        "--aggregation-mode",
        aggregation_mode,
    ]
    if brands_csv:
        args.extend(["--brands", str(Path(brands_csv).expanduser().resolve())])
    if brand:
        args.extend(["--brand", brand])
    if target_brand:
        args.extend(["--target-brand", target_brand])
    if target_brand_id:
        args.extend(["--target-brand-id", target_brand_id])

    result = _run_command(args)
    overview_path = output_path / "feature_gap_overview.csv"
    details_path = output_path / "feature_gap_details.csv"
    summary_path = output_path / "feature_gap_summary.md"
    metadata_path = output_path / "run_metadata.json"

    return {
        **result,
        "output_dir": str(output_path),
        "overview_csv": str(overview_path),
        "details_csv": str(details_path),
        "summary_md": str(summary_path),
        "metadata_json": str(metadata_path),
        "overview_preview": _preview_overview(overview_path) if result["ok"] else [],
    }


@mcp.tool()
def summarize_feature_gaps(
    output_dir: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Read the feature-gap outputs from a previous run and return a concise summary.
    """
    output_path = Path(output_dir).expanduser().resolve()
    overview_path = output_path / "feature_gap_overview.csv"
    summary_path = output_path / "feature_gap_summary.md"

    if not overview_path.exists():
        raise FileNotFoundError(f"Missing {overview_path}")

    overview = pd.read_csv(overview_path)
    if overview.empty:
        return {
            "ok": True,
            "output_dir": str(output_path),
            "row_count": 0,
            "preview": [],
            "summary_markdown": summary_path.read_text(encoding="utf-8") if summary_path.exists() else "",
        }

    sort_columns = [column for column in ("gap_severity", "visibility_share") if column in overview.columns]
    ascending = [True, True][: len(sort_columns)]
    preview = overview.sort_values(sort_columns, ascending=ascending).head(limit)

    return {
        "ok": True,
        "output_dir": str(output_path),
        "row_count": int(len(overview)),
        "preview": preview.to_dict(orient="records"),
        "summary_markdown": summary_path.read_text(encoding="utf-8") if summary_path.exists() else "",
    }


@mcp.tool()
def export_peec_chats(
    project_id: str,
    start_date: str,
    end_date: str,
    output_csv: str,
    topic_id: str | None = None,
    tag_id: str | None = None,
    model_id: str | None = None,
    brand_id: str | None = None,
    limit: int = 10000,
    connect_timeout: float = 60.0,
) -> dict[str, Any]:
    """
    Export tracked Peec MCP chats into the prompts CSV shape used by the pipeline.
    """
    output_path = Path(output_csv).expanduser().resolve()
    args = [
        sys.executable,
        str(PEEC_EXPORT_SCRIPT),
        "--project-id",
        project_id,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--output",
        str(output_path),
        "--limit",
        str(limit),
        "--connect-timeout",
        str(connect_timeout),
    ]
    if topic_id:
        args.extend(["--topic-id", topic_id])
    if tag_id:
        args.extend(["--tag-id", tag_id])
    if model_id:
        args.extend(["--model-id", model_id])
    if brand_id:
        args.extend(["--brand-id", brand_id])

    result = _run_command(args)
    row_count = 0
    if result["ok"] and output_path.exists():
        row_count = int(len(pd.read_csv(output_path)))

    return {
        **result,
        "output_csv": str(output_path),
        "row_count": row_count,
    }


if __name__ == "__main__":
    mcp.run()
