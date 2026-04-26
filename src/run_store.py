from __future__ import annotations

import csv
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from visibility_mvp import (
    Config,
    build_cluster_summary,
    build_coverage,
    build_feature_gap_details,
    build_feature_gap_overview,
    build_pm_summary,
    resolve_target_brand,
)


ROOT = Path(__file__).resolve().parent.parent
STORE_ROOT = ROOT / ".cache" / "feature_visibility" / "system"
RUNS_DIR = STORE_ROOT / "runs"
RUNS_INDEX = STORE_ROOT / "runs_index.csv"
FEATURE_SETS_DIR = STORE_ROOT / "feature_sets"
FEATURE_SETS_INDEX = STORE_ROOT / "feature_sets.csv"

OUTPUT_FILES = [
    "query_mapping.csv",
    "coverage_by_feature_cluster.csv",
    "clusters.csv",
    "feature_gap_overview.csv",
    "feature_gap_details.csv",
    "feature_gap_summary.md",
    "run_metadata.json",
]


def ensure_store() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_SETS_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_id() -> str:
    return f"run_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"


def feature_set_id(features_csv: Path) -> str:
    stamp = features_csv.stat().st_mtime_ns
    size = features_csv.stat().st_size
    key = f"{features_csv.resolve()}::{stamp}::{size}"
    return f"features_{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:12]}"


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def persist_feature_set(features_csv: Path) -> dict[str, Any]:
    ensure_store()
    fs_id = feature_set_id(features_csv)
    target = FEATURE_SETS_DIR / f"{fs_id}.csv"
    if not target.exists():
        shutil.copy2(features_csv, target)
        frame = pd.read_csv(target)
        append_csv_row(
            FEATURE_SETS_INDEX,
            {
                "feature_set_id": fs_id,
                "created_at": now_iso(),
                "path": str(target),
                "feature_count": len(frame),
            },
            ["feature_set_id", "created_at", "path", "feature_count"],
        )
    return {
        "feature_set_id": fs_id,
        "path": str(target),
    }


def persist_run(
    *,
    source: str,
    prompts_csv: Path,
    features_csv: Path,
    brands_csv: Path,
    output_dir: Path,
    target_brand: str,
    aggregation_mode: str,
    extra_meta: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    ensure_store()
    rid = run_id()
    run_dir = RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_info = persist_feature_set(features_csv)
    shutil.copy2(prompts_csv, run_dir / "prompts.csv")
    shutil.copy2(features_csv, run_dir / "features.csv")
    shutil.copy2(brands_csv, run_dir / "brands.csv")
    for name in OUTPUT_FILES:
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, run_dir / name)

    metadata = {}
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    overview_rows = dataframe_records(run_dir / "feature_gap_overview.csv")
    mapping_rows = dataframe_records(run_dir / "query_mapping.csv")
    manifest = {
        "run_id": rid,
        "created_at": now_iso(),
        "source": source,
        "parent_run_id": parent_run_id or "",
        "feature_set_id": feature_info["feature_set_id"],
        "target_brand": target_brand,
        "aggregation_mode": aggregation_mode,
        "prompt_rows": len(mapping_rows),
        "overview_rows": len(overview_rows),
        "paths": {
            "run_dir": str(run_dir),
            "prompts_csv": str(run_dir / "prompts.csv"),
            "features_csv": str(run_dir / "features.csv"),
            "brands_csv": str(run_dir / "brands.csv"),
        },
        "metadata": metadata,
        "extra_meta": extra_meta or {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    append_csv_row(
        RUNS_INDEX,
        {
            "run_id": rid,
            "created_at": manifest["created_at"],
            "source": source,
            "parent_run_id": manifest["parent_run_id"],
            "feature_set_id": feature_info["feature_set_id"],
            "target_brand": target_brand,
            "aggregation_mode": aggregation_mode,
            "prompt_rows": len(mapping_rows),
            "overview_rows": len(overview_rows),
            "run_dir": str(run_dir),
        },
        [
            "run_id",
            "created_at",
            "source",
            "parent_run_id",
            "feature_set_id",
            "target_brand",
            "aggregation_mode",
            "prompt_rows",
            "overview_rows",
            "run_dir",
        ],
    )
    return manifest


def dataframe_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    if not RUNS_INDEX.exists():
        return []
    frame = pd.read_csv(RUNS_INDEX)
    if frame.empty:
        return []
    frame = frame.sort_values("created_at", ascending=False).head(limit)
    return json.loads(frame.to_json(orient="records"))


def load_manifest(run_id_value: str) -> dict[str, Any]:
    path = RUNS_DIR / run_id_value / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Run {run_id_value!r} was not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_result(run_id_value: str) -> dict[str, Any]:
    manifest = load_manifest(run_id_value)
    run_dir = Path(manifest["paths"]["run_dir"])
    return {
        "ok": True,
        "run_id": run_id_value,
        "overview": dataframe_records(run_dir / "feature_gap_overview.csv"),
        "details": dataframe_records(run_dir / "feature_gap_details.csv"),
        "coverage": dataframe_records(run_dir / "coverage_by_feature_cluster.csv"),
        "summary": read_text(run_dir / "feature_gap_summary.md"),
        "metadata": manifest.get("metadata", {}),
        "manifest": manifest,
    }


def load_run_mappings(
    run_id_value: str,
    *,
    feature_id: str = "",
    cluster_id: str = "",
    limit: int = 200,
) -> list[dict[str, Any]]:
    manifest = load_manifest(run_id_value)
    frame = pd.read_csv(Path(manifest["paths"]["run_dir"]) / "query_mapping.csv")
    target_brand = str(manifest.get("target_brand") or "")
    if target_brand and "brand_name" in frame.columns:
        frame = frame[frame["brand_name"].astype(str) == target_brand].copy()
    if feature_id and "mapped_feature_id" in frame.columns:
        frame = frame[frame["mapped_feature_id"].astype(str) == feature_id].copy()
    if cluster_id and "cluster_id" in frame.columns:
        frame = frame[frame["cluster_id"].astype(str) == cluster_id].copy()
    keep_cols = [
        column
        for column in [
            "prompt_id",
            "original_prompt",
            "canonical_query",
            "cluster_id",
            "cluster_label",
            "mapped_feature_id",
            "mapped_feature_name",
            "feature_similarity",
            "feature_present",
            "feature_evidence_strength",
            "source_domains",
            "engine",
            "brand_present",
            "brand_detection_source",
        ]
        if column in frame.columns
    ]
    if not keep_cols:
        keep_cols = list(frame.columns)
    frame = frame[keep_cols].drop_duplicates().head(limit)
    return json.loads(frame.to_json(orient="records"))


def reaggregate_saved_run(run_id_value: str, aggregation_mode: str) -> dict[str, Any]:
    manifest = load_manifest(run_id_value)
    run_dir = Path(manifest["paths"]["run_dir"])
    brand_rows = pd.read_csv(run_dir / "query_mapping.csv")
    brands = pd.read_csv(run_dir / "brands.csv")
    metadata = manifest.get("metadata", {})
    prompts = (
        brand_rows[
            [
                column
                for column in [
                    "prompt_id",
                    "canonical_query",
                    "cluster_id",
                    "cluster_label",
                    "mapped_feature_id",
                    "mapped_feature_name",
                ]
                if column in brand_rows.columns
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    config = Config(
        prompts_csv=run_dir / "prompts.csv",
        features_csv=run_dir / "features.csv",
        brand=None,
        brands_csv=run_dir / "brands.csv",
        target_brand=manifest["target_brand"],
        target_brand_id=metadata.get("target_brand_id"),
        output_dir=Path(tempfile.mkdtemp(prefix="feature_visibility_reaggregate_")),
        embedding_backend=str(metadata.get("embedding_backend", "hash")),
        embedding_model=str(metadata.get("embedding_model", "BAAI/bge-m3")),
        normalizer=str(metadata.get("normalizer", "openai_mock")),
        normalizer_model=str(metadata.get("normalizer_model", "gpt-4.1-mini")),
        brand_detector=str(metadata.get("brand_detector", "openai_mock")),
        brand_detector_model=str(metadata.get("brand_detector_model", "gpt-4.1-mini")),
        feature_evidence_mode=str(metadata.get("feature_evidence_mode", "openai_mock")),
        feature_evidence_model=str(metadata.get("feature_evidence_model", "gpt-4.1-mini")),
        cluster_threshold=float(metadata.get("cluster_threshold", 0.2)),
        feature_threshold=float(metadata.get("feature_threshold", 0.05)),
        min_cluster_size=int(metadata.get("min_cluster_size", 2)),
        min_coverage_n=int(metadata.get("min_coverage_n", 1)),
        aggregation_mode=aggregation_mode,
    )

    coverage = build_coverage(brand_rows, config.min_coverage_n, aggregation_mode)
    cluster_summary = build_cluster_summary(prompts)
    target_brand = resolve_target_brand(brands, config)
    feature_gap_overview = build_feature_gap_overview(coverage, brand_rows, target_brand)
    feature_gap_details = build_feature_gap_details(coverage, brand_rows, target_brand)
    pm_summary = build_pm_summary(feature_gap_overview, target_brand)

    output_dir = config.output_dir
    brand_rows.to_csv(output_dir / "query_mapping.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    coverage.to_csv(output_dir / "coverage_by_feature_cluster.csv", index=False)
    cluster_summary.to_csv(output_dir / "clusters.csv", index=False)
    feature_gap_overview.to_csv(output_dir / "feature_gap_overview.csv", index=False)
    feature_gap_details.to_csv(output_dir / "feature_gap_details.csv", index=False)
    (output_dir / "feature_gap_summary.md").write_text(pm_summary, encoding="utf-8")
    metadata["aggregation_mode"] = aggregation_mode
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return persist_run(
        source="saved_run",
        prompts_csv=run_dir / "prompts.csv",
        features_csv=run_dir / "features.csv",
        brands_csv=run_dir / "brands.csv",
        output_dir=output_dir,
        target_brand=manifest["target_brand"],
        aggregation_mode=aggregation_mode,
        extra_meta={"reaggregated_from": run_id_value},
        parent_run_id=run_id_value,
    )
