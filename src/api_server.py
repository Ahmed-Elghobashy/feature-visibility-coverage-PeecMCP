#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route


ROOT = Path(__file__).resolve().parent.parent
VISIBILITY_SCRIPT = ROOT / "src" / "visibility_mvp.py"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feature_extraction import extract_features_from_pdf  # noqa: E402


def dataframe_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def run_pipeline(
    *,
    prompts_csv: Path,
    features_csv: Path,
    brands_csv: Path,
    target_brand: str,
    output_dir: Path,
    normalizer: str,
    brand_detector: str,
    embedding_backend: str,
    aggregation_mode: str,
) -> dict[str, Any]:
    args = [
        sys.executable,
        str(VISIBILITY_SCRIPT),
        "--prompts",
        str(prompts_csv),
        "--features",
        str(features_csv),
        "--brands",
        str(brands_csv),
        "--target-brand",
        target_brand,
        "--normalizer",
        normalizer,
        "--brand-detector",
        brand_detector,
        "--embedding-backend",
        embedding_backend,
        "--aggregation-mode",
        aggregation_mode,
        "--output-dir",
        str(output_dir),
    ]
    if embedding_backend == "hash":
        args.extend(["--feature-threshold", "0.05", "--cluster-threshold", "0.2"])

    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            "ok": False,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {
        "ok": True,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "overview": dataframe_records(output_dir / "feature_gap_overview.csv"),
        "details": dataframe_records(output_dir / "feature_gap_details.csv"),
        "coverage": dataframe_records(output_dir / "coverage_by_feature_cluster.csv"),
        "summary": read_text(output_dir / "feature_gap_summary.md"),
        "metadata": metadata,
    }


async def save_upload(upload: UploadFile, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = await upload.read()
    path.write_bytes(data)
    return path


def first_brand_name(brands_csv: Path) -> str:
    with brands_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("brand_name") or row.get("brand") or row.get("name")
            if value:
                return value
    raise ValueError("Brands CSV did not contain a usable brand name.")


async def health(_request) -> JSONResponse:
    return JSONResponse({"ok": True})


async def analyze_sample(request) -> JSONResponse:
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    workdir = Path(tempfile.mkdtemp(prefix="feature_visibility_api_sample_"))
    try:
        output_dir = workdir / "outputs"
        result = run_pipeline(
            prompts_csv=ROOT / "data" / "sample_prompts.csv",
            features_csv=ROOT / "data" / "demo_features.csv",
            brands_csv=ROOT / "data" / "demo_brands.csv",
            target_brand=str(body.get("target_brand") or "Peec AI"),
            output_dir=output_dir,
            normalizer=str(body.get("normalizer") or "openai_mock"),
            brand_detector=str(body.get("brand_detector") or "openai_mock"),
            embedding_backend=str(body.get("embedding_backend") or "hash"),
            aggregation_mode=str(body.get("aggregation_mode") or "prompt"),
        )
        return JSONResponse(result, status_code=200 if result["ok"] else 500)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def analyze_upload(request) -> JSONResponse:
    form = await request.form()
    prompts = form.get("prompts_csv")
    brands = form.get("brands_csv")
    features = form.get("features_csv")
    feature_pdf = form.get("feature_pdf")
    if not isinstance(prompts, UploadFile) or not isinstance(brands, UploadFile):
        return JSONResponse({"ok": False, "error": "prompts_csv and brands_csv are required."}, status_code=400)
    if not isinstance(features, UploadFile) and not isinstance(feature_pdf, UploadFile):
        return JSONResponse({"ok": False, "error": "Upload features_csv or feature_pdf."}, status_code=400)

    workdir = Path(tempfile.mkdtemp(prefix="feature_visibility_api_"))
    try:
        inputs_dir = workdir / "inputs"
        prompts_path = await save_upload(prompts, inputs_dir / "prompts.csv")
        brands_path = await save_upload(brands, inputs_dir / "brands.csv")

        extracted_features: list[dict[str, Any]] = []
        extracted_text = ""
        if isinstance(features, UploadFile):
            features_path = await save_upload(features, inputs_dir / "features.csv")
        else:
            assert isinstance(feature_pdf, UploadFile)
            feature_mode = str(form.get("feature_mode") or "mock")
            frame, extracted_text = extract_features_from_pdf(await feature_pdf.read(), mode=feature_mode)
            if frame.empty:
                return JSONResponse({"ok": False, "error": "No usable features were extracted from the PDF."}, status_code=400)
            features_path = inputs_dir / "features_from_pdf.csv"
            frame.to_csv(features_path, index=False)
            extracted_features = json.loads(frame.to_json(orient="records"))

        target_brand = str(form.get("target_brand") or first_brand_name(brands_path))
        result = run_pipeline(
            prompts_csv=prompts_path,
            features_csv=features_path,
            brands_csv=brands_path,
            target_brand=target_brand,
            output_dir=workdir / "outputs",
            normalizer=str(form.get("normalizer") or "openai_mock"),
            brand_detector=str(form.get("brand_detector") or "openai_mock"),
            embedding_backend=str(form.get("embedding_backend") or "hash"),
            aggregation_mode=str(form.get("aggregation_mode") or "prompt"),
        )
        result["extracted_features"] = extracted_features
        result["extracted_text_preview"] = extracted_text[:12000]
        return JSONResponse(result, status_code=200 if result["ok"] else 500)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


routes = [
    Route("/api/health", health, methods=["GET"]),
    Route("/api/analyze-sample", analyze_sample, methods=["POST"]),
    Route("/api/analyze", analyze_upload, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
