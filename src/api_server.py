#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


ROOT = Path(__file__).resolve().parent.parent
VISIBILITY_SCRIPT = ROOT / "src" / "visibility_mvp.py"
PEEC_EXPORT_SCRIPT = ROOT / "src" / "peec_mcp_export.py"
CACHE_ROOT = ROOT / ".cache" / "feature_visibility"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feature_extraction import extract_features_from_pdf  # noqa: E402
from run_store import load_run_mappings, load_run_result, list_runs, persist_run, reaggregate_saved_run  # noqa: E402


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ROOT / ".env")


def now_ms() -> int:
    return int(time.time() * 1000)


def progress_event(event_type: str, **payload: Any) -> bytes:
    body = {"type": event_type, "timestamp_ms": now_ms(), **payload}
    return (json.dumps(body) + "\n").encode("utf-8")


def step_started(name: str, message: str) -> tuple[bytes, float]:
    return progress_event("stage", stage=name, status="running", message=message), time.perf_counter()


def step_finished(name: str, started_at: float, message: str, **extra: Any) -> bytes:
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    return progress_event("stage", stage=name, status="completed", message=message, duration_ms=duration_ms, **extra)


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
    feature_evidence_mode: str,
    embedding_backend: str,
    aggregation_mode: str,
) -> dict[str, Any]:
    return run_pipeline_capture(
        prompts_csv=prompts_csv,
        features_csv=features_csv,
        brands_csv=brands_csv,
        target_brand=target_brand,
        output_dir=output_dir,
        normalizer=normalizer,
        brand_detector=brand_detector,
        feature_evidence_mode=feature_evidence_mode,
        embedding_backend=embedding_backend,
        aggregation_mode=aggregation_mode,
    )


def visibility_args(
    *,
    prompts_csv: Path,
    features_csv: Path,
    brands_csv: Path,
    target_brand: str,
    output_dir: Path,
    normalizer: str,
    brand_detector: str,
    feature_evidence_mode: str,
    embedding_backend: str,
    aggregation_mode: str,
) -> list[str]:
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
        "--feature-evidence-mode",
        feature_evidence_mode,
        "--embedding-backend",
        embedding_backend,
        "--aggregation-mode",
        aggregation_mode,
        "--output-dir",
        str(output_dir),
    ]
    if embedding_backend == "hash":
        args.extend(["--feature-threshold", "0.05", "--cluster-threshold", "0.2"])
    return args


def parse_visibility_stdout(stdout: str) -> tuple[list[dict[str, Any]], str]:
    progress: list[dict[str, Any]] = []
    plain_lines: list[str] = []
    prefix = "__FV_PROGRESS__ "
    for raw_line in stdout.splitlines():
        if raw_line.startswith(prefix):
            try:
                progress.append(json.loads(raw_line[len(prefix):]))
            except json.JSONDecodeError:
                plain_lines.append(raw_line)
        else:
            plain_lines.append(raw_line)
    cleaned = "\n".join(plain_lines).strip()
    return progress, (cleaned + "\n" if cleaned else "")


def run_pipeline_capture(
    *,
    prompts_csv: Path,
    features_csv: Path,
    brands_csv: Path,
    target_brand: str,
    output_dir: Path,
    normalizer: str,
    brand_detector: str,
    feature_evidence_mode: str,
    embedding_backend: str,
    aggregation_mode: str,
) -> dict[str, Any]:
    args = visibility_args(
        prompts_csv=prompts_csv,
        features_csv=features_csv,
        brands_csv=brands_csv,
        target_brand=target_brand,
        output_dir=output_dir,
        normalizer=normalizer,
        brand_detector=brand_detector,
        feature_evidence_mode=feature_evidence_mode,
        embedding_backend=embedding_backend,
        aggregation_mode=aggregation_mode,
    )
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    progress, cleaned_stdout = parse_visibility_stdout(result.stdout)
    if result.returncode != 0:
        return {
            "ok": False,
            "stdout": cleaned_stdout,
            "stderr": result.stderr,
            "progress": progress,
        }

    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {
        "ok": True,
        "stdout": cleaned_stdout,
        "stderr": result.stderr,
        "overview": dataframe_records(output_dir / "feature_gap_overview.csv"),
        "details": dataframe_records(output_dir / "feature_gap_details.csv"),
        "coverage": dataframe_records(output_dir / "coverage_by_feature_cluster.csv"),
        "summary": read_text(output_dir / "feature_gap_summary.md"),
        "metadata": metadata,
        "progress": progress,
    }


def run_pipeline_stream(
    *,
    prompts_csv: Path,
    features_csv: Path,
    brands_csv: Path,
    target_brand: str,
    output_dir: Path,
    normalizer: str,
    brand_detector: str,
    feature_evidence_mode: str,
    embedding_backend: str,
    aggregation_mode: str,
):
    args = visibility_args(
        prompts_csv=prompts_csv,
        features_csv=features_csv,
        brands_csv=brands_csv,
        target_brand=target_brand,
        output_dir=output_dir,
        normalizer=normalizer,
        brand_detector=brand_detector,
        feature_evidence_mode=feature_evidence_mode,
        embedding_backend=embedding_backend,
        aggregation_mode=aggregation_mode,
    )
    return subprocess.Popen(
        args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


async def save_upload(upload: UploadFile, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = await upload.read()
    path.write_bytes(data)
    return path


def run_peec_export(
    *,
    project_id: str,
    start_date: str,
    end_date: str,
    output_csv: Path,
    limit: int = 250,
) -> dict[str, Any]:
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "project_id": project_id,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_dir = CACHE_ROOT / "peec_exports"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_csv = cache_dir / f"{cache_key}.csv"
    if cached_csv.exists():
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_csv, output_csv)
        return {
            "ok": True,
            "stdout": f"Loaded cached Peec export from {cached_csv}\n",
            "stderr": "",
        }

    args = [
        sys.executable,
        str(PEEC_EXPORT_SCRIPT),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--output",
        str(output_csv),
        "--limit",
        str(limit),
        "--connect-timeout",
        "30",
    ]
    if project_id:
        args.extend(["--project-id", project_id])
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and output_csv.exists():
        shutil.copy2(output_csv, cached_csv)
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def summarize_peec_export_error(stdout: str, stderr: str) -> str:
    text = (stderr or stdout).strip()
    if not text:
        return "Peec export failed."

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.lower().startswith("error:"):
            return line.removeprefix("error:").strip()
    for line in reversed(lines):
        if "Peec MCP" in line or "Unauthorized" in line or "temporarily unavailable" in line:
            return line
    return lines[-1]


def summarize_pipeline_error(stdout: str, stderr: str) -> str:
    text = (stderr or stdout).strip()
    if not text:
        return "Visibility coverage pipeline failed."
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1]


def first_brand_name(brands_csv: Path) -> str:
    names = brand_names(brands_csv)
    if names:
        return names[0]
    raise ValueError("Brands CSV did not contain a usable brand name.")


def brand_names(brands_csv: Path) -> list[str]:
    names: list[str] = []
    with brands_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("brand_name") or row.get("brand") or row.get("name")
            if value:
                names.append(value)
    return names


def validate_target_brand(target_brand: str, brands_csv: Path) -> str | None:
    available = brand_names(brands_csv)
    if target_brand in available:
        return None
    return f"Target brand {target_brand!r} was not found in brands CSV. Available brands: {', '.join(available) or '(none)'}."


def extract_brand_name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("name", "brand_name", "title", "domain"):
            if value.get(key):
                return str(value[key]).strip()
    return ""


def derive_brands_csv_from_prompts(prompts_csv: Path, output_csv: Path, target_brand: str) -> Path:
    try:
        frame = pd.read_csv(prompts_csv)
    except EmptyDataError as exc:
        raise ValueError("Peec export returned no prompt rows for the selected filters.") from exc
    names: list[str] = []
    if target_brand.strip():
        names.append(target_brand.strip())
    if "brands_mentioned" in frame.columns:
        for raw in frame["brands_mentioned"].fillna(""):
            try:
                parsed = json.loads(str(raw)) if str(raw).strip() else []
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, list):
                for item in parsed:
                    name = extract_brand_name(item)
                    if name:
                        names.append(name)
    deduped = list(dict.fromkeys(name for name in names if name))
    if not deduped:
        raise ValueError("Could not derive brands from Peec data. Upload a brands CSV or provide a target brand.")
    pd.DataFrame(
        [{"brand_id": f"b{i + 1}", "brand_name": name, "aliases": ""} for i, name in enumerate(deduped)]
    ).to_csv(output_csv, index=False)
    return output_csv


async def prepare_feature_file(form, inputs_dir: Path) -> tuple[Path, list[dict[str, Any]], str]:
    feature_file = form.get("feature_file")
    features = form.get("features_csv")
    feature_pdf = form.get("feature_pdf")
    upload = feature_file or features or feature_pdf
    if not isinstance(upload, UploadFile):
        raise ValueError("Upload one feature file: CSV or PDF.")

    suffix = Path(upload.filename or "").suffix.casefold()
    extracted_features: list[dict[str, Any]] = []
    extracted_text = ""
    if suffix == ".csv" or upload.content_type in {"text/csv", "application/vnd.ms-excel"}:
        return await save_upload(upload, inputs_dir / "features.csv"), extracted_features, extracted_text
    if suffix == ".pdf" or upload.content_type == "application/pdf":
        feature_mode = str(form.get("feature_mode") or "mock")
        frame, extracted_text = extract_features_from_pdf(await upload.read(), mode=feature_mode)
        if frame.empty:
            raise ValueError("No usable features were extracted from the PDF.")
        features_path = inputs_dir / "features_from_pdf.csv"
        frame.to_csv(features_path, index=False)
        extracted_features = json.loads(frame.to_json(orient="records"))
        return features_path, extracted_features, extracted_text
    raise ValueError("Feature file must be a CSV or PDF.")


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
            feature_evidence_mode=str(body.get("feature_evidence_mode") or "openai_mock"),
            embedding_backend=str(body.get("embedding_backend") or "hash"),
            aggregation_mode=str(body.get("aggregation_mode") or "prompt"),
        )
        if result["ok"]:
            manifest = persist_run(
                source="sample",
                prompts_csv=ROOT / "data" / "sample_prompts.csv",
                features_csv=ROOT / "data" / "demo_features.csv",
                brands_csv=ROOT / "data" / "demo_brands.csv",
                output_dir=output_dir,
                target_brand=str(body.get("target_brand") or "Peec AI"),
                aggregation_mode=str(body.get("aggregation_mode") or "prompt"),
            )
            result["run_id"] = manifest["run_id"]
            result["manifest"] = manifest
        return JSONResponse(result, status_code=200 if result["ok"] else 500)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def analyze_upload(request) -> JSONResponse:
    form = await request.form()
    prompts = form.get("prompts_csv")
    brands = form.get("brands_csv")
    if not isinstance(prompts, UploadFile) or not isinstance(brands, UploadFile):
        return JSONResponse({"ok": False, "error": "prompts_csv and brands_csv are required."}, status_code=400)

    workdir = Path(tempfile.mkdtemp(prefix="feature_visibility_api_"))
    try:
        inputs_dir = workdir / "inputs"
        prompts_path = await save_upload(prompts, inputs_dir / "prompts.csv")
        brands_path = await save_upload(brands, inputs_dir / "brands.csv")
        try:
            features_path, extracted_features, extracted_text = await prepare_feature_file(form, inputs_dir)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

        target_brand = str(form.get("target_brand") or first_brand_name(brands_path))
        target_error = validate_target_brand(target_brand, brands_path)
        if target_error:
            return JSONResponse({"ok": False, "error": target_error}, status_code=400)
        result = run_pipeline(
            prompts_csv=prompts_path,
            features_csv=features_path,
            brands_csv=brands_path,
            target_brand=target_brand,
            output_dir=workdir / "outputs",
            normalizer=str(form.get("normalizer") or "openai_mock"),
            brand_detector=str(form.get("brand_detector") or "openai_mock"),
            feature_evidence_mode=str(form.get("feature_evidence_mode") or "openai_mock"),
            embedding_backend=str(form.get("embedding_backend") or "hash"),
            aggregation_mode=str(form.get("aggregation_mode") or "prompt"),
        )
        if result["ok"]:
            manifest = persist_run(
                source="csv",
                prompts_csv=prompts_path,
                features_csv=features_path,
                brands_csv=brands_path,
                output_dir=workdir / "outputs",
                target_brand=target_brand,
                aggregation_mode=str(form.get("aggregation_mode") or "prompt"),
            )
            result["run_id"] = manifest["run_id"]
            result["manifest"] = manifest
        result["extracted_features"] = extracted_features
        result["extracted_text_preview"] = extracted_text[:12000]
        return JSONResponse(result, status_code=200 if result["ok"] else 500)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def analyze_peec(request) -> JSONResponse:
    form = await request.form()
    project_id = str(form.get("project_id") or "").strip()
    start_date = str(form.get("start_date") or "").strip()
    end_date = str(form.get("end_date") or "").strip()
    target_brand = str(form.get("target_brand") or "").strip()
    if not start_date or not end_date:
        return JSONResponse({"ok": False, "error": "start_date and end_date are required."}, status_code=400)
    if not target_brand:
        return JSONResponse({"ok": False, "error": "target_brand is required for Peec MCP runs."}, status_code=400)

    workdir = Path(tempfile.mkdtemp(prefix="feature_visibility_peec_api_"))
    try:
        inputs_dir = workdir / "inputs"
        prompts_path = inputs_dir / "peec_chats.csv"
        export_result = run_peec_export(
            project_id=project_id,
            start_date=start_date,
            end_date=end_date,
            output_csv=prompts_path,
            limit=int(str(form.get("limit") or "250")),
        )
        if not export_result["ok"]:
            return JSONResponse(
                {
                    "ok": False,
                    "error": summarize_peec_export_error(export_result["stdout"], export_result["stderr"]),
                },
                status_code=500,
            )
        if not prompts_path.exists() or prompts_path.stat().st_size == 0:
            return JSONResponse(
                {"ok": False, "error": "Peec export returned no prompt rows for the selected filters."},
                status_code=400,
            )

        try:
            features_path, extracted_features, extracted_text = await prepare_feature_file(form, inputs_dir)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

        brands_upload = form.get("brands_csv")
        try:
            if isinstance(brands_upload, UploadFile):
                brands_path = await save_upload(brands_upload, inputs_dir / "brands.csv")
            else:
                brands_path = derive_brands_csv_from_prompts(prompts_path, inputs_dir / "brands_from_peec.csv", target_brand)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

        target_error = validate_target_brand(target_brand, brands_path)
        if target_error:
            return JSONResponse({"ok": False, "error": target_error}, status_code=400)

        result = run_pipeline(
            prompts_csv=prompts_path,
            features_csv=features_path,
            brands_csv=brands_path,
            target_brand=target_brand,
            output_dir=workdir / "outputs",
            normalizer=str(form.get("normalizer") or "openai_mock"),
            brand_detector=str(form.get("brand_detector") or "openai_mock"),
            feature_evidence_mode=str(form.get("feature_evidence_mode") or "openai_mock"),
            embedding_backend=str(form.get("embedding_backend") or "hash"),
            aggregation_mode=str(form.get("aggregation_mode") or "prompt"),
        )
        if result["ok"]:
            manifest = persist_run(
                source="peec",
                prompts_csv=prompts_path,
                features_csv=features_path,
                brands_csv=brands_path,
                output_dir=workdir / "outputs",
                target_brand=target_brand,
                aggregation_mode=str(form.get("aggregation_mode") or "prompt"),
                extra_meta={
                    "project_id": project_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "peec_limit": int(str(form.get("limit") or "250")),
                },
            )
            result["run_id"] = manifest["run_id"]
            result["manifest"] = manifest
        result["peec_export_stdout"] = export_result["stdout"]
        result["peec_export_stderr"] = export_result["stderr"]
        result["extracted_features"] = extracted_features
        result["extracted_text_preview"] = extracted_text[:12000]
        return JSONResponse(result, status_code=200 if result["ok"] else 500)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def analyze_upload_stream(request) -> StreamingResponse:
    form = await request.form()

    async def generate():
        yield progress_event("run", status="started", source="csv")
        prompts = form.get("prompts_csv")
        brands = form.get("brands_csv")
        if not isinstance(prompts, UploadFile) or not isinstance(brands, UploadFile):
            yield progress_event("error", error="prompts_csv and brands_csv are required.")
            return

        workdir = Path(tempfile.mkdtemp(prefix="feature_visibility_api_stream_"))
        try:
            inputs_dir = workdir / "inputs"

            started, tick = step_started("inputs", "Saving uploaded CSV files")
            yield started
            prompts_path = await save_upload(prompts, inputs_dir / "prompts.csv")
            brands_path = await save_upload(brands, inputs_dir / "brands.csv")
            yield step_finished("inputs", tick, "Saved prompt and brand inputs")

            started, tick = step_started("features", "Preparing feature descriptions")
            yield started
            try:
                features_path, extracted_features, extracted_text = await prepare_feature_file(form, inputs_dir)
            except ValueError as exc:
                yield progress_event("error", error=str(exc))
                return
            yield step_finished(
                "features",
                tick,
                "Prepared feature descriptions",
                feature_count=len(extracted_features),
            )

            started, tick = step_started("validation", "Validating target brand")
            yield started
            target_brand = str(form.get("target_brand") or first_brand_name(brands_path))
            target_error = validate_target_brand(target_brand, brands_path)
            if target_error:
                yield progress_event("error", error=target_error)
                return
            yield step_finished("validation", tick, "Validated target brand")

            yield progress_event("stage", stage="pipeline", status="running", message="Running visibility coverage pipeline")
            process = run_pipeline_stream(
                prompts_csv=prompts_path,
                features_csv=features_path,
                brands_csv=brands_path,
                target_brand=target_brand,
                output_dir=workdir / "outputs",
                normalizer=str(form.get("normalizer") or "openai_mock"),
                brand_detector=str(form.get("brand_detector") or "openai_mock"),
                feature_evidence_mode=str(form.get("feature_evidence_mode") or "openai_mock"),
                embedding_backend=str(form.get("embedding_backend") or "hash"),
                aggregation_mode=str(form.get("aggregation_mode") or "prompt"),
            )
            assert process.stdout is not None
            pipeline_stdout_lines: list[str] = []
            for raw_line in process.stdout:
                if raw_line.startswith("__FV_PROGRESS__ "):
                    payload = json.loads(raw_line[len("__FV_PROGRESS__ "):])
                    yield progress_event("stage", **payload)
                else:
                    pipeline_stdout_lines.append(raw_line)
            stderr = process.stderr.read() if process.stderr is not None else ""
            return_code = process.wait()
            cleaned_stdout = "".join(pipeline_stdout_lines).strip()
            if return_code != 0:
                yield progress_event("error", error=summarize_pipeline_error(cleaned_stdout, stderr))
                return
            metadata_path = workdir / "outputs" / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            result = {
                "ok": True,
                "stdout": (cleaned_stdout + "\n") if cleaned_stdout else "",
                "stderr": stderr,
                "overview": dataframe_records(workdir / "outputs" / "feature_gap_overview.csv"),
                "details": dataframe_records(workdir / "outputs" / "feature_gap_details.csv"),
                "coverage": dataframe_records(workdir / "outputs" / "coverage_by_feature_cluster.csv"),
                "summary": read_text(workdir / "outputs" / "feature_gap_summary.md"),
                "metadata": metadata,
            }
            manifest = persist_run(
                source="csv",
                prompts_csv=prompts_path,
                features_csv=features_path,
                brands_csv=brands_path,
                output_dir=workdir / "outputs",
                target_brand=target_brand,
                aggregation_mode=str(form.get("aggregation_mode") or "prompt"),
            )
            result["run_id"] = manifest["run_id"]
            result["manifest"] = manifest
            result["extracted_features"] = extracted_features
            result["extracted_text_preview"] = extracted_text[:12000]
            yield progress_event("result", result=result)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


async def analyze_peec_stream(request) -> StreamingResponse:
    form = await request.form()

    async def generate():
        yield progress_event("run", status="started", source="peec")
        project_id = str(form.get("project_id") or "").strip()
        start_date = str(form.get("start_date") or "").strip()
        end_date = str(form.get("end_date") or "").strip()
        target_brand = str(form.get("target_brand") or "").strip()
        if not start_date or not end_date:
            yield progress_event("error", error="start_date and end_date are required.")
            return
        if not target_brand:
            yield progress_event("error", error="target_brand is required for Peec MCP runs.")
            return

        workdir = Path(tempfile.mkdtemp(prefix="feature_visibility_peec_stream_"))
        try:
            inputs_dir = workdir / "inputs"
            prompts_path = inputs_dir / "peec_chats.csv"

            started, tick = step_started("peec_export", "Exporting prompts and responses from Peec MCP")
            yield started
            export_result = run_peec_export(
                project_id=project_id,
                start_date=start_date,
                end_date=end_date,
                output_csv=prompts_path,
                limit=int(str(form.get("limit") or "250")),
            )
            if not export_result["ok"]:
                yield progress_event(
                    "error",
                    error=summarize_peec_export_error(export_result["stdout"], export_result["stderr"]),
                )
                return
            if not prompts_path.exists() or prompts_path.stat().st_size == 0:
                yield progress_event("error", error="Peec export returned no prompt rows for the selected filters.")
                return
            prompt_rows = 0
            try:
                prompt_rows = int(len(pd.read_csv(prompts_path)))
            except EmptyDataError:
                yield progress_event("error", error="Peec export returned no prompt rows for the selected filters.")
                return
            yield step_finished("peec_export", tick, "Peec export completed", prompt_rows=prompt_rows)

            started, tick = step_started("features", "Preparing feature descriptions")
            yield started
            try:
                features_path, extracted_features, extracted_text = await prepare_feature_file(form, inputs_dir)
            except ValueError as exc:
                yield progress_event("error", error=str(exc))
                return
            yield step_finished(
                "features",
                tick,
                "Prepared feature descriptions",
                feature_count=len(extracted_features),
            )

            started, tick = step_started("brands", "Resolving tracked brands")
            yield started
            brands_upload = form.get("brands_csv")
            try:
                if isinstance(brands_upload, UploadFile):
                    brands_path = await save_upload(brands_upload, inputs_dir / "brands.csv")
                    brand_source = "upload"
                else:
                    brands_path = derive_brands_csv_from_prompts(prompts_path, inputs_dir / "brands_from_peec.csv", target_brand)
                    brand_source = "peec_mentions"
            except ValueError as exc:
                yield progress_event("error", error=str(exc))
                return
            yield step_finished("brands", tick, "Tracked brands resolved", brand_source=brand_source)

            started, tick = step_started("validation", "Validating target brand")
            yield started
            target_error = validate_target_brand(target_brand, brands_path)
            if target_error:
                yield progress_event("error", error=target_error)
                return
            yield step_finished("validation", tick, "Validated target brand")

            yield progress_event("stage", stage="pipeline", status="running", message="Running visibility coverage pipeline")
            process = run_pipeline_stream(
                prompts_csv=prompts_path,
                features_csv=features_path,
                brands_csv=brands_path,
                target_brand=target_brand,
                output_dir=workdir / "outputs",
                normalizer=str(form.get("normalizer") or "openai_mock"),
                brand_detector=str(form.get("brand_detector") or "openai_mock"),
                feature_evidence_mode=str(form.get("feature_evidence_mode") or "openai_mock"),
                embedding_backend=str(form.get("embedding_backend") or "hash"),
                aggregation_mode=str(form.get("aggregation_mode") or "prompt"),
            )
            assert process.stdout is not None
            pipeline_stdout_lines: list[str] = []
            for raw_line in process.stdout:
                if raw_line.startswith("__FV_PROGRESS__ "):
                    payload = json.loads(raw_line[len("__FV_PROGRESS__ "):])
                    yield progress_event("stage", **payload)
                else:
                    pipeline_stdout_lines.append(raw_line)
            stderr = process.stderr.read() if process.stderr is not None else ""
            return_code = process.wait()
            cleaned_stdout = "".join(pipeline_stdout_lines).strip()
            if return_code != 0:
                yield progress_event("error", error=summarize_pipeline_error(cleaned_stdout, stderr))
                return
            metadata_path = workdir / "outputs" / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            result = {
                "ok": True,
                "stdout": (cleaned_stdout + "\n") if cleaned_stdout else "",
                "stderr": stderr,
                "overview": dataframe_records(workdir / "outputs" / "feature_gap_overview.csv"),
                "details": dataframe_records(workdir / "outputs" / "feature_gap_details.csv"),
                "coverage": dataframe_records(workdir / "outputs" / "coverage_by_feature_cluster.csv"),
                "summary": read_text(workdir / "outputs" / "feature_gap_summary.md"),
                "metadata": metadata,
            }
            manifest = persist_run(
                source="peec",
                prompts_csv=prompts_path,
                features_csv=features_path,
                brands_csv=brands_path,
                output_dir=workdir / "outputs",
                target_brand=target_brand,
                aggregation_mode=str(form.get("aggregation_mode") or "prompt"),
                extra_meta={
                    "project_id": project_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "peec_limit": int(str(form.get("limit") or "250")),
                },
            )
            result["run_id"] = manifest["run_id"]
            result["manifest"] = manifest
            result["peec_export_stdout"] = export_result["stdout"]
            result["peec_export_stderr"] = export_result["stderr"]
            result["extracted_features"] = extracted_features
            result["extracted_text_preview"] = extracted_text[:12000]
            yield progress_event("result", result=result)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


async def runs_index(_request) -> JSONResponse:
    return JSONResponse({"ok": True, "runs": list_runs()})


async def run_detail(request) -> JSONResponse:
    run_id_value = str(request.path_params["run_id"])
    try:
        result = load_run_result(run_id_value)
    except FileNotFoundError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
    return JSONResponse(result)


async def run_mappings(request) -> JSONResponse:
    run_id_value = str(request.path_params["run_id"])
    feature_id = str(request.query_params.get("feature_id") or "")
    cluster_id = str(request.query_params.get("cluster_id") or "")
    limit = int(str(request.query_params.get("limit") or "200"))
    try:
        mappings = load_run_mappings(run_id_value, feature_id=feature_id, cluster_id=cluster_id, limit=limit)
    except FileNotFoundError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
    return JSONResponse({"ok": True, "run_id": run_id_value, "mappings": mappings})


async def run_reaggregate(request) -> JSONResponse:
    run_id_value = str(request.path_params["run_id"])
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    aggregation_mode = str(body.get("aggregation_mode") or "prompt")
    try:
        manifest = reaggregate_saved_run(run_id_value, aggregation_mode)
        result = load_run_result(str(manifest["run_id"]))
    except FileNotFoundError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
    result["run_id"] = manifest["run_id"]
    return JSONResponse(result)


routes = [
    Route("/api/health", health, methods=["GET"]),
    Route("/api/analyze-sample", analyze_sample, methods=["POST"]),
    Route("/api/analyze", analyze_upload, methods=["POST"]),
    Route("/api/analyze-peec", analyze_peec, methods=["POST"]),
    Route("/api/analyze-stream", analyze_upload_stream, methods=["POST"]),
    Route("/api/analyze-peec-stream", analyze_peec_stream, methods=["POST"]),
    Route("/api/runs", runs_index, methods=["GET"]),
    Route("/api/runs/{run_id}", run_detail, methods=["GET"]),
    Route("/api/runs/{run_id}/mappings", run_mappings, methods=["GET"]),
    Route("/api/runs/{run_id}/reaggregate", run_reaggregate, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
