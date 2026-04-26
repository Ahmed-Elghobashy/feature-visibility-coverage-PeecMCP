#!/usr/bin/env python3
"""
Feature visibility coverage MVP.

Inputs:
  - prompts CSV with a prompt column and optional response/answer column.
  - features CSV with feature name and feature description columns.
  - target brand keyword or brands CSV.

Outputs:
  - row-level query mapping
  - brand x feature x cluster coverage
  - cluster summary
  - feature visibility gap overview and detail outputs

The default embedding backend is BAAI/bge-m3 through sentence-transformers.
When sentence-transformers is unavailable, use --embedding-backend hash for a
deterministic local smoke-test backend.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_NORMALIZER_MODEL = "gpt-4.1-mini"
PROGRESS_PREFIX = "__FV_PROGRESS__ "
CACHE_ROOT = Path(".cache/feature_visibility")

PROMPT_COLUMNS = ("prompt", "raw_prompt", "query", "question", "text")
RESPONSE_COLUMNS = ("response", "answer", "ai_response", "model_response", "output")
FEATURE_NAME_COLUMNS = ("feature", "feature_name", "name", "title")
FEATURE_DESCRIPTION_COLUMNS = ("description", "feature_description", "desc")
BRAND_NAME_COLUMNS = ("brand", "brand_name", "name")
BRAND_ALIAS_COLUMNS = ("aliases", "brand_aliases", "alias")


@dataclass(frozen=True)
class Config:
    prompts_csv: Path
    features_csv: Path
    brand: str | None
    brands_csv: Path | None
    target_brand: str | None
    target_brand_id: str | None
    output_dir: Path
    embedding_backend: str
    embedding_model: str
    normalizer: str
    normalizer_model: str
    brand_detector: str
    brand_detector_model: str
    cluster_threshold: float
    feature_threshold: float
    min_cluster_size: int
    min_coverage_n: int
    aggregation_mode: str


_JSON_CACHE: dict[str, dict[str, object]] = {}


def emit_progress(stage: str, status: str, message: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "status": status,
        "message": message,
        "timestamp_ms": int(time.time() * 1000),
        **extra,
    }
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def cache_file(name: str) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / name


def load_json_cache(name: str) -> dict[str, object]:
    if name in _JSON_CACHE:
        return _JSON_CACHE[name]
    path = cache_file(name)
    if path.exists():
        try:
            _JSON_CACHE[name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            _JSON_CACHE[name] = {}
    else:
        _JSON_CACHE[name] = {}
    return _JSON_CACHE[name]


def cache_lookup(name: str, key_payload: object) -> object | None:
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return load_json_cache(name).get(key)


def cache_store(name: str, key_payload: object, value: object) -> None:
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    cache = load_json_cache(name)
    cache[key] = value
    cache_file(name).write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(Path(".env"))

    load_started = time.perf_counter()
    emit_progress("load_inputs", "running", "Loading prompts, features, and brands")
    prompts = pd.read_csv(config.prompts_csv)
    features = pd.read_csv(config.features_csv)
    brands = load_brands(config)
    emit_progress(
        "load_inputs",
        "completed",
        "Loaded prompts, features, and brands",
        duration_ms=int((time.perf_counter() - load_started) * 1000),
        prompt_rows=len(prompts),
        feature_rows=len(features),
        brand_rows=len(brands),
    )

    prompt_col = pick_column(prompts, PROMPT_COLUMNS, "prompts")
    response_col = pick_optional_column(prompts, RESPONSE_COLUMNS)
    feature_name_col = pick_column(features, FEATURE_NAME_COLUMNS, "features")
    feature_desc_col = pick_column(features, FEATURE_DESCRIPTION_COLUMNS, "features")

    prompts = prompts.copy()
    features = features.copy()
    prompts["prompt_id"] = ensure_id(prompts, "prompt_id", "prompt")
    features["feature_id"] = ensure_id(features, "feature_id", "feature")

    normalizer_version = (
        f"openai:{config.normalizer_model}"
        if config.normalizer == "openai"
        else f"openai-mock:{config.normalizer_model}"
        if config.normalizer == "openai_mock"
        else "heuristic:v1"
    )
    normalize_started = time.perf_counter()
    emit_progress("normalize_queries", "running", "Normalizing prompts into canonical queries")
    prompts["canonical_query"] = [
        normalize_prompt(str(prompt), config, brands)
        for prompt in prompts[prompt_col].fillna("")
    ]
    emit_progress(
        "normalize_queries",
        "completed",
        "Normalized prompts into canonical queries",
        duration_ms=int((time.perf_counter() - normalize_started) * 1000),
        prompt_rows=len(prompts),
        normalizer=config.normalizer,
    )

    embed_started = time.perf_counter()
    emit_progress("embed_text", "running", "Embedding queries and features")
    embedder = make_embedder(config)
    query_vectors = embedder.encode(prompts["canonical_query"].tolist())
    feature_texts = [
        f"{row[feature_name_col]}. {row[feature_desc_col]}"
        for _, row in features.iterrows()
    ]
    feature_vectors = embedder.encode(feature_texts)
    emit_progress(
        "embed_text",
        "completed",
        "Embedded queries and features",
        duration_ms=int((time.perf_counter() - embed_started) * 1000),
        embedding_backend=config.embedding_backend,
        query_count=len(prompts),
        feature_count=len(features),
    )

    cluster_started = time.perf_counter()
    emit_progress("cluster_queries", "running", "Clustering normalized queries")
    clusters = cluster_vectors(
        query_vectors,
        threshold=config.cluster_threshold,
        min_cluster_size=config.min_cluster_size,
    )
    prompts["cluster_id"] = clusters

    cluster_labels = label_clusters(prompts, query_vectors)
    prompts["cluster_label"] = prompts["cluster_id"].map(cluster_labels)
    emit_progress(
        "cluster_queries",
        "completed",
        "Clustered normalized queries",
        duration_ms=int((time.perf_counter() - cluster_started) * 1000),
        cluster_count=int(prompts["cluster_id"].nunique()),
    )

    feature_map_started = time.perf_counter()
    emit_progress("map_features", "running", "Mapping queries to product features")
    feature_mapping = map_features(
        query_vectors=query_vectors,
        feature_vectors=feature_vectors,
        features=features,
        feature_name_col=feature_name_col,
        threshold=config.feature_threshold,
    )
    emit_progress(
        "map_features",
        "completed",
        "Mapped queries to product features",
        duration_ms=int((time.perf_counter() - feature_map_started) * 1000),
        matched_rows=int(sum(1 for item in feature_mapping if item["feature_id"])),
    )

    text_for_brand_detection = (
        prompts[response_col].fillna("").astype(str)
        if response_col
        else prompts[prompt_col].fillna("").astype(str)
    )
    rows = prompts.copy()
    rows["original_prompt"] = prompts[prompt_col]
    rows["brand_detection_source"] = response_col or prompt_col
    rows["normalizer_version"] = normalizer_version
    rows["embedder_version"] = embedder.version
    rows["clusterer_version"] = (
        f"connected-components-cosine-threshold:{config.cluster_threshold}:"
        f"min-size:{config.min_cluster_size}"
    )
    rows["detector_version"] = (
        f"openai:{config.brand_detector_model}"
        if config.brand_detector == "openai"
        else f"openai-mock:{config.brand_detector_model}"
        if config.brand_detector == "openai_mock"
        else "casefold-word-boundary:v1"
    )
    rows["mapped_feature_id"] = [m["feature_id"] for m in feature_mapping]
    rows["mapped_feature_name"] = [m["feature_name"] for m in feature_mapping]
    rows["feature_similarity"] = [m["similarity"] for m in feature_mapping]
    rows["feature_match_status"] = [
        "matched" if m["feature_id"] else "unmatched"
        for m in feature_mapping
    ]

    brand_started = time.perf_counter()
    emit_progress("detect_brands", "running", "Detecting brand presence in responses")
    brand_rows = expand_brand_rows(rows, brands, text_for_brand_detection, config)
    emit_progress(
        "detect_brands",
        "completed",
        "Detected brand presence in responses",
        duration_ms=int((time.perf_counter() - brand_started) * 1000),
        expanded_rows=len(brand_rows),
        brand_detector=config.brand_detector,
    )

    coverage_started = time.perf_counter()
    emit_progress("aggregate_coverage", "running", "Aggregating coverage by feature and cluster")
    coverage = build_coverage(brand_rows, config.min_coverage_n, config.aggregation_mode)
    cluster_summary = build_cluster_summary(rows)
    emit_progress(
        "aggregate_coverage",
        "completed",
        "Aggregated coverage by feature and cluster",
        duration_ms=int((time.perf_counter() - coverage_started) * 1000),
        coverage_rows=len(coverage),
        cluster_rows=len(cluster_summary),
    )

    gap_started = time.perf_counter()
    emit_progress("compute_gaps", "running", "Computing feature visibility gaps")
    target_brand = resolve_target_brand(brands, config)
    feature_gap_overview = build_feature_gap_overview(coverage, rows, target_brand)
    feature_gap_details = build_feature_gap_details(coverage, rows, target_brand)
    pm_summary = build_pm_summary(feature_gap_overview, target_brand)
    emit_progress(
        "compute_gaps",
        "completed",
        "Computed feature visibility gaps",
        duration_ms=int((time.perf_counter() - gap_started) * 1000),
        gap_rows=len(feature_gap_overview),
    )

    write_started = time.perf_counter()
    emit_progress("write_outputs", "running", "Writing CSV and summary outputs")
    write_outputs(
        config.output_dir,
        brand_rows,
        coverage,
        cluster_summary,
        feature_gap_overview,
        feature_gap_details,
        pm_summary,
        config,
        brands,
        target_brand,
    )
    emit_progress(
        "write_outputs",
        "completed",
        "Wrote CSV and summary outputs",
        duration_ms=int((time.perf_counter() - write_started) * 1000),
        output_dir=str(config.output_dir),
    )
    print_summary(config.output_dir, brand_rows, coverage, feature_gap_overview, target_brand, config, brands)
    return 0


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Compute feature visibility coverage from prompt and feature CSVs."
    )
    parser.add_argument("--prompts", required=True, type=Path, help="Prompt CSV path.")
    parser.add_argument("--features", required=True, type=Path, help="Feature CSV path.")
    parser.add_argument("--brand", help="Single target brand keyword/name.")
    parser.add_argument(
        "--brands",
        type=Path,
        help="CSV of target brands. Expected columns include brand_id and brand_name/name/brand.",
    )
    parser.add_argument(
        "--target-brand",
        help="Target brand name for feature visibility gap reporting.",
    )
    parser.add_argument(
        "--target-brand-id",
        help="Target brand id for feature visibility gap reporting. Overrides --target-brand.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated CSV/JSON outputs.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("bge-m3", "hash"),
        default="bge-m3",
        help="Use bge-m3 for production; hash is only for local smoke tests.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "--normalizer",
        choices=("heuristic", "openai", "openai_mock"),
        default="heuristic",
        help="Prompt compression backend.",
    )
    parser.add_argument(
        "--normalizer-model",
        default=DEFAULT_NORMALIZER_MODEL,
        help="OpenAI model for --normalizer openai.",
    )
    parser.add_argument(
        "--brand-detector",
        choices=("keyword", "openai", "openai_mock"),
        default="keyword",
        help="Brand presence detector: keyword is exact/auditable; openai is LLM-judge based.",
    )
    parser.add_argument(
        "--brand-detector-model",
        default=DEFAULT_NORMALIZER_MODEL,
        help="OpenAI model for --brand-detector openai.",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.60,
        help="Cosine similarity threshold for clustering canonical queries.",
    )
    parser.add_argument(
        "--feature-threshold",
        type=float,
        default=0.45,
        help="Minimum cosine similarity required to map a query to a feature.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Clusters smaller than this are marked as outliers.",
    )
    parser.add_argument(
        "--min-coverage-n",
        type=int,
        default=1,
        help="Minimum rows required before showing a numeric coverage rate.",
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=("response", "prompt", "prompt_model"),
        default="response",
        help="Counting unit for coverage: response rows, unique prompts, or unique prompt/model pairs.",
    )
    args = parser.parse_args()
    if not args.brand and not args.brands:
        parser.error("Provide --brand for one brand or --brands for a brands CSV.")

    return Config(
        prompts_csv=args.prompts,
        features_csv=args.features,
        brand=args.brand,
        brands_csv=args.brands,
        target_brand=args.target_brand,
        target_brand_id=args.target_brand_id,
        output_dir=args.output_dir,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        normalizer=args.normalizer,
        normalizer_model=args.normalizer_model,
        brand_detector=args.brand_detector,
        brand_detector_model=args.brand_detector_model,
        cluster_threshold=args.cluster_threshold,
        feature_threshold=args.feature_threshold,
        min_cluster_size=args.min_cluster_size,
        min_coverage_n=args.min_coverage_n,
        aggregation_mode=args.aggregation_mode,
    )


def pick_column(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    by_lower = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    raise ValueError(
        f"Could not find a {label} column. Expected one of: {', '.join(candidates)}. "
        f"Found: {', '.join(frame.columns)}"
    )


def pick_optional_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    by_lower = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    return None


def load_brands(config: Config) -> pd.DataFrame:
    if config.brands_csv:
        brands = pd.read_csv(config.brands_csv).copy()
        brand_name_col = pick_column(brands, BRAND_NAME_COLUMNS, "brands")
        alias_col = pick_optional_column(brands, BRAND_ALIAS_COLUMNS)
        brands["brand_name"] = brands[brand_name_col].astype(str)
        brands["brand_id"] = ensure_id(brands, "brand_id", "brand")
        brands["brand_aliases"] = brands[alias_col].fillna("").astype(str) if alias_col else ""
    else:
        assert config.brand
        brands = pd.DataFrame(
            [{"brand_id": "brand_001", "brand_name": config.brand, "brand_aliases": ""}]
        )

    brands = brands[brands["brand_name"].astype(str).str.strip() != ""].copy()
    if brands.empty:
        raise ValueError("No non-empty brands were provided.")
    return brands[["brand_id", "brand_name", "brand_aliases"]].drop_duplicates().reset_index(drop=True)


def ensure_id(frame: pd.DataFrame, column: str, prefix: str) -> list[str]:
    if column in frame.columns:
        return [str(value) for value in frame[column]]
    width = max(3, int(math.log10(max(len(frame), 1))) + 1)
    return [f"{prefix}_{i:0{width}d}" for i in range(1, len(frame) + 1)]


def normalize_prompt(prompt: str, config: Config, brands: pd.DataFrame) -> str:
    if config.normalizer == "openai":
        return normalize_prompt_openai(prompt, config.normalizer_model, brands)
    if config.normalizer == "openai_mock":
        return normalize_prompt_openai_mock(prompt, config.normalizer_model, brands)
    return normalize_prompt_heuristic(prompt, brands["brand_name"].tolist())


def normalize_prompt_heuristic(prompt: str, brand_names: Sequence[str]) -> str:
    text = prompt.casefold()
    for brand in brand_names:
        text = re.sub(re.escape(str(brand).casefold()), " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ÖØ-öø-ÿ\u0600-\u06FF\u0400-\u04FF\s-]", " ", text)
    text = re.sub(r"\b(best|top|leading|modern|new|please|recommend|compare|tools?)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    return " ".join(words[:12]) if words else "empty query"


def normalize_prompt_openai(prompt: str, model: str, brands: pd.DataFrame) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("--normalizer openai requires OPENAI_API_KEY.")

    cache_key = {
        "kind": "normalizer",
        "model": model,
        "prompt": prompt,
        "brands": brands["brand_name"].astype(str).tolist(),
    }
    cached = cache_lookup("openai_normalizer.json", cache_key)
    if isinstance(cached, str) and cached.strip():
        return cached

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Compress user prompts into short canonical search queries. "
                    "Preserve the underlying need, remove brand names, remove filler, "
                    "and return only the query in lowercase. Maximum 8 words. "
                    "Remove these tracked brand names if they appear: "
                    + ", ".join(brands["brand_name"].astype(str).tolist())
                    + "."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
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
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI normalizer call failed: {exc}") from exc

    content = body["choices"][0]["message"]["content"]
    normalized = re.sub(r"\s+", " ", content.strip().casefold())
    cache_store("openai_normalizer.json", cache_key, normalized)
    return normalized


def normalize_prompt_openai_mock(prompt: str, model: str, brands: pd.DataFrame) -> str:
    del model
    text = normalize_prompt_heuristic(prompt, brands["brand_name"].tolist())
    return " ".join(text.split()[:8]) if text else "empty query"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class Embedder:
    version: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class BgeM3Embedder(Embedder):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "BAAI/bge-m3 requires sentence-transformers and torch. "
                "Install dependencies from requirements.txt, or run with "
                "--embedding-backend hash for a smoke test."
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.version = f"sentence-transformers:{model_name}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class HashEmbedder(Embedder):
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.version = f"hash-embedding:v1:{dimensions}:smoke-test-only"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, bucket] += sign
        return l2_normalize(vectors)


def make_embedder(config: Config) -> Embedder:
    if config.embedding_backend == "hash":
        return HashEmbedder()
    return BgeM3Embedder(config.embedding_model)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ÖØ-öø-ÿ\u0600-\u06FF\u0400-\u04FF-]+", text.casefold())


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(vectors.astype(np.float32))
    return normalized @ normalized.T


def cluster_vectors(vectors: np.ndarray, threshold: float, min_cluster_size: int) -> list[str]:
    if len(vectors) == 0:
        return []

    similarities = cosine_similarity_matrix(vectors)
    parent = list(range(len(vectors)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if similarities[i, j] >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(vectors)):
        groups.setdefault(find(i), []).append(i)

    cluster_ids = ["outlier" for _ in range(len(vectors))]
    cluster_number = 1
    for members in sorted(groups.values(), key=lambda group: (-len(group), group[0])):
        if len(members) < min_cluster_size:
            continue
        cluster_id = f"cluster_{cluster_number:03d}"
        cluster_number += 1
        for member in members:
            cluster_ids[member] = cluster_id
    return cluster_ids


def label_clusters(rows: pd.DataFrame, query_vectors: np.ndarray) -> dict[str, str]:
    labels: dict[str, str] = {"outlier": "Outlier / unique query"}
    for cluster_id, group in rows.groupby("cluster_id", sort=True):
        if cluster_id == "outlier":
            continue
        indices = group.index.to_list()
        centroid = query_vectors[indices].mean(axis=0, keepdims=True)
        centroid = l2_normalize(centroid)[0]
        sims = query_vectors[indices] @ centroid
        medoid_index = indices[int(np.argmax(sims))]
        labels[cluster_id] = str(rows.loc[medoid_index, "canonical_query"])
    return labels


def map_features(
    query_vectors: np.ndarray,
    feature_vectors: np.ndarray,
    features: pd.DataFrame,
    feature_name_col: str,
    threshold: float,
) -> list[dict[str, object]]:
    query_vectors = l2_normalize(query_vectors.astype(np.float32))
    feature_vectors = l2_normalize(feature_vectors.astype(np.float32))
    similarities = query_vectors @ feature_vectors.T
    mappings: list[dict[str, object]] = []

    for row in range(similarities.shape[0]):
        best_index = int(np.argmax(similarities[row]))
        best_score = float(similarities[row, best_index])
        feature = features.iloc[best_index]
        if best_score < threshold:
            mappings.append(
                {
                    "feature_id": "",
                    "feature_name": "",
                    "similarity": round(best_score, 4),
                }
            )
        else:
            mappings.append(
                {
                    "feature_id": feature["feature_id"],
                    "feature_name": feature[feature_name_col],
                    "similarity": round(best_score, 4),
                }
            )
    return mappings


def split_aliases(value: object) -> list[str]:
    if value is None:
        return []
    raw = str(value).strip()
    if not raw or raw.casefold() == "nan":
        return []
    return [part.strip() for part in re.split(r"[|,;]", raw) if part.strip()]


def brand_terms(brand: pd.Series) -> list[str]:
    terms = [str(brand["brand_name"])]
    terms.extend(split_aliases(brand.get("brand_aliases", "")))
    return list(dict.fromkeys(term for term in terms if term.strip()))


def contains_brand(text: str, brand: str | Sequence[str]) -> bool:
    terms = [brand] if isinstance(brand, str) else list(brand)
    return any(contains_brand_term(text, str(term)) for term in terms)


def contains_brand_term(text: str, brand: str) -> bool:
    if not brand.strip():
        return False
    escaped = re.escape(brand.strip())
    pattern = rf"(?<![\w-]){escaped}(?![\w-])"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def judge_brand_openai(text: str, brand: str, model: str, aliases: Sequence[str] | None = None) -> bool:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("--brand-detector openai requires OPENAI_API_KEY.")

    cache_key = {
        "kind": "brand_detector",
        "model": model,
        "brand": brand,
        "aliases": list(aliases or []),
        "response": text,
    }
    cached = cache_lookup("openai_brand_detector.json", cache_key)
    if isinstance(cached, bool):
        return cached

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict brand-mention judge. Decide whether the AI "
                    "response explicitly mentions the target brand, product, or an "
                    "obvious spelling/spacing variant of it. Do not count generic "
                    "descriptions or semantic hints without a name. Return only JSON "
                    "with this shape: {\"brand_present\": true} or {\"brand_present\": false}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "brand": brand,
                        "aliases": list(aliases or []),
                        "response": text,
                    },
                    ensure_ascii=False,
                ),
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
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI brand detector call failed: {exc}") from exc

    content = body["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI brand detector returned invalid JSON: {content}") from exc
    verdict = bool(parsed.get("brand_present"))
    cache_store("openai_brand_detector.json", cache_key, verdict)
    return verdict


def detect_brand(text: str, brand: str, aliases: Sequence[str], config: Config) -> bool:
    if config.brand_detector == "openai":
        return judge_brand_openai(text, brand, config.brand_detector_model, aliases)
    if config.brand_detector == "openai_mock":
        return judge_brand_openai_mock(text, [brand, *aliases], config.brand_detector_model)
    return contains_brand(text, [brand, *aliases])


def judge_brand_openai_mock(text: str, brand: str | Sequence[str], model: str) -> bool:
    del model
    if contains_brand(text, brand):
        return True
    collapsed_text = re.sub(r"[\s_-]+", "", text.casefold())
    terms = [brand] if isinstance(brand, str) else list(brand)
    for term in terms:
        collapsed_brand = re.sub(r"[\s_-]+", "", str(term).casefold())
        if collapsed_brand and collapsed_brand in collapsed_text:
            return True
    return False


def expand_brand_rows(
    rows: pd.DataFrame,
    brands: pd.DataFrame,
    detection_text: pd.Series,
    config: Config,
) -> pd.DataFrame:
    expanded: list[pd.DataFrame] = []
    detection_values = detection_text.reset_index(drop=True)
    base = rows.reset_index(drop=True)
    for _, brand in brands.iterrows():
        brand_rows = base.copy()
        brand_name = str(brand["brand_name"])
        aliases = split_aliases(brand.get("brand_aliases", ""))
        brand_rows["brand_id"] = str(brand["brand_id"])
        brand_rows["brand_name"] = brand_name
        brand_rows["brand_aliases"] = "|".join(aliases)
        brand_rows["brand_present"] = [
            detect_brand(text, brand_name, aliases, config)
            for text in detection_values.astype(str)
        ]
        expanded.append(brand_rows)
    return pd.concat(expanded, ignore_index=True)


def coverage_unit_columns(rows: pd.DataFrame, aggregation_mode: str) -> list[str]:
    if aggregation_mode == "prompt":
        return ["prompt_id"]
    if aggregation_mode == "prompt_model":
        for candidate in ("model", "model_name", "model_id", "engine"):
            if candidate in rows.columns:
                return ["prompt_id", candidate]
        return ["prompt_id"]
    return []


def aggregate_coverage_group(group: pd.DataFrame, aggregation_mode: str) -> pd.DataFrame:
    unit_cols = coverage_unit_columns(group, aggregation_mode)
    if not unit_cols:
        return group.copy()

    records: list[dict[str, object]] = []
    for _, unit in group.groupby(unit_cols, dropna=False, sort=False):
        first = unit.iloc[0].to_dict()
        first["brand_present"] = bool(unit["brand_present"].any())
        first["prompt_id"] = join_ids(unit["prompt_id"])
        records.append(first)
    return pd.DataFrame.from_records(records)


def build_coverage(rows: pd.DataFrame, min_coverage_n: int, aggregation_mode: str = "response") -> pd.DataFrame:
    matched = rows[rows["mapped_feature_id"].astype(str) != ""].copy()
    if matched.empty:
        return pd.DataFrame(
            columns=[
                "brand_id",
                "brand_name",
                "brand_aliases",
                "mapped_feature_id",
                "mapped_feature_name",
                "cluster_id",
                "cluster_label",
                "aggregation_mode",
                "prompt_count",
                "brand_present_count",
                "brand_absent_count",
                "coverage_rate",
                "coverage_status",
                "present_prompt_ids",
                "missing_prompt_ids",
            ]
        )

    records: list[dict[str, object]] = []
    group_cols = [
        "brand_id",
        "brand_name",
        "brand_aliases",
        "mapped_feature_id",
        "mapped_feature_name",
        "cluster_id",
        "cluster_label",
    ]
    for keys, group in matched.groupby(group_cols, dropna=False, sort=True):
        unit_group = aggregate_coverage_group(group, aggregation_mode)
        prompt_count = len(unit_group)
        present = int(unit_group["brand_present"].sum())
        missing = prompt_count - present
        coverage_rate = present / prompt_count if prompt_count >= min_coverage_n else ""
        status = (
            "insufficient_data"
            if prompt_count < min_coverage_n
            else "covered"
            if present == prompt_count
            else "missing"
            if present == 0
            else "partial"
        )
        records.append(
            {
                "brand_id": keys[0],
                "brand_name": keys[1],
                "brand_aliases": keys[2],
                "mapped_feature_id": keys[3],
                "mapped_feature_name": keys[4],
                "cluster_id": keys[5],
                "cluster_label": keys[6],
                "aggregation_mode": aggregation_mode,
                "prompt_count": prompt_count,
                "brand_present_count": present,
                "brand_absent_count": missing,
                "coverage_rate": coverage_rate,
                "coverage_status": status,
                "present_prompt_ids": join_ids(unit_group[unit_group["brand_present"]]["prompt_id"]),
                "missing_prompt_ids": join_ids(unit_group[~unit_group["brand_present"]]["prompt_id"]),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["brand_name", "mapped_feature_name", "coverage_status", "coverage_rate"],
        ascending=[True, True, True, True],
    )


def build_cluster_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for cluster_id, group in rows.groupby("cluster_id", sort=True):
        records.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": group["cluster_label"].iloc[0],
                "prompt_count": len(group),
                "example_prompt_ids": join_ids(group["prompt_id"].head(5)),
                "example_canonical_queries": " | ".join(
                    sorted(set(group["canonical_query"].head(5)))
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["cluster_id"],
        ascending=True,
    )


def resolve_target_brand(brands: pd.DataFrame, config: Config) -> pd.Series:
    if config.target_brand_id:
        matched = brands[brands["brand_id"].astype(str) == str(config.target_brand_id)]
        if matched.empty:
            raise ValueError(f"Target brand id {config.target_brand_id!r} was not found in brands CSV.")
        return matched.iloc[0]
    if config.target_brand:
        matched = brands[brands["brand_name"].astype(str).str.casefold() == str(config.target_brand).casefold()]
        if matched.empty:
            raise ValueError(f"Target brand {config.target_brand!r} was not found in brands CSV.")
        return matched.iloc[0]
    return brands.iloc[0]


def consistency_band(visibility_share: float) -> str:
    if visibility_share >= 0.70:
        return "high"
    if visibility_share >= 0.40:
        return "medium"
    return "low"


def visibility_status(visibility_share: float) -> str:
    if visibility_share == 0:
        return "missing"
    if visibility_share < 0.70:
        return "inconsistent"
    return "strong"


def is_feature_visibility_gap(visibility_share: float, competitor_present: bool) -> bool:
    return competitor_present and visibility_share < 0.70


def gap_category(visibility_share: float, competitor_present: bool) -> str:
    if is_feature_visibility_gap(visibility_share, competitor_present):
        return "competitive_gap"
    if visibility_share < 0.70:
        return "weak_category_visibility"
    return "strong_presence"


def gap_severity(visibility_share: float, competitor_present: bool) -> str:
    if competitor_present and visibility_share < 0.40:
        return "high"
    if competitor_present and visibility_share < 0.70:
        return "medium"
    return "low"


def gap_reason(visibility_share: float, competitor_present: bool) -> str:
    status = visibility_status(visibility_share)
    if competitor_present and status == "missing":
        return "Feature intent is present, competitors appear, and the target brand is missing."
    if competitor_present and status == "inconsistent":
        return "Feature intent is present, competitors appear, and the target brand is inconsistently present."
    if status in {"missing", "inconsistent"}:
        return "Feature intent is present and the target brand is weak, but tracked competitors are not present."
    return "The target brand has strong presence for this feature cluster."


def severity_signal(severity: str, category: str = "") -> str:
    if category == "weak_category_visibility":
        return "Weak category visibility"
    if severity == "high":
        return "Consistently missing — opportunity"
    if severity == "medium":
        return "Partial — worth investigating"
    return "Strong presence"


def dedupe_join(values: Iterable[object], limit: int | None = None) -> str:
    seen: list[str] = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.append(value)
    if limit is not None:
        seen = seen[:limit]
    return ";".join(seen)


def build_feature_gap_overview(
    coverage: pd.DataFrame,
    rows: pd.DataFrame,
    target_brand: pd.Series,
) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()

    target_brand_id = str(target_brand["brand_id"])
    target_brand_name = str(target_brand["brand_name"])
    target_coverage = coverage[coverage["brand_id"].astype(str) == target_brand_id].copy()
    if target_coverage.empty:
        return pd.DataFrame()

    all_example_queries = (
        rows.groupby(["mapped_feature_id", "cluster_id"], dropna=False)["canonical_query"]
        .apply(lambda values: dedupe_join(values, limit=5))
        .to_dict()
    )
    top_query_map = (
        rows.groupby(["mapped_feature_id", "cluster_id"], dropna=False)["canonical_query"]
        .agg(lambda values: pd.Series(list(values)).value_counts().index[0])
        .to_dict()
    )

    overview_records: list[dict[str, object]] = []
    for _, target_row in target_coverage.iterrows():
        cluster_mask = (
            (coverage["mapped_feature_id"] == target_row["mapped_feature_id"])
            & (coverage["cluster_id"] == target_row["cluster_id"])
            & (coverage["brand_id"] != target_brand_id)
        )
        competitor_rows = coverage[cluster_mask].copy()
        competitor_rows["coverage_rate"] = pd.to_numeric(
            competitor_rows["coverage_rate"],
            errors="coerce",
        ).fillna(0.0)
        competitor_rows = competitor_rows.sort_values("coverage_rate", ascending=False)
        top_competitor = competitor_rows.iloc[0] if not competitor_rows.empty else None
        competitor_visibility_share = float(top_competitor["coverage_rate"]) if top_competitor is not None else 0.0
        competitor_present_count = int((competitor_rows["brand_present_count"] > 0).sum()) if not competitor_rows.empty else 0
        competitor_present = competitor_present_count > 0
        visibility_share = float(pd.to_numeric(pd.Series([target_row["coverage_rate"]]), errors="coerce").fillna(0.0).iloc[0])
        consistency = consistency_band(visibility_share)
        status = visibility_status(visibility_share)
        category = gap_category(visibility_share, competitor_present)
        is_gap = is_feature_visibility_gap(visibility_share, competitor_present)
        severity = gap_severity(visibility_share, competitor_present)
        key = (target_row["mapped_feature_id"], target_row["cluster_id"])
        overview_records.append(
            {
                "target_brand_id": target_brand_id,
                "target_brand_name": target_brand_name,
                "feature_intent_detected": True,
                "mapped_feature_id": target_row["mapped_feature_id"],
                "mapped_feature_name": target_row["mapped_feature_name"],
                "cluster_id": target_row["cluster_id"],
                "cluster_label": target_row["cluster_label"],
                "visibility_share": round(visibility_share, 4),
                "consistency_band": consistency,
                "target_visibility_status": status,
                "competitor_present": competitor_present,
                "is_feature_visibility_gap": is_gap,
                "gap_category": category,
                "gap_severity": severity,
                "gap_reason": gap_reason(visibility_share, competitor_present),
                "signal": severity_signal(severity, category),
                "prompt_count": int(target_row["prompt_count"]),
                "target_brand_present_count": int(target_row["brand_present_count"]),
                "target_brand_absent_count": int(target_row["brand_absent_count"]),
                "top_competitor_brand_id": str(top_competitor["brand_id"]) if top_competitor is not None else "",
                "top_competitor_brand_name": str(top_competitor["brand_name"]) if top_competitor is not None else "",
                "top_competitor_visibility_share": round(competitor_visibility_share, 4),
                "competitor_present_count": competitor_present_count,
                "top_query": top_query_map.get(key, ""),
                "example_queries": all_example_queries.get(key, ""),
                "present_prompt_ids": target_row["present_prompt_ids"],
                "missing_prompt_ids": target_row["missing_prompt_ids"],
            }
        )
    return pd.DataFrame.from_records(overview_records).sort_values(
        ["gap_severity", "visibility_share", "mapped_feature_name"],
        ascending=[True, True, True],
    )


def build_feature_gap_details(
    coverage: pd.DataFrame,
    rows: pd.DataFrame,
    target_brand: pd.Series,
) -> pd.DataFrame:
    overview = build_feature_gap_overview(coverage, rows, target_brand)
    if overview.empty:
        return pd.DataFrame()

    detail_records: list[dict[str, object]] = []
    for _, overview_row in overview.iterrows():
        cluster_rows = coverage[
            (coverage["mapped_feature_id"] == overview_row["mapped_feature_id"])
            & (coverage["cluster_id"] == overview_row["cluster_id"])
        ].copy()
        brand_comparison = (
            cluster_rows[["brand_id", "brand_name", "coverage_rate", "brand_present_count", "brand_absent_count"]]
            .fillna("")
            .to_dict(orient="records")
        )
        example_queries = rows[
            (rows["mapped_feature_id"] == overview_row["mapped_feature_id"])
            & (rows["cluster_id"] == overview_row["cluster_id"])
        ]["canonical_query"]
        detail_records.append(
            {
                **overview_row.to_dict(),
                "brand_comparison_json": json.dumps(brand_comparison, ensure_ascii=False),
                "example_queries_json": json.dumps(list(dict.fromkeys(example_queries.tolist()))[:10], ensure_ascii=False),
            }
        )
    return pd.DataFrame.from_records(detail_records)


def build_pm_summary(feature_gap_overview: pd.DataFrame, target_brand: pd.Series) -> str:
    brand_name = str(target_brand["brand_name"])
    if feature_gap_overview.empty:
        return f"# Feature Visibility Gaps\n\nNo feature gaps were generated for {brand_name}.\n"

    lines = [f"# Feature Visibility Gaps", "", f"Brand: {brand_name}", ""]
    current_feature = None
    for _, row in feature_gap_overview.sort_values(["mapped_feature_name", "visibility_share"]).iterrows():
        feature_name = str(row["mapped_feature_name"])
        if feature_name != current_feature:
            if current_feature is not None:
                lines.append("")
            lines.extend([f"## {feature_name}", "", "| Demand cluster | Visibility share | Signal | Top competitor |", "| --- | ---: | --- | --- |"])
            current_feature = feature_name
        lines.append(
            f"| {row['cluster_label']} | {row['visibility_share'] * 100:.1f}% | {row['signal']} | {row['top_competitor_brand_name'] or '-'} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def join_ids(values: Iterable[object]) -> str:
    return ";".join(str(value) for value in values)


def write_outputs(
    output_dir: Path,
    rows: pd.DataFrame,
    coverage: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    feature_gap_overview: pd.DataFrame,
    feature_gap_details: pd.DataFrame,
    pm_summary: str,
    config: Config,
    brands: pd.DataFrame,
    target_brand: pd.Series,
) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows.to_csv(output_dir / "query_mapping.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    coverage.to_csv(output_dir / "coverage_by_feature_cluster.csv", index=False)
    cluster_summary.to_csv(output_dir / "clusters.csv", index=False)
    feature_gap_overview.to_csv(output_dir / "feature_gap_overview.csv", index=False)
    feature_gap_details.to_csv(output_dir / "feature_gap_details.csv", index=False)
    (output_dir / "feature_gap_summary.md").write_text(pm_summary, encoding="utf-8")
    metadata = {
        "generated_at": timestamp,
        "prompts_csv": str(config.prompts_csv),
        "features_csv": str(config.features_csv),
        "brand": config.brand,
        "brands_csv": str(config.brands_csv) if config.brands_csv else None,
        "brand_count": len(brands),
        "target_brand_id": str(target_brand["brand_id"]),
        "target_brand_name": str(target_brand["brand_name"]),
        "embedding_backend": config.embedding_backend,
        "embedding_model": config.embedding_model,
        "normalizer": config.normalizer,
        "brand_detector": config.brand_detector,
        "brand_detector_model": config.brand_detector_model,
        "cluster_threshold": config.cluster_threshold,
        "feature_threshold": config.feature_threshold,
        "min_cluster_size": config.min_cluster_size,
        "min_coverage_n": config.min_coverage_n,
        "aggregation_mode": config.aggregation_mode,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def print_summary(
    output_dir: Path,
    rows: pd.DataFrame,
    coverage: pd.DataFrame,
    feature_gap_overview: pd.DataFrame,
    target_brand: pd.Series,
    config: Config,
    brands: pd.DataFrame,
) -> None:
    print(f"Wrote outputs to {output_dir}")
    print(f"Prompt-brand rows evaluated: {len(rows)}")
    print(f"Prompts evaluated: {rows['prompt_id'].nunique()}")
    print(f"Brands evaluated: {len(brands)}")
    print(f"Matched rows: {int((rows['mapped_feature_id'].astype(str) != '').sum())}")
    print(f"Clusters: {rows['cluster_id'].nunique()}")
    print(f"Brand source: {config.brands_csv or config.brand!r}")
    print(f"Brand detector: {config.brand_detector}")
    print(f"Target brand: {target_brand['brand_name']}")
    if coverage.empty:
        print("No feature/cluster coverage rows were produced.")
        return
    preview = coverage.head(10).to_string(index=False)
    print("\nCoverage preview:")
    print(preview)
    if not feature_gap_overview.empty:
        gap_preview = feature_gap_overview[
            ["mapped_feature_name", "cluster_label", "visibility_share", "gap_severity", "top_competitor_brand_name"]
        ].head(10).to_string(index=False)
        print("\nFeature gap overview preview:")
        print(gap_preview)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
