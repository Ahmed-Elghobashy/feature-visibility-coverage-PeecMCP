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

PROMPT_COLUMNS = ("prompt", "raw_prompt", "query", "question", "text")
RESPONSE_COLUMNS = ("response", "answer", "ai_response", "model_response", "output")
FEATURE_NAME_COLUMNS = ("feature", "feature_name", "name", "title")
FEATURE_DESCRIPTION_COLUMNS = ("description", "feature_description", "desc")
BRAND_NAME_COLUMNS = ("brand", "brand_name", "name")


@dataclass(frozen=True)
class Config:
    prompts_csv: Path
    features_csv: Path
    brand: str | None
    brands_csv: Path | None
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


def main() -> int:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(Path(".env"))

    prompts = pd.read_csv(config.prompts_csv)
    features = pd.read_csv(config.features_csv)
    brands = load_brands(config)

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
        else "heuristic:v1"
    )
    prompts["canonical_query"] = [
        normalize_prompt(str(prompt), config, brands)
        for prompt in prompts[prompt_col].fillna("")
    ]

    embedder = make_embedder(config)
    query_vectors = embedder.encode(prompts["canonical_query"].tolist())
    feature_texts = [
        f"{row[feature_name_col]}. {row[feature_desc_col]}"
        for _, row in features.iterrows()
    ]
    feature_vectors = embedder.encode(feature_texts)

    clusters = cluster_vectors(
        query_vectors,
        threshold=config.cluster_threshold,
        min_cluster_size=config.min_cluster_size,
    )
    prompts["cluster_id"] = clusters

    cluster_labels = label_clusters(prompts, query_vectors)
    prompts["cluster_label"] = prompts["cluster_id"].map(cluster_labels)

    feature_mapping = map_features(
        query_vectors=query_vectors,
        feature_vectors=feature_vectors,
        features=features,
        feature_name_col=feature_name_col,
        threshold=config.feature_threshold,
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
        else "casefold-word-boundary:v1"
    )
    rows["mapped_feature_id"] = [m["feature_id"] for m in feature_mapping]
    rows["mapped_feature_name"] = [m["feature_name"] for m in feature_mapping]
    rows["feature_similarity"] = [m["similarity"] for m in feature_mapping]
    rows["feature_match_status"] = [
        "matched" if m["feature_id"] else "unmatched"
        for m in feature_mapping
    ]

    brand_rows = expand_brand_rows(rows, brands, text_for_brand_detection, config)
    coverage = build_coverage(brand_rows, config.min_coverage_n)
    cluster_summary = build_cluster_summary(rows)

    write_outputs(config.output_dir, brand_rows, coverage, cluster_summary, config, brands)
    print_summary(config.output_dir, brand_rows, coverage, config, brands)
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
        choices=("heuristic", "openai"),
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
        choices=("keyword", "openai"),
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
    args = parser.parse_args()
    if not args.brand and not args.brands:
        parser.error("Provide --brand for one brand or --brands for a brands CSV.")

    return Config(
        prompts_csv=args.prompts,
        features_csv=args.features,
        brand=args.brand,
        brands_csv=args.brands,
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
        brands["brand_name"] = brands[brand_name_col].astype(str)
        brands["brand_id"] = ensure_id(brands, "brand_id", "brand")
    else:
        assert config.brand
        brands = pd.DataFrame(
            [{"brand_id": "brand_001", "brand_name": config.brand}]
        )

    brands = brands[brands["brand_name"].astype(str).str.strip() != ""].copy()
    if brands.empty:
        raise ValueError("No non-empty brands were provided.")
    return brands[["brand_id", "brand_name"]].drop_duplicates().reset_index(drop=True)


def ensure_id(frame: pd.DataFrame, column: str, prefix: str) -> list[str]:
    if column in frame.columns:
        return [str(value) for value in frame[column]]
    width = max(3, int(math.log10(max(len(frame), 1))) + 1)
    return [f"{prefix}_{i:0{width}d}" for i in range(1, len(frame) + 1)]


def normalize_prompt(prompt: str, config: Config, brands: pd.DataFrame) -> str:
    if config.normalizer == "openai":
        return normalize_prompt_openai(prompt, config.normalizer_model, brands)
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
    return re.sub(r"\s+", " ", content.strip().casefold())


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


def contains_brand(text: str, brand: str) -> bool:
    if not brand.strip():
        return False
    escaped = re.escape(brand.strip())
    pattern = rf"(?<![\w-]){escaped}(?![\w-])"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def judge_brand_openai(text: str, brand: str, model: str) -> bool:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("--brand-detector openai requires OPENAI_API_KEY.")

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
    return bool(parsed.get("brand_present"))


def detect_brand(text: str, brand: str, config: Config) -> bool:
    if config.brand_detector == "openai":
        return judge_brand_openai(text, brand, config.brand_detector_model)
    return contains_brand(text, brand)


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
        brand_rows["brand_id"] = str(brand["brand_id"])
        brand_rows["brand_name"] = brand_name
        brand_rows["brand_present"] = [
            detect_brand(text, brand_name, config)
            for text in detection_values.astype(str)
        ]
        expanded.append(brand_rows)
    return pd.concat(expanded, ignore_index=True)


def build_coverage(rows: pd.DataFrame, min_coverage_n: int) -> pd.DataFrame:
    matched = rows[rows["mapped_feature_id"].astype(str) != ""].copy()
    if matched.empty:
        return pd.DataFrame(
            columns=[
                "brand_id",
                "brand_name",
                "mapped_feature_id",
                "mapped_feature_name",
                "cluster_id",
                "cluster_label",
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
        "mapped_feature_id",
        "mapped_feature_name",
        "cluster_id",
        "cluster_label",
    ]
    for keys, group in matched.groupby(group_cols, dropna=False, sort=True):
        prompt_count = len(group)
        present = int(group["brand_present"].sum())
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
                "mapped_feature_id": keys[2],
                "mapped_feature_name": keys[3],
                "cluster_id": keys[4],
                "cluster_label": keys[5],
                "prompt_count": prompt_count,
                "brand_present_count": present,
                "brand_absent_count": missing,
                "coverage_rate": coverage_rate,
                "coverage_status": status,
                "present_prompt_ids": join_ids(group[group["brand_present"]]["prompt_id"]),
                "missing_prompt_ids": join_ids(group[~group["brand_present"]]["prompt_id"]),
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


def join_ids(values: Iterable[object]) -> str:
    return ";".join(str(value) for value in values)


def write_outputs(
    output_dir: Path,
    rows: pd.DataFrame,
    coverage: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    config: Config,
    brands: pd.DataFrame,
) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows.to_csv(output_dir / "query_mapping.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    coverage.to_csv(output_dir / "coverage_by_feature_cluster.csv", index=False)
    cluster_summary.to_csv(output_dir / "clusters.csv", index=False)
    metadata = {
        "generated_at": timestamp,
        "prompts_csv": str(config.prompts_csv),
        "features_csv": str(config.features_csv),
        "brand": config.brand,
        "brands_csv": str(config.brands_csv) if config.brands_csv else None,
        "brand_count": len(brands),
        "embedding_backend": config.embedding_backend,
        "embedding_model": config.embedding_model,
        "normalizer": config.normalizer,
        "brand_detector": config.brand_detector,
        "brand_detector_model": config.brand_detector_model,
        "cluster_threshold": config.cluster_threshold,
        "feature_threshold": config.feature_threshold,
        "min_cluster_size": config.min_cluster_size,
        "min_coverage_n": config.min_coverage_n,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def print_summary(
    output_dir: Path,
    rows: pd.DataFrame,
    coverage: pd.DataFrame,
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
    if coverage.empty:
        print("No feature/cluster coverage rows were produced.")
        return
    preview = coverage.head(10).to_string(index=False)
    print("\nCoverage preview:")
    print(preview)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
