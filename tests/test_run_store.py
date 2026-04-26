from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import run_store as rs  # noqa: E402
from visibility_mvp import (  # noqa: E402
    Config,
    build_cluster_summary,
    build_coverage,
    build_feature_gap_details,
    build_feature_gap_overview,
    build_pm_summary,
    resolve_target_brand,
)


class RunStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        base = Path(self.tempdir.name)
        self.originals = (
            rs.STORE_ROOT,
            rs.RUNS_DIR,
            rs.RUNS_INDEX,
            rs.FEATURE_SETS_DIR,
            rs.FEATURE_SETS_INDEX,
        )
        rs.STORE_ROOT = base / "store"
        rs.RUNS_DIR = rs.STORE_ROOT / "runs"
        rs.RUNS_INDEX = rs.STORE_ROOT / "runs_index.csv"
        rs.FEATURE_SETS_DIR = rs.STORE_ROOT / "feature_sets"
        rs.FEATURE_SETS_INDEX = rs.STORE_ROOT / "feature_sets.csv"
        self.base = base

    def tearDown(self) -> None:
        (
            rs.STORE_ROOT,
            rs.RUNS_DIR,
            rs.RUNS_INDEX,
            rs.FEATURE_SETS_DIR,
            rs.FEATURE_SETS_INDEX,
        ) = self.originals
        self.tempdir.cleanup()

    def test_persist_and_list_runs(self) -> None:
        prompts_csv, features_csv, brands_csv, output_dir = self.build_run_artifacts("prompt")
        manifest = rs.persist_run(
            source="csv",
            prompts_csv=prompts_csv,
            features_csv=features_csv,
            brands_csv=brands_csv,
            output_dir=output_dir,
            target_brand="Peec AI",
            aggregation_mode="prompt",
        )

        runs = rs.list_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], manifest["run_id"])

        result = rs.load_run_result(manifest["run_id"])
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["overview"]), 1)

    def test_load_run_mappings_filters_feature_and_cluster(self) -> None:
        prompts_csv, features_csv, brands_csv, output_dir = self.build_run_artifacts("prompt")
        manifest = rs.persist_run(
            source="csv",
            prompts_csv=prompts_csv,
            features_csv=features_csv,
            brands_csv=brands_csv,
            output_dir=output_dir,
            target_brand="Peec AI",
            aggregation_mode="prompt",
        )

        mappings = rs.load_run_mappings(manifest["run_id"], feature_id="f1", cluster_id="c1")
        self.assertEqual(len(mappings), 2)
        self.assertTrue(all(row["mapped_feature_id"] == "f1" for row in mappings))

    def test_reaggregate_saved_run_creates_new_run(self) -> None:
        prompts_csv, features_csv, brands_csv, output_dir = self.build_run_artifacts("response")
        manifest = rs.persist_run(
            source="csv",
            prompts_csv=prompts_csv,
            features_csv=features_csv,
            brands_csv=brands_csv,
            output_dir=output_dir,
            target_brand="Peec AI",
            aggregation_mode="response",
        )

        new_manifest = rs.reaggregate_saved_run(manifest["run_id"], "prompt")
        self.assertNotEqual(new_manifest["run_id"], manifest["run_id"])
        self.assertEqual(new_manifest["parent_run_id"], manifest["run_id"])
        result = rs.load_run_result(new_manifest["run_id"])
        self.assertEqual(result["manifest"]["aggregation_mode"], "prompt")

    def build_run_artifacts(self, aggregation_mode: str) -> tuple[Path, Path, Path, Path]:
        prompts_csv = self.base / "prompts.csv"
        features_csv = self.base / "features.csv"
        brands_csv = self.base / "brands.csv"
        output_dir = self.base / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"prompt_id": "p1", "prompt": "best ai search tools", "response": "Peec AI is good"},
                {"prompt_id": "p2", "prompt": "ai search visibility tools", "response": "Another tool"},
            ]
        ).to_csv(prompts_csv, index=False)
        pd.DataFrame(
            [
                {"feature_id": "f1", "feature_name": "AI Search Visibility Tracking", "description": "Tracks visibility"},
            ]
        ).to_csv(features_csv, index=False)
        pd.DataFrame(
            [
                {"brand_id": "b1", "brand_name": "Peec AI", "aliases": ""},
                {"brand_id": "b2", "brand_name": "Profound", "aliases": ""},
            ]
        ).to_csv(brands_csv, index=False)

        brand_rows = pd.DataFrame(
            [
                {
                    "prompt_id": "p1",
                    "original_prompt": "best ai search tools",
                    "canonical_query": "ai search tools",
                    "cluster_id": "c1",
                    "cluster_label": "ai search tools",
                    "mapped_feature_id": "f1",
                    "mapped_feature_name": "AI Search Visibility Tracking",
                    "feature_similarity": 0.9,
                    "brand_detection_source": "response",
                    "feature_present": True,
                    "feature_evidence_strength": 2,
                    "engine": "chatgpt",
                    "source_domains": "peec.ai;docs.peec.ai",
                    "brand_id": "b1",
                    "brand_name": "Peec AI",
                    "brand_aliases": "",
                    "brand_present": True,
                },
                {
                    "prompt_id": "p2",
                    "original_prompt": "ai search visibility tools",
                    "canonical_query": "ai search tools",
                    "cluster_id": "c1",
                    "cluster_label": "ai search tools",
                    "mapped_feature_id": "f1",
                    "mapped_feature_name": "AI Search Visibility Tracking",
                    "feature_similarity": 0.9,
                    "brand_detection_source": "response",
                    "feature_present": True,
                    "feature_evidence_strength": 2,
                    "engine": "chatgpt",
                    "source_domains": "peec.ai;docs.peec.ai",
                    "brand_id": "b1",
                    "brand_name": "Peec AI",
                    "brand_aliases": "",
                    "brand_present": False,
                },
                {
                    "prompt_id": "p1",
                    "original_prompt": "best ai search tools",
                    "canonical_query": "ai search tools",
                    "cluster_id": "c1",
                    "cluster_label": "ai search tools",
                    "mapped_feature_id": "f1",
                    "mapped_feature_name": "AI Search Visibility Tracking",
                    "feature_similarity": 0.9,
                    "brand_detection_source": "response",
                    "feature_present": True,
                    "feature_evidence_strength": 2,
                    "engine": "chatgpt",
                    "source_domains": "peec.ai;docs.peec.ai",
                    "brand_id": "b2",
                    "brand_name": "Profound",
                    "brand_aliases": "",
                    "brand_present": True,
                },
                {
                    "prompt_id": "p2",
                    "original_prompt": "ai search visibility tools",
                    "canonical_query": "ai search tools",
                    "cluster_id": "c1",
                    "cluster_label": "ai search tools",
                    "mapped_feature_id": "f1",
                    "mapped_feature_name": "AI Search Visibility Tracking",
                    "feature_similarity": 0.9,
                    "brand_detection_source": "response",
                    "feature_present": True,
                    "feature_evidence_strength": 2,
                    "engine": "chatgpt",
                    "source_domains": "peec.ai;docs.peec.ai",
                    "brand_id": "b2",
                    "brand_name": "Profound",
                    "brand_aliases": "",
                    "brand_present": True,
                },
            ]
        )
        prompt_rows = brand_rows[
            ["prompt_id", "canonical_query", "cluster_id", "cluster_label", "mapped_feature_id", "mapped_feature_name"]
        ].drop_duplicates()
        brands = pd.read_csv(brands_csv)
        config = Config(
            prompts_csv=prompts_csv,
            features_csv=features_csv,
            brand=None,
            brands_csv=brands_csv,
            target_brand="Peec AI",
            target_brand_id="b1",
            output_dir=output_dir,
            embedding_backend="hash",
            embedding_model="BAAI/bge-m3",
            normalizer="openai_mock",
            normalizer_model="gpt-4.1-mini",
            brand_detector="openai_mock",
            brand_detector_model="gpt-4.1-mini",
            feature_evidence_mode="openai_mock",
            feature_evidence_model="gpt-4.1-mini",
            cluster_threshold=0.2,
            feature_threshold=0.05,
            min_cluster_size=2,
            min_coverage_n=1,
            aggregation_mode=aggregation_mode,
        )
        target_brand = resolve_target_brand(brands, config)
        coverage = build_coverage(brand_rows, 1, aggregation_mode)
        cluster_summary = build_cluster_summary(prompt_rows)
        overview = build_feature_gap_overview(coverage, prompt_rows, target_brand)
        details = build_feature_gap_details(coverage, prompt_rows, target_brand)
        summary = build_pm_summary(overview, target_brand)

        brand_rows.to_csv(output_dir / "query_mapping.csv", index=False)
        coverage.to_csv(output_dir / "coverage_by_feature_cluster.csv", index=False)
        cluster_summary.to_csv(output_dir / "clusters.csv", index=False)
        overview.to_csv(output_dir / "feature_gap_overview.csv", index=False)
        details.to_csv(output_dir / "feature_gap_details.csv", index=False)
        (output_dir / "feature_gap_summary.md").write_text(summary, encoding="utf-8")
        (output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "target_brand_id": "b1",
                    "target_brand_name": "Peec AI",
                    "aggregation_mode": aggregation_mode,
                    "embedding_backend": "hash",
                    "embedding_model": "BAAI/bge-m3",
                    "normalizer": "openai_mock",
                    "brand_detector": "openai_mock",
                    "brand_detector_model": "gpt-4.1-mini",
                    "cluster_threshold": 0.2,
                    "feature_threshold": 0.05,
                    "min_cluster_size": 2,
                    "min_coverage_n": 1,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return prompts_csv, features_csv, brands_csv, output_dir


if __name__ == "__main__":
    unittest.main()
