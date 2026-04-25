from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from visibility_mvp import (  # noqa: E402
    aggregate_coverage_group,
    build_coverage,
    build_feature_gap_overview,
    contains_brand,
    split_aliases,
)


class BrandAliasTests(unittest.TestCase):
    def test_aliases_are_split_from_common_separators(self) -> None:
        self.assertEqual(
            split_aliases("PeecAI|peec.ai,Peec;Peec Search"),
            ["PeecAI", "peec.ai", "Peec", "Peec Search"],
        )

    def test_contains_brand_matches_aliases_case_insensitively(self) -> None:
        self.assertTrue(contains_brand("Teams use peec.ai for AI search tracking.", ["Peec AI", "peec.ai"]))
        self.assertFalse(contains_brand("Teams use another AI search tool.", ["Peec AI", "peec.ai"]))


class AggregationTests(unittest.TestCase):
    def test_prompt_aggregation_counts_repeated_prompt_once(self) -> None:
        group = pd.DataFrame(
            [
                {"prompt_id": "p1", "engine": "chatgpt", "brand_present": False},
                {"prompt_id": "p1", "engine": "chatgpt", "brand_present": True},
                {"prompt_id": "p2", "engine": "chatgpt", "brand_present": False},
            ]
        )

        aggregated = aggregate_coverage_group(group, "prompt")

        self.assertEqual(len(aggregated), 2)
        self.assertEqual(int(aggregated["brand_present"].sum()), 1)

    def test_prompt_model_aggregation_keeps_different_engines(self) -> None:
        group = pd.DataFrame(
            [
                {"prompt_id": "p1", "engine": "chatgpt", "brand_present": True},
                {"prompt_id": "p1", "engine": "gemini", "brand_present": False},
            ]
        )

        aggregated = aggregate_coverage_group(group, "prompt_model")

        self.assertEqual(len(aggregated), 2)


class GapClassificationTests(unittest.TestCase):
    def test_competitor_present_and_target_low_is_high_gap(self) -> None:
        coverage = build_coverage(self.rows(target_present=1, competitor_present=2), 1, "response")
        overview = build_feature_gap_overview(coverage, self.prompt_rows(), self.target_brand())
        row = overview.iloc[0]

        self.assertTrue(bool(row["is_feature_visibility_gap"]))
        self.assertTrue(bool(row["competitor_present"]))
        self.assertEqual(row["target_visibility_status"], "inconsistent")
        self.assertEqual(row["gap_category"], "competitive_gap")
        self.assertEqual(row["gap_severity"], "high")

    def test_target_weak_without_competitor_is_not_strict_gap(self) -> None:
        coverage = build_coverage(self.rows(target_present=0, competitor_present=0), 1, "response")
        overview = build_feature_gap_overview(coverage, self.prompt_rows(), self.target_brand())
        row = overview.iloc[0]

        self.assertFalse(bool(row["is_feature_visibility_gap"]))
        self.assertFalse(bool(row["competitor_present"]))
        self.assertEqual(row["gap_category"], "weak_category_visibility")
        self.assertEqual(row["gap_severity"], "low")

    def test_target_strong_is_not_gap_even_with_competitor(self) -> None:
        coverage = build_coverage(self.rows(target_present=3, competitor_present=1), 1, "response")
        overview = build_feature_gap_overview(coverage, self.prompt_rows(), self.target_brand())
        row = overview.iloc[0]

        self.assertFalse(bool(row["is_feature_visibility_gap"]))
        self.assertTrue(bool(row["competitor_present"]))
        self.assertEqual(row["target_visibility_status"], "strong")
        self.assertEqual(row["gap_category"], "strong_presence")
        self.assertEqual(row["gap_severity"], "low")

    def rows(self, target_present: int, competitor_present: int) -> pd.DataFrame:
        records = []
        for brand_id, brand_name, present_count in [
            ("b1", "Peec AI", target_present),
            ("b2", "Profound", competitor_present),
        ]:
            for idx in range(3):
                records.append(
                    {
                        "brand_id": brand_id,
                        "brand_name": brand_name,
                        "brand_aliases": "",
                        "mapped_feature_id": "f1",
                        "mapped_feature_name": "AI Search Visibility Tracking",
                        "cluster_id": "c1",
                        "cluster_label": "ai search visibility tools",
                        "prompt_id": f"p{idx + 1}",
                        "brand_present": idx < present_count,
                    }
                )
        return pd.DataFrame.from_records(records)

    def prompt_rows(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "mapped_feature_id": "f1",
                    "mapped_feature_name": "AI Search Visibility Tracking",
                    "cluster_id": "c1",
                    "cluster_label": "ai search visibility tools",
                    "canonical_query": "ai search visibility tools",
                }
            ]
        )

    def target_brand(self) -> pd.Series:
        return pd.Series({"brand_id": "b1", "brand_name": "Peec AI"})


if __name__ == "__main__":
    unittest.main()
