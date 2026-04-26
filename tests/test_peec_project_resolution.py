from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from peec_mcp_export import ExportConfig, resolve_project_id  # noqa: E402


def config(*, project: str | None = None, project_id: str | None = None) -> ExportConfig:
    return ExportConfig(
        project=project,
        project_id=project_id,
        start_date="2026-04-01",
        end_date="2026-04-02",
        output=ROOT / "data" / "tmp.csv",
        server_url="https://api.peec.ai/mcp",
        token_path=ROOT / ".peec_mcp_tokens.json",
        limit=100,
        topic_id=None,
        tag_id=None,
        model_id=None,
        brand_id=None,
        list_projects=False,
        list_tools=False,
        connect_timeout=30.0,
    )


class ProjectResolutionTests(unittest.TestCase):
    def test_missing_project_defaults_to_first_accessible_project(self) -> None:
        projects = pd.DataFrame(
            [
                {"id": "p1", "name": "First Project"},
                {"id": "p2", "name": "Second Project"},
            ]
        )

        self.assertEqual(resolve_project_id(projects, config()), "p1")

    def test_explicit_project_name_still_filters(self) -> None:
        projects = pd.DataFrame(
            [
                {"id": "p1", "name": "First Project"},
                {"id": "p2", "name": "Second Project"},
            ]
        )

        self.assertEqual(resolve_project_id(projects, config(project="second")), "p2")


if __name__ == "__main__":
    unittest.main()
