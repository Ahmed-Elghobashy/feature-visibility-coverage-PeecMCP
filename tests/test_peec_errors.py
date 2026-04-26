from __future__ import annotations

import sys
import unittest
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from peec_mcp_export import format_error, has_http_status  # noqa: E402


class PeecErrorFormattingTests(unittest.TestCase):
    def test_401_inside_exception_group_is_detected_and_formatted(self) -> None:
        request = httpx.Request("POST", "https://api.peec.ai/mcp")
        response = httpx.Response(401, request=request)
        error = httpx.HTTPStatusError("unauthorized", request=request, response=response)
        group = ExceptionGroup("taskgroup", [error])

        self.assertTrue(has_http_status(group, 401))
        self.assertEqual(
            format_error(group),
            "Peec MCP returned 401 Unauthorized. Reconnect your Peec OAuth session and try again.",
        )


if __name__ == "__main__":
    unittest.main()
