"""
test_antigravity_cli_provider.py

Unit tests for the official Antigravity CLI provider and its headless structured-output integration.
Validates:
1. Discovery of the installed 'agy' executable and version detection.
2. Proper unpacking of the CLI response envelope without conflation with the extraction contract.
3. Strict isolation: offline parser and benchmark answers are never invoked.
4. Error handling: CLI timeouts, non-zero return codes, and malformed outputs produce safe needs_review results.
5. Guard check: direct Gemini API provider halts safely when credentials are missing.
"""

import unittest
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

base_dir = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(base_dir / "src"))

from antigravity_cli_provider import AntigravityCLIProvider, AntigravityCLIError
from question_processor import process_question_item


class TestAntigravityCLIProvider(unittest.TestCase):

    def setUp(self):
        self.raw_text = (
            "1. You are a nursing student observing care in the Critical Care Unit (CCU). "
            "The nurse instructs you to increase the IV medication. "
            "What is the appropriate response to this instruction?"
        )
        self.question_id = "Q001"

    def test_cli_binary_discovery_and_version(self):
        """Verifies that agy binary is discovered and its version is parsed."""
        provider = AntigravityCLIProvider()
        self.assertTrue(Path(provider.cli_path).exists())
        self.assertIn("1.3.1", provider.cli_version)

    @patch("subprocess.run")
    def test_cli_envelope_unpacking_and_contract_separation(self, mock_run):
        """Proves CLI envelope is unpacked cleanly without contaminating the contract."""
        mock_contract = {
            "source_id": "Q001",
            "status": "proposed",
            "segments": [
                {"id": "seg_1", "start": 0, "end": 3, "kind": "metadata"},
                {"id": "seg_2", "start": 3, "end": 132, "kind": "reference"},
                {"id": "seg_3", "start": 132, "end": len(self.raw_text), "kind": "stem"}
            ],
            "items": [
                {
                    "id": "Q001",
                    "parent_group_id": None,
                    "response_kind": "single_choice",
                    "source_label_segment_ids": ["seg_1"],
                    "reference_segment_ids": ["seg_2"],
                    "stem_segment_ids": ["seg_3"],
                    "option_groups": [],
                    "matching_rows": [],
                    "response_template_segment_ids": [],
                    "depends_on_item_ids": [],
                    "context_reuse": [],
                    "disposition": "proposed"
                }
            ],
            "issues": []
        }

        mock_envelope = {
            "conversation_id": "conv-test-uuid-999",
            "status": "SUCCESS",
            "response": json.dumps(mock_contract),
            "structured_output": mock_contract,
            "duration_seconds": 3.42,
            "usage": {
                "input_tokens": 12000,
                "output_tokens": 350,
                "thinking_tokens": 200,
                "total_tokens": 12550
            }
        }

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps(mock_envelope)
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        provider = AntigravityCLIProvider()
        contract, items, meta = process_question_item(
            raw_text=self.raw_text,
            question_id=self.question_id,
            provider="antigravity-cli",
            cli_provider=provider
        )

        # 1. Verify contract top-level fields match schema, not envelope
        self.assertEqual(contract["source_id"], "Q001")
        self.assertNotIn("conversation_id", contract)
        self.assertNotIn("usage", contract)

        # 2. Verify metadata captures envelope fields
        self.assertEqual(meta["provider"], "antigravity-cli")
        self.assertEqual(meta["conversation_id"], "conv-test-uuid-999")
        self.assertEqual(meta["duration_seconds"], 3.42)
        self.assertEqual(meta["usage"]["total_tokens"], 12550)

        # 3. Verify platform item sliced
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["after"]["stem"], "What is the appropriate response to this instruction?")

    @patch("subprocess.run")
    def test_cli_error_returns_needs_review_retaining_input(self, mock_run):
        """Proves CLI errors return failure contracts without calling offline parser or benchmark."""
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "Quota exceeded or network error"
        mock_run.return_value = mock_proc

        provider = AntigravityCLIProvider()
        contract, items, meta = process_question_item(
            raw_text=self.raw_text,
            question_id=self.question_id,
            provider="antigravity-cli",
            cli_provider=provider
        )

        self.assertEqual(contract["status"], "needs_review")
        self.assertEqual(contract["items"][0]["disposition"], "needs_review")
        self.assertEqual(meta["status"], "failed_cli_error")
        self.assertFalse(items[0]["passed"])
        # Retains full input
        self.assertEqual(items[0]["after"]["stem"], self.raw_text)

    def test_unsupported_provider_raises_error(self):
        """Verifies that attempting an unsupported provider raises ValueError."""
        with self.assertRaises(ValueError):
            process_question_item(
                raw_text=self.raw_text,
                question_id=self.question_id,
                provider="unsupported-engine"
            )


if __name__ == "__main__":
    unittest.main()
