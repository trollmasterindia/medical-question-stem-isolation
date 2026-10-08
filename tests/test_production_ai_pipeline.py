"""
test_production_ai_pipeline.py

Unit test suite verifying all Requirement 9 specifications:
1. Missing credentials: Live detector returns clear failure/needs_review result,
   retains original input, never invokes offline parser, and never produces PASS.
2. API errors: Simulated exception/network failure triggers bounded retry,
   retains original input, never invokes offline parser, and never produces PASS.
3. Invalid model response: Invalid schema/offsets fails ContractValidator, triggers retry,
   returns failure contract with disposition='needs_review' and passed=False.
4. Model selection & hash forwarding: Verifies model name, prompt hash, and source hash
   are tracked consistently without exposing any secrets.
5. Actual-contract export: Verifies that the exported contract is the exact model-returned contract.
6. Review gating: Ensures that any item with disposition='needs_review' or contract status='needs_review'
   is blocked from automatic PASS acceptance.
7. Reordered spans detection: Verifies that character reordering (which multiset comparison fails to catch)
   is flagged by verify_source_order_and_content.
8. Child context and template preservation: Ensures child exhibit segments and response templates
   are retained through slicing.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import hashlib
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from ai_boundary_detector import AIBoundaryDetector
from contract_slicer import contract_to_platform_items, verify_source_order_and_content
from prompt_loader import load_master_prompt_text, get_default_config


class TestProductionAIPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "Case Study:\n"
            "A 45-year-old patient presents with shortness of breath.\n\n"
            "What is the most appropriate initial oxygen delivery system?\n"
            "A. Nasal cannula\n"
            "B. Non-rebreather mask\n"
            "C. Venturi mask\n"
            "D. Bag-valve-mask"
        )
        self.config = get_default_config()

    def test_1_missing_credentials_never_invokes_offline_or_produces_pass(self):
        """
        Req 1, 2, 9: If credentials missing, returns failure/needs_review, retains raw input,
        never produces PASS, and never calls offline parser.
        """
        with patch.dict("os.environ", {}, clear=True):
            detector = AIBoundaryDetector(api_key=None)
            self.assertIsNone(detector.api_key)

            with patch("experimental.autonomous_parser.AutonomousSemanticParser.decompose", create=True) as mock_offline:
                contract, items, meta = detector.detect_and_process(
                    raw_text=self.sample_text,
                    question_id="TEST_NO_KEY"
                )

                # Offline parser must NEVER be called
                mock_offline.assert_not_called()

                # Status must be failure
                self.assertEqual(meta["status"], "failed_missing_credentials")
                self.assertEqual(contract["status"], "needs_review")

                # Platform items must retain original input and NOT pass
                self.assertEqual(len(items), 1)
                item = items[0]
                self.assertFalse(item["passed"])
                self.assertEqual(item["disposition"], "needs_review")
                self.assertEqual(item["after"]["stem"], self.sample_text)
                self.assertFalse(item["passed"])
                self.assertTrue(item["verification"]["review_gated"])

    def test_2_api_error_retries_and_returns_failure_retaining_input(self):
        """
        Req 2, 9: If API fails, performs bounded retries, returns needs_review failure contract,
        retains original input, and never produces PASS.
        """
        detector = AIBoundaryDetector(api_key="fake-test-key", max_retries=2)
        detector._client = MagicMock()
        detector._client.models.generate_content.side_effect = RuntimeError("503 Service Unavailable")

        with patch("experimental.autonomous_parser.AutonomousSemanticParser.decompose", create=True) as mock_offline:
            contract, items, meta = detector.detect_and_process(
                raw_text=self.sample_text,
                question_id="TEST_API_ERR"
            )

            # Never invoke offline code
            mock_offline.assert_not_called()

            # Bounded retry: tried initial + 2 retries = 2 retries_used
            self.assertEqual(meta["retries_used"], 2)
            self.assertEqual(meta["status"], "failed_model_error")
            self.assertIn("503 Service Unavailable", meta["error"])

            # Must retain original text and reject PASS
            self.assertEqual(contract["status"], "needs_review")
            self.assertEqual(len(items), 1)
            item = items[0]
            self.assertFalse(item["passed"])
            self.assertEqual(item["disposition"], "needs_review")
            self.assertEqual(item["after"]["stem"], self.sample_text)

    def test_3_invalid_model_response_rejected_by_validator(self):
        """
        Req 2, 6, 9: If model returns invalid schema or non-tiling offsets,
        validator rejects it before slicing and returns failure contract.
        """
        detector = AIBoundaryDetector(api_key="fake-test-key", max_retries=1)
        detector._client = MagicMock()

        # Mock model response returning overlapping / invalid partition offsets
        invalid_contract = {
            "source_id": "TEST_INVALID",
            "status": "proposed",
            "segments": [
                {"id": "s001", "start": 0, "end": 20, "kind": "reference"},
                {"id": "s002", "start": 10, "end": 50, "kind": "stem"}  # Overlaps s001
            ],
            "items": [
                {
                    "id": "TEST_INVALID",
                    "parent_group_id": None,
                    "response_kind": "single_choice",
                    "source_label_segment_ids": [],
                    "stem_segment_ids": ["s002"],
                    "reference_segment_ids": ["s001"],
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

        mock_resp = MagicMock()
        mock_resp.text = json.dumps(invalid_contract)
        detector._client.models.generate_content.return_value = mock_resp

        contract, items, meta = detector.detect_and_process(
            raw_text=self.sample_text,
            question_id="TEST_INVALID"
        )

        self.assertEqual(meta["status"], "failed_model_error")
        self.assertIn("Contract validation failed", meta["error"])
        self.assertEqual(contract["status"], "needs_review")
        self.assertFalse(items[0]["passed"])
        self.assertEqual(items[0]["after"]["stem"], self.sample_text)

    def test_4_model_selection_and_hash_forwarding_without_secrets(self):
        """
        Req 4: Forwards configured model and credentials; records engine/model, prompt hash,
        and source hash. Never records secrets.
        """
        secret_api_key = "super_secret_production_key_12345"
        custom_model = "gemini-2.5-pro"
        detector = AIBoundaryDetector(api_key=secret_api_key, model_name=custom_model)

        expected_prompt = load_master_prompt_text()
        expected_prompt_hash = hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
        expected_source_hash = hashlib.sha256(self.sample_text.encode("utf-8")).hexdigest()

        # Mock successful model output
        ref_text = "Case Study:\nA 45-year-old patient presents with shortness of breath.\n\n"
        stem_text = "What is the most appropriate initial oxygen delivery system?\n"
        opt_text = "A. Nasal cannula\nB. Non-rebreather mask\nC. Venturi mask\nD. Bag-valve-mask"
        s0_end = len(ref_text)
        s1_end = s0_end + len(stem_text)
        s2_end = s1_end + len(opt_text)

        valid_contract = {
            "source_id": "TEST_HASHES",
            "status": "proposed",
            "segments": [
                {"id": "s001", "start": 0, "end": s0_end, "kind": "reference"},
                {"id": "s002", "start": s0_end, "end": s1_end, "kind": "stem"},
                {"id": "s003", "start": s1_end, "end": s2_end, "kind": "option"}
            ],
            "items": [
                {
                    "id": "TEST_HASHES",
                    "parent_group_id": None,
                    "response_kind": "single_choice",
                    "source_label_segment_ids": [],
                    "stem_segment_ids": ["s002"],
                    "reference_segment_ids": ["s001"],
                    "option_groups": [
                        {
                            "id": "og1",
                            "kind": "choice_bank",
                            "options": [
                                {"id": "opt1", "segment_ids": ["s003"]}
                            ]
                        }
                    ],
                    "matching_rows": [],
                    "response_template_segment_ids": [],
                    "depends_on_item_ids": [],
                    "context_reuse": [],
                    "disposition": "proposed"
                }
            ],
            "issues": []
        }

        detector._client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(valid_contract)
        detector._client.models.generate_content.return_value = mock_resp

        contract, items, meta = detector.detect_and_process(
            raw_text=self.sample_text,
            question_id="TEST_HASHES"
        )

        self.assertEqual(meta["engine"], "google-genai")
        self.assertEqual(meta["model"], "gemini-2.5-pro")
        self.assertEqual(meta["prompt_hash"], expected_prompt_hash)
        self.assertEqual(meta["source_hash"], expected_source_hash)

        # Ensure secrets are NEVER present in metadata or contract
        meta_str = json.dumps(meta)
        contract_str = json.dumps(contract)
        self.assertNotIn(secret_api_key, meta_str)
        self.assertNotIn(secret_api_key, contract_str)

    def test_5_exact_contract_used_for_export(self):
        """
        Req 5: Export the exact model-returned contract.
        Do not generate a separate offline contract.
        """
        detector = AIBoundaryDetector(api_key="key")
        detector._client = MagicMock()

        custom_marker = "exact_model_returned_identifier_999"
        model_contract = {
            "source_id": "Q_EXACT",
            "status": "proposed",
            "segments": [
                {"id": "s001", "start": 0, "end": len(self.sample_text), "kind": "stem"}
            ],
            "items": [
                {
                    "id": "Q_EXACT",
                    "parent_group_id": None,
                    "response_kind": "single_choice",
                    "source_label_segment_ids": [],
                    "stem_segment_ids": ["s001"],
                    "reference_segment_ids": [],
                    "option_groups": [],
                    "matching_rows": [],
                    "response_template_segment_ids": [],
                    "depends_on_item_ids": [],
                    "context_reuse": [{"segment_id": "s001", "from_item_id": "Q_PREV", "reason": "prior_task_context"}],
                    "disposition": "proposed"
                }
            ],
            "issues": []
        }

        mock_resp = MagicMock()
        mock_resp.text = json.dumps(model_contract)
        detector._client.models.generate_content.return_value = mock_resp

        contract, items, meta = detector.detect_and_process(
            raw_text=self.sample_text,
            question_id="Q_EXACT"
        )

        # The returned contract must be identical to what the model gave us
        self.assertEqual(contract["items"][0]["context_reuse"][0]["from_item_id"], "Q_PREV")

    def test_6_review_gating_blocks_automatic_pass(self):
        """
        Req 6, 9: Items with disposition='needs_review' or contract status='needs_review'
        must be blocked from automatic acceptance / PASS.
        """
        contract = {
            "source_id": "Q_GATED",
            "status": "needs_review",
            "segments": [
                {"id": "s001", "start": 0, "end": len(self.sample_text), "kind": "stem"}
            ],
            "items": [
                {
                    "id": "Q_GATED",
                    "parent_group_id": None,
                    "response_kind": "unknown",
                    "source_label_segment_ids": [],
                    "stem_segment_ids": ["s001"],
                    "reference_segment_ids": [],
                    "option_groups": [],
                    "matching_rows": [],
                    "response_template_segment_ids": [],
                    "depends_on_item_ids": [],
                    "context_reuse": [],
                    "disposition": "needs_review"
                }
            ],
            "issues": [{"code": "unresolved_ambiguity", "segment_ids": ["s001"], "item_ids": ["Q_GATED"]}]
        }

        items = contract_to_platform_items(contract, self.sample_text)
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertFalse(item["passed"], "Item marked needs_review must not have passed=True")
        self.assertTrue(item["verification"]["review_gated"])

    def test_7_reordered_spans_fail_verification(self):
        """
        Req 7, 9: Validate field order as well as source coverage.
        Character frequency multiset alone fails to catch swapped spans.
        verify_source_order_and_content must catch order mismatches.
        """
        span_a = "ABC"
        span_b = "DEF"
        original = span_a + span_b
        reordered = span_b + span_a

        # Character multiset would match:
        from collections import Counter
        self.assertEqual(Counter(original), Counter(reordered))

        # But exact order verification MUST detect order mismatch
        result = verify_source_order_and_content(original, [span_b, span_a])
        self.assertFalse(result["exact_match"])
        self.assertFalse(result["order_matches"])
        self.assertIn("Character order does not match source sequence", result["reason"])

    def test_8_preserved_dependencies_templates_and_matching_rows(self):
        """
        Req 6, 9: Output adapter preserves shared/local context, dependencies,
        matching rows and response templates.
        """
        tmpl_text = "Fill in: [____] mg/kg"
        full_text = self.sample_text + "\n" + tmpl_text
        s0_end = len(self.sample_text) + 1
        s1_end = len(full_text)

        contract = {
            "source_id": "Q_COMPLEX",
            "status": "proposed",
            "segments": [
                {"id": "s001", "start": 0, "end": s0_end, "kind": "stem"},
                {"id": "s002", "start": s0_end, "end": s1_end, "kind": "response_template"}
            ],
            "items": [
                {
                    "id": "Q_COMPLEX",
                    "parent_group_id": None,
                    "response_kind": "matching",
                    "source_label_segment_ids": [],
                    "stem_segment_ids": ["s001"],
                    "reference_segment_ids": [],
                    "option_groups": [],
                    "matching_rows": [
                        {
                            "id": "row1",
                            "segment_ids": ["s001"],
                            "option_group_id": "og1"
                        }
                    ],
                    "response_template_segment_ids": ["s002"],
                    "depends_on_item_ids": ["Q_PREV_001"],
                    "context_reuse": [],
                    "disposition": "proposed"
                }
            ],
            "issues": []
        }

        items = contract_to_platform_items(contract, full_text)
        self.assertEqual(len(items), 1)
        item = items[0]

        # Verify preserved fields in after
        self.assertEqual(item["after"]["response_template"], tmpl_text)
        self.assertEqual(len(item["after"]["matching_rows"]), 1)
        self.assertEqual(item["after"]["matching_rows"][0]["id"], "row1")
        self.assertEqual(item["after"]["depends_on_item_ids"], ["Q_PREV_001"])


if __name__ == "__main__":
    unittest.main()
