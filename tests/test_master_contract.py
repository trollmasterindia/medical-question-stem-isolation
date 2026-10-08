"""
test_master_contract.py

Comprehensive test suite verifying the Generic Stem Isolation Master Contract:
- Gap-free, non-overlapping canonical partition tiling 0..len(raw_text).
- Deterministic Python slicing and exact character reconstruction.
- Strict ContractValidator checks (rejecting booleans, overlaps, dangling IDs).
- Empty input handling ('empty_input' issue).
- Stem limit risk detection and controlled issue codes.
"""

import unittest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from prompt_loader import load_master_prompt_text, get_default_config
from contract_validator import ContractValidator
from semantic_parser import AutonomousSemanticParser, build_gap_free_partition
from contract_slicer import contract_to_platform_items


class TestMasterContract(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "Case Study 1:\n"
            "A 65-year-old male with acute heart failure is admitted to the ICU. "
            "Vital signs: BP 85/50 mmHg, HR 112 bpm, SpO2 88% on room air.\n\n"
            "What is the primary hemodynamic mechanism of Dobutamine in this patient?\n"
            "A. Alpha-1 vasoconstriction\n"
            "B. Beta-1 inotropic stimulation\n"
            "C. Arterial dilation\n"
            "D. Parasympathetic blockade"
        )
        self.config = get_default_config()

    def test_master_prompt_loaded(self):
        """Verify that medical_stem_isolation_system_prompt.txt loads and has full instructions."""
        text = load_master_prompt_text()
        self.assertIn("You are a semantic span annotator", text)
        self.assertIn("OUTPUT CONTRACT", text)
        self.assertIn("Controlled issue codes:", text)
        self.assertIn("max_stem_chars", text)

    def test_gap_free_partition_coverage(self):
        """Verify that segments tile the source with no gaps or overlaps."""
        contract = AutonomousSemanticParser.parse_to_contract(self.sample_text, "TEST_001", self.config)
        segments = contract["segments"]

        self.assertGreater(len(segments), 0)
        self.assertEqual(segments[0]["start"], 0)
        self.assertEqual(segments[-1]["end"], len(self.sample_text))

        for i in range(len(segments) - 1):
            self.assertEqual(
                segments[i]["end"], segments[i + 1]["start"],
                f"Gap or overlap between segment {segments[i]['id']} and {segments[i+1]['id']}"
            )

    def test_exact_source_reconstruction_from_slices(self):
        """Verify that concatenating all partition slices reconstructs raw_text 100% exactly."""
        contract = AutonomousSemanticParser.parse_to_contract(self.sample_text, "TEST_001", self.config)
        reconstructed = "".join(self.sample_text[s["start"]:s["end"]] for s in contract["segments"])
        self.assertEqual(reconstructed, self.sample_text)

    def test_validator_passes_valid_contract(self):
        """Verify that valid contract passes ContractValidator with 0 errors."""
        contract = AutonomousSemanticParser.parse_to_contract(self.sample_text, "TEST_001", self.config)
        is_valid, errors = ContractValidator.validate(self.sample_text, contract, "TEST_001", self.config)
        self.assertTrue(is_valid, f"Validation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_validator_rejects_boolean_offsets(self):
        """Verify that boolean start/end offsets (which inherit from int) are strictly rejected."""
        contract = AutonomousSemanticParser.parse_to_contract(self.sample_text, "TEST_001", self.config)
        contract["segments"][0]["start"] = False  # boolean
        is_valid, errors = ContractValidator.validate(self.sample_text, contract, "TEST_001", self.config)
        self.assertFalse(is_valid)
        self.assertTrue(any("must be integer" in err for err in errors))

    def test_validator_rejects_overlapping_segments(self):
        """Verify that overlapping segments fail validation."""
        contract = AutonomousSemanticParser.parse_to_contract(self.sample_text, "TEST_001", self.config)
        contract["segments"][1]["start"] = contract["segments"][0]["end"] - 2
        is_valid, errors = ContractValidator.validate(self.sample_text, contract, "TEST_001", self.config)
        self.assertFalse(is_valid)
        self.assertTrue(any("Gap or overlap" in err for err in errors))

    def test_empty_input_handling(self):
        """Verify that empty input produces empty partition, no items, and 'empty_input' issue."""
        contract = AutonomousSemanticParser.parse_to_contract("", "EMPTY_001", self.config)
        self.assertEqual(contract["status"], "needs_review")
        self.assertEqual(contract["segments"], [])
        self.assertEqual(contract["items"], [])
        self.assertTrue(any(iss["code"] == "empty_input" for iss in contract["issues"]))

        is_valid, errors = ContractValidator.validate("", contract, "EMPTY_001", self.config)
        self.assertTrue(is_valid, f"Empty input validation failed: {errors}")

    def test_stem_limit_risk_detection(self):
        """Verify that a stem exceeding max_stem_chars is flagged with stem_limit_risk."""
        long_stem_question = (
            "What is the best initial nursing action for a critically ill patient presenting with "
            "profound septic shock, refractory hypotension despite aggressive crystalloid fluid "
            "resuscitation of thirty milliliters per kilogram, elevated serum lactate above four millimoles "
            "per liter, acute respiratory failure requiring invasive endotracheal mechanical ventilation, "
            "progressive oliguria with urine output less than zero point five milliliters per kilogram per hour, "
            "and evidence of multiorgan dysfunction syndrome?"
        )
        tight_config = dict(self.config, max_stem_chars=50)
        contract = AutonomousSemanticParser.parse_to_contract(long_stem_question, "LONG_STEM", tight_config)
        self.assertEqual(contract["status"], "needs_review")
        self.assertTrue(any(iss["code"] == "stem_limit_risk" for iss in contract["issues"]))

    def test_multipart_contract_and_slicing(self):
        """Verify multi-part items produce valid parent group and independent children."""
        multipart_raw = (
            "Pediatric Dosing Case:\n"
            "A 4-year-old child weighing 16 kg is prescribed Amoxicillin suspension for otitis media. "
            "The recommended dosage is 80 to 90 mg/kg/day divided into two doses.\n\n"
            "a. What is the minimum recommended dose in mg per dose for this child?\n\n"
            "b. What is the maximum recommended dose in mg per dose for this child?"
        )
        contract = AutonomousSemanticParser.parse_to_contract(multipart_raw, "MULTI_001", self.config)
        is_valid, errors = ContractValidator.validate(multipart_raw, contract, "MULTI_001", self.config)
        self.assertTrue(is_valid, f"Multi-part contract validation failed: {errors}")

        self.assertEqual(len(contract["items"]), 2)
        self.assertEqual(contract["items"][0]["id"], "MULTI_001-a")
        self.assertEqual(contract["items"][1]["id"], "MULTI_001-b")
        self.assertEqual(contract["items"][0]["parent_group_id"], "group_MULTI_001")

        # Slice to platform items
        platform_items = contract_to_platform_items(contract, multipart_raw)
        self.assertEqual(len(platform_items), 2)
        self.assertTrue(platform_items[0]["verification"]["exact_match"])
        self.assertTrue(platform_items[1]["verification"]["exact_match"])
        self.assertEqual(platform_items[0]["after"]["layout"], "split_screen_with_reference")


if __name__ == '__main__':
    unittest.main()
