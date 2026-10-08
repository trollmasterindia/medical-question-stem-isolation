import unittest
import json
import os
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "src"))

from prompt_loader import load_system_instruction, load_json_schema_definition
from experimental.autonomous_parser import AutonomousSemanticParser


class TestStemIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        json_path = base_dir / 'data' / 'review_output.json'
        if not json_path.exists():
            json_path = base_dir / 'data' / 'before_vs_after_questions.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            cls.data = json.load(f)

    def test_total_items_count(self):
        """Test that all 46 platform items (27 base + 19 subquestions) are present."""
        self.assertEqual(len(self.data['questions']), 46)

    def test_all_questions_have_zero_alteration(self):
        """Test that zero characters were added or removed in any question (100% exact text match)."""
        for q in self.data['questions']:
            v = q['verification']
            self.assertEqual(v['characters_added'], 0, f"{q['id']} had added characters")
            self.assertEqual(v['characters_removed'], 0, f"{q['id']} had removed characters")
            self.assertTrue(v['exact_match'], f"{q['id']} failed exact character match")

    def test_all_isolated_stems_under_500_chars(self):
        """Test that every isolated stem complies with the database <= 500 character constraint."""
        for q in self.data['questions']:
            stem_len = q['after']['stem_char_count']
            self.assertLessEqual(stem_len, 500, f"{q['id']} stem exceeds 500 characters ({stem_len}c)")
            self.assertTrue(q['after']['fits_500_char_limit'])

    def test_q063_single_page_instruction_in_stem(self):
        """Test Q063 rule: test-taking instruction stays in stem since no reference exhibit exists."""
        q063 = next(q for q in self.data['questions'] if q['id'] == 'Q063')
        self.assertIsNone(q063['after']['reference'])
        self.assertEqual(q063['after']['layout'], 'single_column_stem_only')
        self.assertIn("Unless instructed otherwise", q063['after']['stem'])
        self.assertIn("distribution", q063['after']['stem'])
        self.assertEqual(q063['after']['stem_char_count'], 136)

    def test_q003_and_q095_instructions_in_reference(self):
        """Test Q003 & Q095 rule: instruction precedes scenario in Reference pane for split-screen items."""
        q003 = next(q for q in self.data['questions'] if q['id'] == 'Q003')
        self.assertEqual(q003['after']['layout'], 'split_screen_with_reference')
        self.assertIn("Consider the following scenario", q003['after']['reference'])
        self.assertIn("communication with Mr. Curtis", q003['after']['stem'])

        q095 = next(q for q in self.data['questions'] if q['id'] == 'Q095')
        self.assertEqual(q095['after']['layout'], 'split_screen_with_reference')
        self.assertIn("Use the following steps", q095['after']['reference'])

    def test_subquestions_decomposition(self):
        """Test that multi-part parent cases are decomposed into independent subquestions with clean stems."""
        sub_items = [q for q in self.data['questions'] if '-' in q['id']]
        self.assertEqual(len(sub_items), 19)

        q080_a = next(q for q in sub_items if q['id'] == 'Q080-a')
        self.assertLess(q080_a['after']['stem_char_count'], 100)
        self.assertIn("Cephalexin", q080_a['after']['reference'])

        q084_c = next(q for q in sub_items if q['id'] == 'Q084-c')
        self.assertEqual(q084_c['after']['stem'], "Is this dose within the safe guidelines?")
        self.assertIn("17 kilograms", q084_c['after']['reference'])

    def test_prompt_file_loaded_and_valid(self):
        """Test that extraction_prompt.md is loaded, contains system instructions and JSON schema."""
        sys_inst = load_system_instruction()
        self.assertTrue(len(sys_inst) > 500)
        self.assertTrue("semantic span annotator" in sys_inst or "medical assessment" in sys_inst)
        self.assertIn("500", sys_inst)

        schema = load_json_schema_definition()
        self.assertTrue(len(schema) > 50)
        self.assertTrue("source_id" in schema or "question_id" in schema)
        self.assertIn("segments", schema)

    def test_unseen_question_independent_processing(self):
        """Test that an unseen clinical question is parsed without existing in expected_splits.json."""
        raw = (
            "Case Study:\n"
            "A 65-year-old male with acute decompensated heart failure is admitted to the ICU. "
            "Vital signs: BP 85/50 mmHg, HR 112 bpm, SpO2 88% on room air. "
            "The physician orders Dobutamine 5 mcg/kg/min infusion.\n\n"
            "What is the primary hemodynamic mechanism of Dobutamine in this patient?\n"
            "A. Alpha-1 vasoconstriction\n"
            "B. Beta-1 inotropic stimulation\n"
            "C. Arterial dilation\n"
            "D. Parasympathetic blockade"
        )
        res = AutonomousSemanticParser.decompose(raw, question_id="NEW_UNSEEN_001")
        self.assertEqual(len(res), 1)
        item = res[0]
        self.assertEqual(item["id"], "NEW_UNSEEN_001")
        self.assertEqual(item["after"]["layout"], "split_screen_with_reference")
        self.assertIn("Dobutamine in this patient?", item["after"]["stem"])
        self.assertLessEqual(item["after"]["stem_char_count"], 500)
        self.assertIn("acute decompensated heart failure", item["after"]["reference"])
        self.assertEqual(len(item["after"]["distractors"]), 4)
        self.assertTrue(item["verification"]["exact_match"])
        self.assertEqual(item["verification"]["characters_added"], 0)
        self.assertEqual(item["verification"]["characters_removed"], 0)

    def test_unseen_multipart_independent_decomposition(self):
        """Test that an unseen multi-part question decomposes dynamically into independent subquestions."""
        raw = (
            "Pediatric Inpatient Dosing Case:\n"
            "A 6-year-old child weighing 20 kg is prescribed Cefazolin IV for cellulitis. "
            "The recommended daily dosage is 50 mg/kg/day divided into 3 equal doses every 8 hours.\n\n"
            "a. What is the total recommended daily dosage in mg for this child?\n\n"
            "b. How many mg should the nurse administer per dose?"
        )
        res = AutonomousSemanticParser.decompose(raw, question_id="NEW_MULTIPART_002")
        self.assertEqual(len(res), 2)
        sub_a = res[0]
        sub_b = res[1]
        self.assertEqual(sub_a["id"], "NEW_MULTIPART_002-a")
        self.assertEqual(sub_b["id"], "NEW_MULTIPART_002-b")
        self.assertEqual(sub_a["after"]["layout"], "split_screen_with_reference")
        self.assertEqual(sub_b["after"]["layout"], "split_screen_with_reference")
        self.assertIn("total recommended daily dosage", sub_a["after"]["stem"])
        self.assertIn("administer per dose", sub_b["after"]["stem"])
        self.assertIn("Cefazolin", sub_a["after"]["reference"])
        self.assertIn("Cefazolin", sub_b["after"]["reference"])
        self.assertTrue(sub_a["verification"]["exact_match"])
        self.assertTrue(sub_b["verification"]["exact_match"])

    def test_unseen_instruction_only_independent_processing(self):
        """Test that an unseen instruction-only question routes instruction to stem and renders single-column."""
        raw = (
            "Instructions: Unless instructed otherwise, choose ALL correct answers for this question.\n\n"
            "Which of the following are clinical manifestations of hypovolemic shock?\n"
            "A. Tachycardia\n"
            "B. Hypotension\n"
            "C. Bradycardia\n"
            "D. Oliguria"
        )
        res = AutonomousSemanticParser.decompose(raw, question_id="NEW_INST_ONLY_003")
        self.assertEqual(len(res), 1)
        item = res[0]
        self.assertEqual(item["after"]["layout"], "single_column_stem_only")
        self.assertIsNone(item["after"]["reference"])
        self.assertIn("Unless instructed otherwise", item["after"]["stem"])
        self.assertIn("manifestations of hypovolemic shock?", item["after"]["stem"])
        self.assertLessEqual(item["after"]["stem_char_count"], 500)
        self.assertTrue(item["verification"]["exact_match"])


if __name__ == '__main__':
    unittest.main()
