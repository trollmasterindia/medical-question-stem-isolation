import unittest
import json
import os
from pathlib import Path

class TestStemIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = Path(__file__).resolve().parent.parent
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


if __name__ == '__main__':
    unittest.main()
