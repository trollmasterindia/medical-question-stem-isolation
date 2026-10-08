#!/usr/bin/env python3
"""
review.py - CLI Reviewer for Medical Assessment Stem Isolation

Executes the stem isolation workflow across medical questions using the
'AI Eyes, Python Scissors' method:
- Python performs 100% of the string slicing directly on the raw text.
- Character multiset verification guarantees zero text alteration or omission.
- Admin constraint enforced: Stem <= 500 characters.
- Generates interactive Before vs After HTML comparison viewer.
- Exports verified machine-readable JSON.

Usage:
    python review.py                       # Review all benchmark questions
    python review.py --ids Q001 Q025 Q063  # Review specific questions
    python review.py --open                # Review and open HTML viewer in browser
"""

import sys
import argparse
import webbrowser
from pathlib import Path

# Add src to python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from question_processor import review_questions
from html_generator import generate_html_viewer


def main():
    parser = argparse.ArgumentParser(
        description="Review medical assessment questions, isolate stems <= 500 chars, and verify zero text loss."
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Specific question IDs to review (e.g. --ids Q001 Q025 Q063 Q080)"
    )
    parser.add_argument(
        "--splits",
        default=str(ROOT_DIR / "data" / "expected_splits.json"),
        help="Path to expected splits JSON file"
    )
    parser.add_argument(
        "--inputs",
        default=str(ROOT_DIR / "data" / "inputs"),
        help="Directory containing raw question text files"
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT_DIR / "data" / "review_output.json"),
        help="Path to save output JSON"
    )
    parser.add_argument(
        "--html",
        default=str(ROOT_DIR / "docs" / "index.html"),
        help="Path to save interactive HTML comparison viewer"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the HTML viewer in your default web browser"
    )

    args = parser.parse_args()

    splits_path = Path(args.splits)
    inputs_dir = Path(args.inputs)
    output_json_path = Path(args.output_json)
    html_path = Path(args.html)

    if not splits_path.exists():
        print(f"Error: Splits file not found at {splits_path}")
        sys.exit(1)
    if not inputs_dir.exists():
        print(f"Error: Inputs directory not found at {inputs_dir}")
        sys.exit(1)

    print("=" * 80)
    print(" MEDICAL ASSESSMENT STEM ISOLATION REVIEWER")
    print(" Architecture: AI Boundary Spans -> Python Verbatim Slicing (Zero Text Loss)")
    print("=" * 80)

    payload = review_questions(splits_path, inputs_dir, question_ids=args.ids)
    questions = payload["questions"]

    print(f"\nProcessing {len(questions)} platform items...\n")
    print(f"{'STATUS':<8} {'ID':<10} {'STEM (<=500c)':<15} {'REF CHARS':<12} {'LAYOUT':<28} {'VERIFICATION'}")
    print("-" * 88)

    all_passed = True
    for q in questions:
        v = q["verification"]
        after = q["after"]
        passed = v["exact_match"] and v["stem_under_500"]
        if not passed:
            all_passed = False

        status_icon = "✅ PASS" if passed else "❌ FAIL"
        stem_str = f"{after['stem_char_count']} chars"
        ref_str = f"{after['reference_char_count']} chars" if after['reference'] else "None (0c)"
        layout_str = after["layout"]
        verif_str = f"0 added, 0 removed" if v["exact_match"] else "TEXT MISMATCH"

        print(f"{status_icon:<8} {q['id']:<10} {stem_str:<15} {ref_str:<12} {layout_str:<28} {verif_str}")

    print("-" * 88)
    summary = payload["summary"]
    print(f"\nSummary:")
    print(f"  * Total Processed Items:       {summary['total_items']}")
    print(f"  * All Stems <= 500 chars:      {summary['all_stems_under_500_chars']}")
    print(f"  * Average Stem Length:         {summary['average_stem_length']} chars")
    print(f"  * Split-Screen Items:          {summary['split_screen_count']}")
    print(f"  * Single-Column Items:         {summary['single_column_count']}")
    print(f"  * Zero Text Alteration Rate:   {summary['exact_text_preservation_rate']}")

    # Save output JSON
    import json
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved JSON] {output_json_path}")

    # Also update before_vs_after_questions.json for backward compatibility
    compat_json = ROOT_DIR / "data" / "before_vs_after_questions.json"
    with open(compat_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Generate HTML comparison viewer
    generate_html_viewer(payload, html_path)
    
    # Also update data/stem_isolation_comparison.html for convenience
    generate_html_viewer(payload, ROOT_DIR / "data" / "stem_isolation_comparison.html")

    print(f"[Viewer Ready] file://{html_path.resolve()}")

    if args.open:
        print("Opening interactive viewer in browser...")
        webbrowser.open(f"file://{html_path.resolve()}")

    if not all_passed:
        print("\n❌ Warning: Some items failed verification or exceeded the 500-character limit!")
        sys.exit(1)
    else:
        print("\n✨ All items passed zero-text-loss and stem character limit constraints perfectly!")


if __name__ == "__main__":
    main()
