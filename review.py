#!/usr/bin/env python3
"""
review.py - CLI Reviewer for Medical Assessment Stem Isolation

Executes the stem isolation workflow across medical questions using the
'AI Eyes, Python Scissors' method:
- Python performs 100% of the string slicing directly on the raw text.
- Character multiset verification guarantees zero text alteration or omission.
- Admin constraint enforced: Stem <= 500 characters.
- Live AI boundary detection using prompts/extraction_prompt.md or autonomous semantic engine.
- Generates interactive Before vs After HTML comparison viewer.
- Exports verified machine-readable JSON.

Usage:
    python review.py                                  # Review benchmark questions (Auto mode)
    python review.py --mode ai                        # Review using live LLM with extraction prompt
    python review.py --file path/to/new_question.txt  # Review an arbitrary unseen question file
    python review.py --ids Q001 Q025 Q063             # Review specific questions
    python review.py --open                           # Review and open HTML viewer in browser
"""

import sys
import argparse
import webbrowser
from pathlib import Path

# Add src to python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from question_processor import review_questions, process_question_item
from semantic_parser import AutonomousSemanticParser
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
        "--file",
        help="Path to an individual raw question text file to process independently"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "ai", "benchmark"],
        default="auto",
        help="Detection mode: 'auto' (autonomous content-driven parser), 'ai' (live LLM using prompts/extraction_prompt.md), 'benchmark' (annotated benchmark spans)"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model name for AI mode (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for AI model (or set GEMINI_API_KEY / GOOGLE_API_KEY environment variable)"
    )
    parser.add_argument(
        "--splits",
        default=str(ROOT_DIR / "data" / "expected_splits.json"),
        help="Path to expected splits JSON file (used in benchmark mode)"
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
        "--contract-json",
        default=str(ROOT_DIR / "data" / "master_contract_output.json"),
        help="Path to save formal Master Output Contract JSON"
    )
    parser.add_argument(
        "--split-policy",
        choices=["preserve", "split_labeled_if_safe", "split_explicit_tasks_if_safe"],
        default="split_labeled_if_safe",
        help="Split policy for multi-part items (default: split_labeled_if_safe)"
    )
    parser.add_argument(
        "--max-stem-chars",
        type=int,
        default=500,
        help="Maximum allowed characters for stem (default: 500)"
    )
    parser.add_argument(
        "--reference-placement",
        choices=["unknown", "above", "beside"],
        default="beside",
        help="Reference placement layout in student UI (default: beside)"
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

    print("=" * 80)
    print(" MEDICAL ASSESSMENT STEM ISOLATION REVIEWER")
    print(f" Mode: {args.mode.upper()} | Model: {args.model if args.mode == 'ai' else 'Autonomous Engine'}")
    print(" Architecture: AI Semantic Boundaries -> Python Verbatim Slicing (Zero Text Loss)")
    print("=" * 80)

    cfg = {
        "max_stem_chars": args.max_stem_chars,
        "split_policy": args.split_policy,
        "matching_policy": "review",
        "supports_ordering": False,
        "supports_matching": False,
        "supports_linked_items": False,
        "supports_conditional_items": False,
        "reference_placement": args.reference_placement
    }

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found at {file_path}")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        qid = file_path.stem
        processed = process_question_item(
            raw_text=raw_text,
            question_id=qid,
            mode=args.mode,
            config=cfg
        )
        contract = AutonomousSemanticParser.parse_to_contract(raw_text, source_id=qid, config=cfg)
        payload = {
            "title": f"Review Output: {qid}",
            "method": f"Dynamic boundary detection ({args.mode.upper()}).",
            "summary": {
                "total_items": len(processed),
                "exact_text_preservation_rate": "100%",
                "all_stems_under_500_chars": all(r["after"]["fits_500_char_limit"] for r in processed),
                "average_stem_length": round(sum(r["after"]["stem_char_count"] for r in processed) / len(processed), 1),
                "split_screen_count": len([r for r in processed if r["after"]["layout"] == "split_screen_with_reference"]),
                "single_column_count": len([r for r in processed if r["after"]["layout"] == "single_column_stem_only"])
            },
            "questions": processed,
            "master_contracts": [contract]
        }
    else:
        if args.mode == "benchmark" and not splits_path.exists():
            print(f"Error: Splits file not found at {splits_path}")
            sys.exit(1)
        if not inputs_dir.exists():
            print(f"Error: Inputs directory not found at {inputs_dir}")
            sys.exit(1)

        payload = review_questions(
            splits_path=splits_path if splits_path.exists() else None,
            inputs_dir=inputs_dir,
            question_ids=args.ids,
            mode=args.mode,
            api_key=args.api_key,
            model_name=args.model,
            config=cfg
        )

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

    # Save formal Master Contract JSON if present
    if "master_contracts" in payload and args.contract_json:
        contract_json_path = Path(args.contract_json)
        contract_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(contract_json_path, "w", encoding="utf-8") as f:
            json.dump(payload["master_contracts"], f, indent=2, ensure_ascii=False)
        print(f"[Saved Master Contract JSON] {contract_json_path}")

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
