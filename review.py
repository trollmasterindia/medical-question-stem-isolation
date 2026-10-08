#!/usr/bin/env python3
"""
review.py - CLI Reviewer for Medical Assessment Stem Isolation

Executes the stem isolation workflow across medical questions using the
'AI Eyes, Python Scissors' method:
- Boundary extraction is performed exclusively by genuine LLMs:
    * Antigravity CLI ('antigravity-cli', default): uses developer Antigravity subscription and headless structured-output.
    * Direct Gemini API ('gemini-api', optional): uses google.genai API client with secure env key.
- Never substitutes the offline parser, regex boundary guesses, or benchmark answers during production runs.
- Python performs 100% of deterministic validation, field ordering checks, and verbatim text slicing.
- Character-order and content verification guarantees zero text alteration or omission.
- Admin constraint enforced: Stem <= 500 characters.
- Exports exact model-returned contracts to data/master_contract_output.json.
- Generates interactive Before vs After HTML comparison viewer.
- Exports verified machine-readable JSON.

Usage:
    python review.py                                  # Review all questions using official Antigravity CLI
    python review.py --file path/to/new_question.txt  # Review an individual unseen question file
    python review.py --ids Q001 Q025 Q063             # Review specific questions
    python review.py --open                           # Review and open HTML viewer in browser
    python review.py --provider gemini-api            # Run via direct Gemini API (requires GEMINI_API_KEY)
"""

import sys
import os
import json
import argparse
import webbrowser
from pathlib import Path

# Add src to python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from question_processor import review_questions, process_question_item
from html_generator import generate_html_viewer
from antigravity_cli_provider import AntigravityCLIProvider


def main():
    parser = argparse.ArgumentParser(
        description="Review medical assessment questions, isolate stems <= 500 chars, and verify zero text loss."
    )
    parser.add_argument(
        "--provider",
        choices=["antigravity-cli", "gemini-api"],
        default="antigravity-cli",
        help="LLM provider: 'antigravity-cli' (official Antigravity CLI subscription) or 'gemini-api' (direct API)"
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
        "--model",
        help="Optional model override for the selected provider"
    )
    parser.add_argument(
        "--api-key",
        help="API key for direct Gemini API (or set GEMINI_API_KEY / GOOGLE_API_KEY environment variable)"
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

    # Guard: Direct Gemini API key requirement
    if args.provider == "gemini-api":
        has_key = bool(args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        if not has_key:
            print("\n" + "=" * 80)
            print("❌ GEMINI API KEY REQUIRED")
            print("=" * 80)
            print("Direct Gemini API provider ('--provider gemini-api') requires an API key.")
            print("To configure your key securely in your environment, run:")
            print("   export GEMINI_API_KEY=\"your_api_key_here\"")
            print("\nSecurity notice: Never paste your API key into ordinary chat, print it, or commit it.")
            print("Alternatively, use the default Antigravity CLI provider:")
            print("   python review.py --provider antigravity-cli")
            print("=" * 80 + "\n")
            sys.exit(1)

    inputs_dir = Path(args.inputs)
    output_json_path = Path(args.output_json)
    html_path = Path(args.html)

    print("=" * 80)
    print(" MEDICAL ASSESSMENT STEM ISOLATION REVIEWER")
    if args.provider == "antigravity-cli":
        cli_temp = AntigravityCLIProvider()
        print(f" Engine: Antigravity CLI ('agy' v{cli_temp.cli_version}) | Model: {args.model or 'antigravity-default'}")
        print(" Authentication: Antigravity Subscription Native Session")
    else:
        print(f" Engine: Direct Gemini API | Model: {args.model or 'gemini-2.5-flash'}")
        print(" Authentication: GEMINI_API_KEY Environment Variable")
    print(" Architecture: Genuine LLM Contract -> Deterministic Python Verbatim Slicing")
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
        contract, processed, meta = process_question_item(
            raw_text=raw_text,
            question_id=qid,
            provider=args.provider,
            domain="Clinical Nursing",
            config=cfg,
            api_key=args.api_key,
            model_name=args.model
        )
        total_items = len(processed)
        passed_items = sum(
            1 for r in processed
            if r.get("passed", False)
            and r.get("verification", {}).get("exact_match", False)
            and r.get("verification", {}).get("stem_under_500", False)
            and not r.get("verification", {}).get("review_gated", False)
        )
        preservation_rate = f"{(passed_items / total_items * 100):.1f}%" if total_items > 0 else "0.0%"
        payload = {
            "title": f"Review Output: {qid}",
            "method": f"LLM Boundary Detection ({args.provider}).",
            "provider": args.provider,
            "summary": {
                "total_items": total_items,
                "passed_items": passed_items,
                "exact_text_preservation_rate": preservation_rate,
                "all_stems_under_500_chars": all(r["after"]["fits_500_char_limit"] for r in processed) if processed else False,
                "average_stem_length": round(sum(r["after"]["stem_char_count"] for r in processed) / total_items, 1) if total_items > 0 else 0,
                "split_screen_count": len([r for r in processed if r["after"]["layout"] == "split_screen_with_reference"]),
                "single_column_count": len([r for r in processed if r["after"]["layout"] == "single_column_stem_only"])
            },
            "questions": processed,
            "master_contracts": [contract],
            "execution_metadata": [meta]
        }
    else:
        if not inputs_dir.exists():
            print(f"Error: Inputs directory not found at {inputs_dir}")
            sys.exit(1)

        payload = review_questions(
            inputs_dir=inputs_dir,
            question_ids=args.ids,
            provider=args.provider,
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
        v = q.get("verification", {})
        after = q["after"]
        passed = (
            q.get("passed", False)
            and v.get("exact_match", False)
            and v.get("stem_under_500", False)
            and not v.get("review_gated", False)
        )
        if not passed:
            all_passed = False

        status_icon = "✅ PASS" if passed else "❌ REVIEW"
        stem_str = f"{after['stem_char_count']} chars"
        ref_str = f"{after['reference_char_count']} chars" if after['reference'] else "None (0c)"
        layout_str = after["layout"]
        if passed:
            verif_str = "0 added, 0 removed (order verified)"
        else:
            verif_str = v.get("failure_reason") or ("REVIEW GATED" if v.get("review_gated") else "NEEDS REVIEW")

        print(f"{status_icon:<8} {q['id']:<10} {stem_str:<15} {ref_str:<12} {layout_str:<28} {verif_str}")

    print("-" * 88)
    summary = payload["summary"]
    print(f"\nSummary:")
    print(f"  * Provider:                    {args.provider}")
    print(f"  * Total Processed Items:       {summary['total_items']}")
    print(f"  * Verified Passed Items:       {summary.get('passed_items', 0)}")
    print(f"  * All Stems <= 500 chars:      {summary['all_stems_under_500_chars']}")
    print(f"  * Average Stem Length:         {summary['average_stem_length']} chars")
    print(f"  * Split-Screen Items:          {summary['split_screen_count']}")
    print(f"  * Single-Column Items:         {summary['single_column_count']}")
    print(f"  * Exact Preservation Rate:     {summary['exact_text_preservation_rate']}")

    # Save output JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved JSON] {output_json_path}")

    # Also update before_vs_after_questions.json for backward compatibility
    compat_json = ROOT_DIR / "data" / "before_vs_after_questions.json"
    with open(compat_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Save exact formal Master Contract JSON returned by model
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
        print("\n⚠️ Notice: Items with status 'needs_review' or validation errors require inspection.")
    else:
        print("\n✨ All processed items passed exact source preservation and stem character constraints!")


if __name__ == "__main__":
    main()
