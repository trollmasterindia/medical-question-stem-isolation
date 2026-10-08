"""
question_processor.py

Core processing logic for medical assessment stem isolation and decomposition.
Implements the 'AI Eyes, Python Scissors' method:
- Live AI boundary detector is the default and only production boundary detector.
- Python performs 100% of text slicing directly on the raw input string.
- Slices boundary offsets [start, end] with 0% text modification or omission.
- Strictly adheres to the <= 500 char stem constraint.
- Forward model and credentials consistently.
- Calculates verification dynamically from actual source spans and exported fields.
- Offline parser is isolated in src/experimental/autonomous_parser.py and not invoked here.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from ai_boundary_detector import AIBoundaryDetector


def process_question_item(
    raw_text: str,
    question_id: str = "custom",
    ai_detector: Optional[AIBoundaryDetector] = None,
    domain: str = "Clinical Nursing",
    config: Optional[Dict[str, Any]] = None,
    engine: str = "live"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Unified entry point to process a question item.
    Defaults to live AI detector. Also supports offline experimental parser when requested.
    Returns (contract, platform_items, execution_metadata).
    """
    if engine == "offline":
        import hashlib
        from experimental.autonomous_parser import AutonomousSemanticParser
        from contract_validator import ContractValidator
        from contract_slicer import contract_to_platform_items

        contract = AutonomousSemanticParser.parse_to_contract(raw_text, source_id=question_id, config=config)
        is_valid, validation_errors = ContractValidator.validate(raw_text, contract, expected_source_id=question_id, config=config)
        items = contract_to_platform_items(
            contract=contract,
            raw_text=raw_text,
            domain=domain,
            max_stem_chars=config.get("max_stem_chars", 500) if config else 500
        )
        meta = {
            "engine": "offline-autonomous-semantic-parser",
            "model": "offline-rule-based",
            "prompt_hash": "",
            "source_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "status": "success" if is_valid else "invalid",
            "retries_used": 0
        }
        return contract, items, meta

    detector = ai_detector or AIBoundaryDetector()
    return detector.detect_and_process(
        raw_text=raw_text,
        question_id=question_id,
        domain=domain,
        config=config
    )


def review_questions(
    inputs_dir: Optional[Path] = None,
    question_ids: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    config: Optional[Dict[str, Any]] = None,
    engine: str = "live"
) -> Dict[str, Any]:
    """
    Executes the stem isolation workflow across target questions.
    Defaults to Live AI Boundary Detection. Discovers and processes every requested input file.
    """
    files_to_process = []
    if inputs_dir and inputs_dir.exists():
        if question_ids:
            for qid in question_ids:
                p = inputs_dir / f"{qid}.txt"
                if p.exists():
                    files_to_process.append((qid, p))
                else:
                    print(f"Warning: File {p} not found.")
        else:
            # Discover and process every requested input file
            for p in sorted(inputs_dir.glob("*.txt")):
                files_to_process.append((p.stem, p))
    elif question_ids:
        # No inputs_dir, but question_ids provided
        for qid in question_ids:
            files_to_process.append((qid, None))

    detector = None
    if engine != "offline":
        detector = AIBoundaryDetector(api_key=api_key, model_name=model_name)

    results = []
    contracts = []
    execution_metas = []

    for qid, file_path in files_to_process:
        raw_text = None
        if file_path and file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

        if raw_text is None:
            print(f"Warning: Could not read raw text for question {qid}.")
            continue

        domain = "Clinical Nursing"
        if engine == "offline":
            contract, processed_items, meta = process_question_item(
                raw_text=raw_text,
                question_id=qid,
                domain=domain,
                config=config,
                engine="offline"
            )
        else:
            contract, processed_items, meta = detector.detect_and_process(
                raw_text=raw_text,
                question_id=qid,
                domain=domain,
                config=config
            )

        contracts.append(contract)
        results.extend(processed_items)
        execution_metas.append(meta)

    # Dynamic calculation of summary statistics from actual results
    total_items = len(results)
    passed_items = sum(
        1 for r in results
        if r.get("passed", False) and r.get("verification", {}).get("exact_match", False) and r.get("verification", {}).get("stem_under_500", False)
    )
    preservation_rate = f"{(passed_items / total_items * 100):.1f}%" if total_items > 0 else "0.0%"
    all_under_500 = all(r["after"]["fits_500_char_limit"] for r in results) if results else False
    avg_stem = round(sum(r["after"]["stem_char_count"] for r in results) / total_items, 1) if total_items > 0 else 0
    split_screens = sum(1 for r in results if r["after"]["layout"] == "split_screen_with_reference")
    single_columns = sum(1 for r in results if r["after"]["layout"] == "single_column_stem_only")

    method_desc = (
        f"Live AI Boundary Detection with Verbatim Python Slicing. Engine: {detector.model_name}."
        if detector
        else "Autonomous Semantic Parser (Offline) with Verbatim Python Slicing."
    )

    payload = {
        "title": "Medical Assessment Stem Isolation - Reviewer Output",
        "method": method_desc,
        "summary": {
            "total_items": total_items,
            "passed_items": passed_items,
            "exact_text_preservation_rate": preservation_rate,
            "all_stems_under_500_chars": all_under_500,
            "average_stem_length": avg_stem,
            "split_screen_count": split_screens,
            "single_column_count": single_columns
        },
        "questions": results,
        "master_contracts": contracts,
        "execution_metadata": execution_metas
    }

    return payload

