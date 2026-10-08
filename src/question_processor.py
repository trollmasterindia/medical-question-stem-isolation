"""
question_processor.py

Core processing logic for medical assessment stem isolation and decomposition.
Enforces the 'AI Eyes, Python Scissors' architecture:
- Boundary extraction is performed exclusively by genuine LLMs (Antigravity CLI or Direct Gemini API).
- Never substitutes the offline parser, regex boundary guesses, or benchmark answers during production runs.
- Python performs 100% of deterministic validation, field ordering checks, and verbatim text slicing.
- If model calls fail, authentication is missing, or schema validation fails, the item is marked as
  disposition='needs_review' with exact raw input preserved and zero data loss.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from ai_boundary_detector import AIBoundaryDetector
from antigravity_cli_provider import AntigravityCLIProvider
from contract_validator import ContractValidator
from contract_slicer import contract_to_platform_items
from prompt_loader import load_master_prompt_text, get_default_config


def process_question_item(
    raw_text: str,
    question_id: str = "custom",
    provider: str = "antigravity-cli",
    ai_detector: Optional[AIBoundaryDetector] = None,
    cli_provider: Optional[AntigravityCLIProvider] = None,
    domain: str = "Clinical Nursing",
    config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Unified production entry point to process a question item using a real LLM.
    Defaults to Antigravity CLI ('antigravity-cli').
    Also supports direct Gemini API ('gemini-api') if selected.
    Never falls back to regex or benchmark answers.
    Returns: (contract, platform_items, execution_metadata).
    """
    cfg = config or get_default_config()
    max_stem_chars = cfg.get("max_stem_chars", 500)

    if provider == "antigravity-cli":
        cli = cli_provider or AntigravityCLIProvider(model_name=model_name)
        sys_prompt = load_master_prompt_text()
        contract, metadata = cli.detect_boundaries(
            raw_text=raw_text,
            question_id=question_id,
            system_prompt=sys_prompt,
            config=cfg
        )
        is_valid, validation_errors = ContractValidator.validate(
            raw_text=raw_text,
            contract=contract,
            expected_source_id=question_id,
            config=cfg
        )
        metadata["validation"] = {
            "valid": is_valid,
            "errors": validation_errors
        }
        if not is_valid:
            metadata["status"] = "failed_validation"
            contract["status"] = "needs_review"
            for it in contract.get("items", []):
                it["disposition"] = "needs_review"

        platform_items = contract_to_platform_items(
            contract=contract,
            raw_text=raw_text,
            domain=domain,
            max_stem_chars=max_stem_chars
        )
        return contract, platform_items, metadata

    elif provider == "gemini-api":
        detector = ai_detector or AIBoundaryDetector(api_key=api_key, model_name=model_name or "gemini-2.5-flash")
        return detector.detect_and_process(
            raw_text=raw_text,
            question_id=question_id,
            domain=domain,
            config=cfg
        )

    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            "Must be 'antigravity-cli' (official Antigravity CLI) or 'gemini-api' (direct API)."
        )


def review_questions(
    inputs_dir: Optional[Path] = None,
    question_ids: Optional[List[str]] = None,
    provider: str = "antigravity-cli",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes the stem isolation workflow across target questions using genuine LLM extraction.
    Defaults to Antigravity CLI provider.
    Discovers and processes every requested input file.
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
        for qid in question_ids:
            files_to_process.append((qid, None))

    cli_provider = None
    ai_detector = None
    if provider == "antigravity-cli":
        cli_provider = AntigravityCLIProvider(model_name=model_name)
    elif provider == "gemini-api":
        ai_detector = AIBoundaryDetector(api_key=api_key, model_name=model_name or "gemini-2.5-flash")

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
        contract, processed_items, meta = process_question_item(
            raw_text=raw_text,
            question_id=qid,
            provider=provider,
            ai_detector=ai_detector,
            cli_provider=cli_provider,
            domain=domain,
            config=config,
            api_key=api_key,
            model_name=model_name
        )

        contracts.append(contract)
        results.extend(processed_items)
        execution_metas.append(meta)

    # Dynamic calculation of summary statistics from actual results
    total_items = len(results)
    passed_items = sum(
        1 for r in results
        if r.get("passed", False)
        and r.get("verification", {}).get("exact_match", False)
        and r.get("verification", {}).get("stem_under_500", False)
        and not r.get("verification", {}).get("review_gated", False)
    )
    preservation_rate = f"{(passed_items / total_items * 100):.1f}%" if total_items > 0 else "0.0%"
    all_under_500 = all(r["after"]["fits_500_char_limit"] for r in results) if results else False
    avg_stem = round(sum(r["after"]["stem_char_count"] for r in results) / total_items, 1) if total_items > 0 else 0
    split_screens = sum(1 for r in results if r["after"]["layout"] == "split_screen_with_reference")
    single_columns = sum(1 for r in results if r["after"]["layout"] == "single_column_stem_only")

    if provider == "antigravity-cli":
        method_desc = (
            f"Genuine Antigravity CLI Extraction (agy v{cli_provider.cli_version}) "
            f"with Verbatim Python Slicing. Model: {cli_provider.model_name or 'antigravity-default'}."
        )
    else:
        method_desc = (
            f"Direct Gemini API Extraction with Verbatim Python Slicing. "
            f"Engine: {ai_detector.model_name}."
        )

    payload = {
        "title": "Medical Assessment Stem Isolation - Reviewer Output",
        "method": method_desc,
        "provider": provider,
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
