"""
ai_boundary_detector.py

Production AI Boundary Detector for Medical Assessment Items.
Enforces that live AI is the default and only production boundary engine.
- Loads prompts/medical_stem_isolation_system_prompt.txt.
- Sends only immutable raw_text, opaque source_id, and configuration.
- Records model name, prompt hash, and source hash; never records secrets.
- Performs bounded retries on model failure/invalid contract.
- If credentials are missing, API fails, or validation fails:
  returns a clear failure/needs_review result, retaining the original input.
- Never falls back to regex parsing or benchmark answers.
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from prompt_loader import load_master_prompt_text, get_default_config
from contract_validator import ContractValidator
from contract_slicer import contract_to_platform_items

logger = logging.getLogger(__name__)


class AIBoundaryDetector:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 2
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name
        self.max_retries = max_retries
        self.system_instruction = load_master_prompt_text()
        self.prompt_hash = hashlib.sha256(self.system_instruction.encode("utf-8")).hexdigest()
        self._client = None

        if self.api_key:
            self._init_client()

    def _init_client(self):
        """Initializes the google.genai Client."""
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.warning(f"Could not initialize google.genai Client: {e}")
            self._client = None

    def detect_and_process(
        self,
        raw_text: str,
        question_id: str = "custom",
        domain: str = "Clinical Nursing",
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes live model boundary detection with strict validation.
        Returns: (contract, platform_items, execution_metadata).
        If credentials missing or API/validation fails, returns failure/needs_review
        result retaining the original input.
        """
        cfg = config or get_default_config()
        source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        metadata = {
            "engine": "google-genai",
            "model": self.model_name,
            "prompt_hash": self.prompt_hash,
            "source_hash": source_hash,
            "status": "pending",
            "retries_used": 0
        }

        # 1. Missing credentials check
        if not self._client or not self.api_key:
            err_msg = "Missing API credentials: GEMINI_API_KEY (or GOOGLE_API_KEY) not provided."
            metadata["status"] = "failed_missing_credentials"
            metadata["error"] = err_msg
            contract = self._create_failure_contract(
                raw_text=raw_text,
                source_id=question_id,
                issue_code="missing_context"
            )
            items = self._create_failure_platform_items(
                raw_text=raw_text,
                question_id=question_id,
                domain=domain,
                error_msg=err_msg
            )
            return contract, items, metadata

        # 2. Live model execution with bounded retries
        last_error = ""
        correction_feedback = ""

        for attempt in range(self.max_retries + 1):
            metadata["retries_used"] = attempt
            try:
                raw_response_text = self._execute_model_call(raw_text, question_id, cfg, correction_feedback)
                contract_json = json.loads(raw_response_text)

                # Strict validation before slicing
                is_valid, validation_errors = ContractValidator.validate(
                    raw_text=raw_text,
                    contract=contract_json,
                    expected_source_id=question_id,
                    config=cfg
                )

                if is_valid:
                    metadata["status"] = "success"
                    # Slices real output dynamically
                    platform_items = contract_to_platform_items(
                        contract=contract_json,
                        raw_text=raw_text,
                        domain=domain,
                        max_stem_chars=cfg.get("max_stem_chars", 500)
                    )
                    return contract_json, platform_items, metadata
                else:
                    last_error = f"Contract validation failed: {validation_errors}"
                    correction_feedback = f"Previous response was invalid: {validation_errors}. Return a strictly valid contract."

            except Exception as e:
                last_error = f"Model execution error: {str(e)}"
                correction_feedback = f"Previous call failed with error: {str(e)}. Return valid JSON matching schema."

        # Retries exhausted: return failure contract and retain original input
        metadata["status"] = "failed_model_error"
        metadata["error"] = last_error

        failure_contract = self._create_failure_contract(
            raw_text=raw_text,
            source_id=question_id,
            issue_code="unsupported_representation"
        )
        failure_items = self._create_failure_platform_items(
            raw_text=raw_text,
            question_id=question_id,
            domain=domain,
            error_msg=last_error
        )
        return failure_contract, failure_items, metadata

    def _execute_model_call(
        self,
        raw_text: str,
        question_id: str,
        config: Dict[str, Any],
        correction_feedback: str = ""
    ) -> str:
        """Calls the live model passing only raw_text, source_id, and config."""
        request_data = {
            "source_id": question_id,
            "raw_text": raw_text,
            "config": config
        }
        user_prompt = f"```json\n{json.dumps(request_data, indent=2, ensure_ascii=False)}\n```"
        if correction_feedback:
            user_prompt += f"\n\nCorrection instruction: {correction_feedback}"

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config={
                "system_instruction": self.system_instruction,
                "response_mime_type": "application/json"
            }
        )
        return response.text

    def _create_failure_contract(
        self,
        raw_text: str,
        source_id: str,
        issue_code: str
    ) -> Dict[str, Any]:
        """Creates an intact fallback contract retaining the original source for review."""
        if len(raw_text) == 0:
            return {
                "source_id": source_id,
                "status": "needs_review",
                "segments": [],
                "items": [],
                "issues": [{"code": "empty_input", "segment_ids": [], "item_ids": []}]
            }

        return {
            "source_id": source_id,
            "status": "needs_review",
            "segments": [
                {
                    "id": "s001",
                    "start": 0,
                    "end": len(raw_text),
                    "kind": "unresolved"
                }
            ],
            "items": [
                {
                    "id": source_id,
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
            "issues": [
                {
                    "code": issue_code,
                    "segment_ids": ["s001"],
                    "item_ids": [source_id]
                }
            ]
        }

    def _create_failure_platform_items(
        self,
        raw_text: str,
        question_id: str,
        domain: str,
        error_msg: str
    ) -> List[Dict[str, Any]]:
        """Creates platform item retaining the original input with disposition='needs_review'."""
        return [
            {
                "id": question_id,
                "parent_id": question_id,
                "domain": domain,
                "response_kind": "unknown",
                "disposition": "needs_review",
                "passed": False,
                "error": error_msg,
                "before": {
                    "stem": raw_text,
                    "reference": None,
                    "distractors": [],
                    "char_count": len(raw_text),
                    "fits_500_char_limit": len(raw_text) <= 500,
                    "layout": "single_column_stem_only"
                },
                "after": {
                    "stem": raw_text,
                    "reference": None,
                    "distractors": [],
                    "metadata": None,
                    "response_template": None,
                    "matching_rows": [],
                    "option_groups": [],
                    "depends_on_item_ids": [],
                    "context_reuse": [],
                    "stem_char_count": len(raw_text),
                    "reference_char_count": 0,
                    "fits_500_char_limit": len(raw_text) <= 500,
                    "layout": "single_column_stem_only"
                },
                "verification": {
                    "characters_added": 0,
                    "characters_removed": 0,
                    "exact_match": False,
                    "order_matches": False,
                    "stem_under_500": len(raw_text) <= 500,
                    "review_gated": True,
                    "failure_reason": error_msg
                }
            }
        ]
