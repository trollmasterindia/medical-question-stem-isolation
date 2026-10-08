"""
ai_boundary_detector.py

AI Boundary Detector Engine for Medical Assessment Items.
Loads master system instructions from prompts/medical_stem_isolation_system_prompt.txt
and coordinates live LLM boundary extraction with strict Python contract validation.

Implements Sections 1 & 2 of Generic Stem Isolation Implementation Notes:
- Passes immutable raw_text, source_id, and config.
- Enforces strict Output Contract schema validation.
- Slices verbatim via Python string scissors with 0% text modification.
- Graceful offline fallback to AutonomousSemanticParser.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from prompt_loader import load_system_instruction, get_default_config
from contract_validator import ContractValidator
from contract_slicer import contract_to_platform_items
from semantic_parser import AutonomousSemanticParser

logger = logging.getLogger(__name__)


class AIBoundaryDetector:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        allow_fallback: bool = True
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self.system_instruction = load_system_instruction()
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
    ) -> List[Dict[str, Any]]:
        """
        Extracts boundaries and decomposes raw question into platform items.
        Calls the configured AI model when credentials exist, or uses the
        autonomous semantic engine.
        """
        cfg = config or get_default_config()

        if self._client:
            try:
                contract = self._call_ai_model(raw_text, question_id, cfg)
                is_valid, errors = ContractValidator.validate(raw_text, contract, question_id, cfg)
                if is_valid:
                    return contract_to_platform_items(contract, raw_text, domain=domain)
                else:
                    print(f"[AI Model Validation Warning for {question_id}]: {errors}. Falling back to autonomous parser.")
            except Exception as e:
                print(f"[AI Model Note] Calling {self.model_name} failed ({e}). Falling back to autonomous parser.")
                if not self.allow_fallback:
                    raise e

        # Offline / Autonomous semantic parsing
        return AutonomousSemanticParser.decompose(
            raw_text=raw_text,
            question_id=question_id,
            domain=domain,
            config=cfg
        )

    def _call_ai_model(
        self,
        raw_text: str,
        question_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sends the question text to the LLM with the loaded extraction prompt.
        Validates returned boundaries against verbatim text slices.
        """
        request_payload = {
            "source_id": question_id,
            "raw_text": raw_text,
            "config": config
        }

        user_content = (
            f"Please analyze this medical assessment item and return the Output Contract JSON.\n\n"
            f"```json\n{json.dumps(request_payload, indent=2, ensure_ascii=False)}\n```"
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_content,
            config={
                "system_instruction": self.system_instruction,
                "response_mime_type": "application/json"
            }
        )

        return json.loads(response.text)
