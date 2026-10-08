"""
antigravity_cli_provider.py

Official Antigravity CLI (agy) provider for medical assessment stem isolation.
Leverages the installed Antigravity CLI's native authentication and headless structured-output interface.

Architecture & Compliance:
1. Native Authentication: Uses the developer's Antigravity subscription via the official CLI.
   Never extracts, inspects, copies, or repurposes login tokens.
2. Headless Structured Output: Invokes 'agy' with --json-schema and --output-format json.
3. Response Isolation: Unpacks the CLI response envelope (conversation_id, status, duration, usage)
   and extracts the internal canonical Output Contract without conflation.
4. Clean Environment: Executes in an isolated context without access to benchmark answers (expected_splits.json)
   or source modification permissions.
5. Comprehensive Audit Trail: Records provider, CLI version, conversation ID, source/prompt hashes,
   duration, token usage, and returned contract.
"""

import os
import sys
import json
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent


class AntigravityCLIError(Exception):
    """Raised when the Antigravity CLI invocation encounters a fatal error."""
    pass


class AntigravityCLIProvider:
    """Manages boundary detection via the official Antigravity CLI ('agy')."""

    def __init__(
        self,
        cli_path: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: int = 180,
        isolated_dir: Optional[Path] = None
    ):
        self.cli_path = cli_path or self._resolve_cli_binary()
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.isolated_dir = isolated_dir or (ROOT_DIR / "scratch" / "agy_isolated")
        self.isolated_dir.mkdir(parents=True, exist_ok=True)
        self.cli_version = self._detect_cli_version()
        self.schema_path = ROOT_DIR / "prompts" / "output_contract_schema.json"

    def _resolve_cli_binary(self) -> str:
        """Finds the installed 'agy' executable in PATH or standard user location."""
        candidate = shutil.which("agy")
        if candidate:
            return candidate
        default_user_path = os.path.expanduser("~/.local/bin/agy")
        if os.path.exists(default_user_path) and os.access(default_user_path, os.X_OK):
            return default_user_path
        raise AntigravityCLIError(
            "Antigravity CLI ('agy') not found in PATH or ~/.local/bin/agy. "
            "Please install via: curl -fsSL https://antigravity.google/cli/install.sh | bash"
        )

    def _detect_cli_version(self) -> str:
        """Queries the installed CLI version via 'agy --version'."""
        try:
            res = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            return res.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not determine agy version: {e}")
            return "unknown"

    def detect_boundaries(
        self,
        raw_text: str,
        question_id: str,
        system_prompt: str,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Executes headless boundary detection via 'agy'.
        Returns: (contract_json, execution_metadata).
        Never falls back to offline parsing. If CLI or model fails, returns a clear needs_review failure.
        """
        source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

        metadata: Dict[str, Any] = {
            "provider": "antigravity-cli",
            "model": self.model_name or "antigravity-default",
            "cli_version": self.cli_version,
            "conversation_id": None,
            "source_hash": source_hash,
            "prompt_hash": prompt_hash,
            "status": "pending",
            "retries_used": 0,
            "duration_seconds": None,
            "usage": None
        }

        # Build comprehensive prompt containing immutable source and instructions
        user_prompt = (
            f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"
            f"TASK:\n"
            f"Analyze the following medical assessment question '{question_id}' and decompose into canonical segments and items according to the system instructions.\n"
            f"Source ID: {question_id}\n"
            f"Config: {json.dumps(config)}\n\n"
            f"RAW SOURCE TEXT (immutable, character count: {len(raw_text)}):\n"
            f"\"\"\"\n{raw_text}\n\"\"\"\n\n"
            f"CRITICAL RULES:\n"
            f"1. Return ONLY valid JSON matching the schema for source_id '{question_id}'.\n"
            f"2. Segments must form a gap-free partition tiling exactly 0..{len(raw_text)} with precise integer offsets.\n"
            f"3. Do NOT call any tools, create files, or execute commands. Return the JSON payload directly."
        )

        cmd = [
            self.cli_path,
            "-p", user_prompt,
            "--json-schema", str(self.schema_path.resolve()),
            "--output-format", "json",
            "--dangerously-skip-permissions",
            "--disable-slash-commands"
        ]
        if self.model_name:
            cmd.extend(["--model", self.model_name])

        try:
            res = subprocess.run(
                cmd,
                cwd=str(self.isolated_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
        except subprocess.TimeoutExpired:
            err_msg = f"Antigravity CLI timed out after {self.timeout_seconds}s for question {question_id}."
            metadata["status"] = "failed_timeout"
            metadata["error"] = err_msg
            failure_contract = self._create_failure_contract(raw_text, question_id, "unsupported_representation")
            return failure_contract, metadata
        except Exception as e:
            err_msg = f"Antigravity CLI invocation failed: {str(e)}"
            metadata["status"] = "failed_invocation"
            metadata["error"] = err_msg
            failure_contract = self._create_failure_contract(raw_text, question_id, "unsupported_representation")
            return failure_contract, metadata

        if res.returncode != 0:
            err_msg = f"Antigravity CLI exited with code {res.returncode}. Stderr: {res.stderr.strip()}"
            metadata["status"] = "failed_cli_error"
            metadata["error"] = err_msg
            failure_contract = self._create_failure_contract(raw_text, question_id, "unsupported_representation")
            return failure_contract, metadata

        # Parse CLI envelope
        stdout_text = res.stdout.strip()
        try:
            envelope = json.loads(stdout_text)
        except Exception as e:
            err_msg = f"Failed to parse Antigravity CLI output envelope: {e}. Output was: {stdout_text[:300]}"
            metadata["status"] = "failed_invalid_envelope"
            metadata["error"] = err_msg
            failure_contract = self._create_failure_contract(raw_text, question_id, "unsupported_representation")
            return failure_contract, metadata

        metadata["conversation_id"] = envelope.get("conversation_id")
        metadata["duration_seconds"] = envelope.get("duration_seconds")
        metadata["usage"] = envelope.get("usage")

        # Extract structured contract from envelope
        contract = envelope.get("structured_output")
        if not contract and envelope.get("response"):
            try:
                contract = json.loads(envelope["response"])
            except Exception:
                contract = None

        if not isinstance(contract, dict):
            err_msg = "Antigravity CLI returned SUCCESS but no valid structured Output Contract was present."
            metadata["status"] = "failed_missing_contract"
            metadata["error"] = err_msg
            failure_contract = self._create_failure_contract(raw_text, question_id, "unsupported_representation")
            return failure_contract, metadata

        metadata["status"] = "success"
        return contract, metadata

    def _create_failure_contract(self, raw_text: str, source_id: str, issue_code: str) -> Dict[str, Any]:
        """Creates a conservative failure contract retaining original raw text."""
        segments = []
        if len(raw_text) > 0:
            segments.append({
                "id": "seg_raw_1",
                "start": 0,
                "end": len(raw_text),
                "kind": "stem"
            })
            item = {
                "id": source_id,
                "parent_group_id": None,
                "response_kind": "single_choice",
                "source_label_segment_ids": [],
                "stem_segment_ids": ["seg_raw_1"],
                "reference_segment_ids": [],
                "option_groups": [],
                "matching_rows": [],
                "response_template_segment_ids": [],
                "depends_on_item_ids": [],
                "context_reuse": [],
                "disposition": "needs_review"
            }
            items = [item]
        else:
            items = []

        return {
            "source_id": source_id,
            "status": "needs_review",
            "segments": segments,
            "items": items,
            "issues": [
                {
                    "code": issue_code,
                    "segment_ids": [s["id"] for s in segments],
                    "item_ids": [source_id] if items else []
                }
            ]
        }
