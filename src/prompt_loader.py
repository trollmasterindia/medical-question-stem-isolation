"""
prompt_loader.py

Utility to load and inspect the master medical stem isolation system prompt
(prompts/medical_stem_isolation_system_prompt.txt) and extraction guidelines.
"""

from pathlib import Path
from typing import Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
MASTER_PROMPT_FILE = ROOT_DIR / "prompts" / "medical_stem_isolation_system_prompt.txt"
LEGACY_PROMPT_FILE = ROOT_DIR / "prompts" / "extraction_prompt.md"


def get_default_config() -> Dict[str, Any]:
    """Returns the recommended initial configuration defined in the contract."""
    return {
        "max_stem_chars": 500,
        "split_policy": "split_labeled_if_safe",
        "matching_policy": "review",
        "supports_ordering": False,
        "supports_matching": False,
        "supports_linked_items": False,
        "supports_conditional_items": False,
        "reference_placement": "beside"
    }


def load_master_prompt_text() -> str:
    """Reads the full master system prompt file."""
    if MASTER_PROMPT_FILE.exists():
        with open(MASTER_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    elif LEGACY_PROMPT_FILE.exists():
        with open(LEGACY_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    raise FileNotFoundError("No system prompt found in prompts directory.")


def load_system_instruction() -> str:
    """Returns the system instruction text to be passed to LLM calls."""
    return load_master_prompt_text()


def load_json_schema_definition() -> str:
    """Returns the Output Contract schema summary from the prompt."""
    return """
{
  "source_id": "string",
  "status": "proposed | needs_review",
  "segments": [
    {"id": "string", "start": 0, "end": 100, "kind": "stem | reference | option | metadata | layout | response_template | unresolved"}
  ],
  "items": [
    {
      "id": "string",
      "parent_group_id": "string | null",
      "response_kind": "single_choice | multiple_response | short_answer | true_false | cloze | ordering | matching | multipart | unknown",
      "source_label_segment_ids": ["string"],
      "stem_segment_ids": ["string"],
      "reference_segment_ids": ["string"],
      "option_groups": [
        {
          "id": "string",
          "kind": "choice_bank | matching_bank | ordering_bank",
          "options": [{"id": "string", "segment_ids": ["string"]}]
        }
      ],
      "matching_rows": [],
      "response_template_segment_ids": [],
      "depends_on_item_ids": [],
      "context_reuse": [],
      "disposition": "proposed | needs_review"
    }
  ],
  "issues": [
    {"code": "controlled_issue_code", "segment_ids": ["string"], "item_ids": ["string"]}
  ]
}
"""
