"""
prompt_loader.py

Utility to load and inspect prompts/extraction_prompt.md.
Extracts system instructions, platform constraints, and JSON schema
for AI boundary detection models.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import re

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT_DIR / "prompts" / "extraction_prompt.md"


def load_extraction_prompt_text() -> str:
    """Reads the full extraction prompt markdown file."""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Extraction prompt not found at {PROMPT_FILE}")
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def load_system_instruction() -> str:
    """
    Extracts the core system instruction markdown block from extraction_prompt.md.
    """
    text = load_extraction_prompt_text()
    match = re.search(r"## System Instructions\s+```markdown(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def load_json_schema_definition() -> str:
    """
    Extracts the JSON output schema from extraction_prompt.md.
    """
    text = load_extraction_prompt_text()
    match = re.search(r"## Output JSON Schema\s+```json(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "{}"
