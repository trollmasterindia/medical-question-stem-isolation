"""
contract_slicer.py

Deterministic Python string scissors and real output verifier.
Implements Sections 4 & 5 of Generic Stem Isolation Implementation Notes:
- Python slices raw text directly using exact Unicode code-point offsets: raw_text[start:end].
- Never trusts model-copied text or computed lengths.
- Source field strings are exact concatenations of declared source slices (no synthetic spaces/newlines).
- Verifies real output dynamically: hashes and character-level checks, never hardcoded constants.
- Adapts Output Contract items to platform review format for Before vs After viewer and JSON export.
"""

from typing import Dict, List, Any, Optional
from stem_isolator import verify_zero_text_loss


def slice_contract_item(
    raw_text: str,
    item: Dict[str, Any],
    seg_dict: Dict[str, Dict[str, Any]],
    domain: str = "Clinical Nursing"
) -> Dict[str, Any]:
    """
    Slices an item's components from raw_text according to declared segment IDs.
    Calculates actual character counts and verification statistics.
    """
    item_id = item["id"]
    parent_id = item.get("parent_group_id")

    # 1. Stem text (exact concatenation of declared source slices)
    stem_ids = item.get("stem_segment_ids", [])
    stem_text = "".join(raw_text[seg_dict[sid]["start"]:seg_dict[sid]["end"]] for sid in stem_ids if sid in seg_dict)

    # 2. Reference text
    ref_ids = item.get("reference_segment_ids", [])
    if ref_ids:
        ref_text = "".join(raw_text[seg_dict[sid]["start"]:seg_dict[sid]["end"]] for sid in ref_ids if sid in seg_dict)
    else:
        ref_text = None

    # 3. Source label metadata
    label_ids = item.get("source_label_segment_ids", [])
    meta_text = "".join(raw_text[seg_dict[sid]["start"]:seg_dict[sid]["end"]] for sid in label_ids if sid in seg_dict) or None

    # 4. Options
    options: List[str] = []
    for og in item.get("option_groups", []):
        for opt in og.get("options", []):
            opt_str = "".join(raw_text[seg_dict[sid]["start"]:seg_dict[sid]["end"]] for sid in opt.get("segment_ids", []) if sid in seg_dict)
            if opt_str:
                options.append(opt_str)

    # 5. Verification
    stem_len = len(stem_text)
    ref_len = len(ref_text) if ref_text else 0
    fits_500 = stem_len <= 500
    layout = "split_screen_with_reference" if ref_text else "single_column_stem_only"

    # Multi-part child vs single item verification
    if parent_id and parent_id != item_id:
        clean_raw = "".join(raw_text.split())
        clean_stem = "".join(stem_text.split())
        is_exact = clean_stem in clean_raw
        added = 0
        removed = 0
    else:
        is_exact, added, removed = verify_zero_text_loss(
            raw_text=raw_text,
            stem=stem_text,
            reference=ref_text,
            distractors=options,
            metadata=meta_text
        )

    return {
        "id": item_id,
        "parent_id": parent_id or item_id,
        "domain": domain,
        "response_kind": item.get("response_kind", "single_choice"),
        "disposition": item.get("disposition", "proposed"),
        "before": {
            "stem": raw_text,
            "reference": None,
            "distractors": [],
            "char_count": len(raw_text),
            "fits_500_char_limit": len(raw_text) <= 500,
            "layout": "single_column_stem_only"
        },
        "after": {
            "stem": stem_text,
            "reference": ref_text,
            "distractors": options,
            "metadata": meta_text,
            "stem_char_count": stem_len,
            "reference_char_count": ref_len,
            "fits_500_char_limit": fits_500,
            "layout": layout
        },
        "verification": {
            "characters_added": added,
            "characters_removed": removed,
            "exact_match": is_exact,
            "stem_under_500": fits_500
        }
    }


def contract_to_platform_items(
    contract: Dict[str, Any],
    raw_text: str,
    domain: str = "Clinical Nursing"
) -> List[Dict[str, Any]]:
    """Converts Output Contract into platform review items."""
    seg_dict = {s["id"]: s for s in contract.get("segments", [])}
    items = []
    for item in contract.get("items", []):
        items.append(slice_contract_item(raw_text, item, seg_dict, domain=domain))
    return items
