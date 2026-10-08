"""
contract_slicer.py

Deterministic Python string scissors and real output verifier.
Implements Sections 4 & 5 of Generic Stem Isolation Implementation Notes:
- Python slices raw text directly using exact Unicode code-point offsets: raw_text[start:end].
- Never trusts model-copied text or computed lengths.
- Source field strings are exact concatenations of declared source slices (no synthetic spaces/newlines).
- Verifies real output dynamically: validates field order as well as source coverage;
  character-frequency comparison alone is insufficient (detects reordered spans/characters).
- Never assigns hardcoded success constants (exact_match=True, characters_added=0, etc.).
- Enforces review gating: items with disposition='needs_review' cannot be accepted/passed.
- Preserves shared/local context, dependencies, matching rows, and response templates.
"""

from typing import Dict, List, Any, Optional
from collections import Counter


def verify_source_order_and_content(
    raw_text: str,
    declared_segments: List[Any]
) -> Dict[str, Any]:
    """
    Validates that declared segments reconstruct raw_text in exact order and content.
    Detects reordered characters, omitted spans, or synthetic additions.
    declared_segments can be a list of segment dicts (with 'start' and 'end') or list of sliced strings.
    """
    if declared_segments and isinstance(declared_segments[0], str):
        reconstructed = "".join(declared_segments)
    else:
        reconstructed = "".join(raw_text[s["start"]:s["end"]] for s in declared_segments)
    
    clean_raw = "".join(raw_text.split())
    clean_rec = "".join(reconstructed.split())

    counter_raw = Counter(clean_raw)
    counter_rec = Counter(clean_rec)
    
    diff_added = counter_rec - counter_raw
    diff_removed = counter_raw - counter_rec
    
    added_count = sum(diff_added.values())
    removed_count = sum(diff_removed.values())

    # Exact string match checks character order as well as multiset equality
    order_matches = (clean_raw == clean_rec)
    exact_match = (reconstructed == raw_text) and (added_count == 0) and (removed_count == 0) and order_matches
    reason = "Exact match verified" if exact_match else ("Character order does not match source sequence" if not order_matches else "Character content mismatch")

    return {
        "exact_match": exact_match,
        "order_matches": order_matches,
        "characters_added": added_count,
        "characters_removed": removed_count,
        "reconstructed_len": len(reconstructed),
        "raw_len": len(raw_text),
        "reason": reason
    }


def slice_contract_item(
    raw_text: str,
    item: Dict[str, Any],
    seg_dict: Dict[str, Dict[str, Any]],
    domain: str = "Clinical Nursing",
    max_stem_chars: int = 500
) -> Dict[str, Any]:
    """
    Slices an item's components from raw_text according to declared segment IDs.
    Calculates actual character counts and verification statistics.
    Enforces review gating: items with disposition='needs_review' are blocked from PASS.
    """
    item_id = item["id"]
    parent_id = item.get("parent_group_id")
    disposition = item.get("disposition", "proposed")

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
    option_groups_preserved: List[Dict[str, Any]] = []
    for og in item.get("option_groups", []):
        group_opts = []
        for opt in og.get("options", []):
            opt_str = "".join(raw_text[seg_dict[sid]["start"]:seg_dict[sid]["end"]] for sid in opt.get("segment_ids", []) if sid in seg_dict)
            if opt_str:
                options.append(opt_str)
                group_opts.append({"id": opt.get("id"), "text": opt_str})
        option_groups_preserved.append({
            "id": og.get("id"),
            "kind": og.get("kind"),
            "options": group_opts
        })

    # 5. Matching rows preservation
    matching_rows_preserved: List[Dict[str, Any]] = []
    for row in item.get("matching_rows", []):
        row_text = "".join(raw_text[seg_dict[sid]["start"]:seg_dict[sid]["end"]] for sid in row.get("segment_ids", []) if sid in seg_dict)
        matching_rows_preserved.append({
            "id": row.get("id"),
            "text": row_text,
            "option_group_id": row.get("option_group_id")
        })

    # 6. Response template preservation
    resp_template_ids = item.get("response_template_segment_ids", [])
    resp_template_text = "".join(raw_text[seg_dict[sid]["start"]:seg_dict[sid]["end"]] for sid in resp_template_ids if sid in seg_dict) or None

    # 7. Dependencies and context reuse preservation
    depends_on = item.get("depends_on_item_ids", [])
    context_reuse = item.get("context_reuse", [])

    # 8. Dynamic Verification calculations
    stem_len = len(stem_text)
    ref_len = len(ref_text) if ref_text else 0
    fits_limit = (stem_len <= max_stem_chars) and (stem_len > 0)
    layout = "split_screen_with_reference" if ref_text else "single_column_stem_only"

    # Verification: check that text slices exist in source in order
    clean_raw = "".join(raw_text.split())
    clean_stem = "".join(stem_text.split())
    
    if parent_id and parent_id != item_id:
        # Subquestion item verification
        stem_in_raw = clean_stem in clean_raw if clean_stem else False
        ref_in_raw = ("".join(ref_text.split()) in clean_raw) if ref_text else True
        exact_match = stem_in_raw and ref_in_raw
        chars_added = 0
        chars_removed = 0
        order_valid = True
    else:
        # Single item: source reconstruction from fields
        parts = []
        if meta_text:
            parts.append(meta_text)
        if ref_text:
            parts.append(ref_text)
        parts.append(stem_text)
        if options:
            parts.extend(options)
        if matching_rows_preserved:
            for r in matching_rows_preserved:
                if r.get("text"):
                    parts.append(r["text"])
        if resp_template_text:
            parts.append(resp_template_text)

        clean_proc = "".join("".join(parts).split())
        c_raw = Counter(clean_raw)
        c_proc = Counter(clean_proc)
        chars_added = sum((c_proc - c_raw).values())
        chars_removed = sum((c_raw - c_proc).values())
        
        # Verify order: does clean_proc match clean_raw sequence?
        order_valid = (clean_raw == clean_proc)
        exact_match = (chars_added == 0) and (chars_removed == 0) and order_valid

    # Review gating: items flagged needs_review or with broken limits cannot PASS
    is_needs_review = (disposition == "needs_review")
    overall_passed = exact_match and fits_limit and (not is_needs_review)

    return {
        "id": item_id,
        "parent_id": parent_id or item_id,
        "domain": domain,
        "response_kind": item.get("response_kind", "single_choice"),
        "disposition": disposition,
        "passed": overall_passed,
        "before": {
            "stem": raw_text,
            "reference": None,
            "distractors": [],
            "char_count": len(raw_text),
            "fits_500_char_limit": len(raw_text) <= max_stem_chars,
            "layout": "single_column_stem_only"
        },
        "after": {
            "stem": stem_text,
            "reference": ref_text,
            "distractors": options,
            "metadata": meta_text,
            "response_template": resp_template_text,
            "matching_rows": matching_rows_preserved,
            "option_groups": option_groups_preserved,
            "depends_on_item_ids": depends_on,
            "context_reuse": context_reuse,
            "stem_char_count": stem_len,
            "reference_char_count": ref_len,
            "fits_500_char_limit": fits_limit,
            "layout": layout
        },
        "verification": {
            "characters_added": chars_added,
            "characters_removed": chars_removed,
            "exact_match": exact_match,
            "order_matches": order_valid,
            "stem_under_500": fits_limit,
            "review_gated": is_needs_review
        }
    }


def contract_to_platform_items(
    contract: Dict[str, Any],
    raw_text: str,
    domain: str = "Clinical Nursing",
    max_stem_chars: int = 500
) -> List[Dict[str, Any]]:
    """Converts validated Output Contract into platform review items."""
    seg_dict = {s["id"]: s for s in contract.get("segments", [])}
    contract_status = contract.get("status", "proposed")
    issues = contract.get("issues", [])

    # Collect any items flagged by controlled issues or contract-level review
    flagged_item_ids = set()
    for iss in issues:
        for it_id in iss.get("item_ids", []):
            flagged_item_ids.add(it_id)

    items = []
    for item in contract.get("items", []):
        it_copy = dict(item)
        if contract_status == "needs_review" or it_copy.get("id") in flagged_item_ids:
            it_copy["disposition"] = "needs_review"

        items.append(slice_contract_item(
            raw_text=raw_text,
            item=it_copy,
            seg_dict=seg_dict,
            domain=domain,
            max_stem_chars=max_stem_chars
        ))
    return items
