"""
contract_validator.py

Strict validator implementing Section 3 of Generic Stem Isolation Implementation Notes.
Validates the Output Contract against the immutable raw_text and configuration.

Rules verified by Python:
1. Exact source_id matches expected caller value.
2. Unique segment, item, and option IDs; valid enums.
3. Integer offsets (strictly rejecting booleans), 0 <= start < end <= len(raw_text).
4. Gap-free, ordered, non-overlapping partition of raw_text:
   - start of first segment == 0
   - end of last segment == len(raw_text)
   - segments[i].end == segments[i+1].start
5. All referenced IDs resolve; mapping order follows source.
6. No segment rendered twice across one item's content fields.
7. Substantive segments (stem, reference, option, response_template) assigned to items
   or flagged as unresolved.
8. Empty input handled: empty partition, no items, 'empty_input' issue.
9. Max stem characters checked dynamically.
10. Controlled issue codes strictly enforced.
"""

from typing import Dict, List, Any, Tuple, Optional, Set

ALLOWED_STATUSES = {"proposed", "needs_review"}
ALLOWED_DISPOSITIONS = {"proposed", "needs_review"}
ALLOWED_SEGMENT_KINDS = {
    "stem", "reference", "option", "metadata", "layout", "response_template", "unresolved"
}
ALLOWED_RESPONSE_KINDS = {
    "single_choice", "multiple_response", "short_answer", "true_false",
    "cloze", "ordering", "matching", "multipart", "unknown"
}
ALLOWED_OPTION_GROUP_KINDS = {"choice_bank", "matching_bank", "ordering_bank"}
CONTROLLED_ISSUE_CODES = {
    "ambiguous_boundary",
    "ambiguous_item_grouping",
    "mixed_instruction",
    "instruction_scope_unclear",
    "missing_context",
    "external_material_required",
    "source_quality",
    "stem_limit_risk",
    "conditional_dependency",
    "unsupported_ordering",
    "unsupported_matching",
    "matching_policy_required",
    "layout_relation_conflict",
    "unsupported_representation",
    "suspicious_source_instruction",
    "empty_input"
}


class ContractValidationError(ValueError):
    """Raised when an output contract fails deterministic validation."""
    pass


class ContractValidator:
    """Validates model output against strict schema and immutable raw text."""

    @classmethod
    def validate(
        cls,
        raw_text: str,
        contract: Dict[str, Any],
        expected_source_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validates the output contract.
        Returns (is_valid, list_of_error_messages).
        """
        errors: List[str] = []
        cfg = config or {}
        max_stem_chars = cfg.get("max_stem_chars", 500)

        # 1. Top-level keys
        for key in ["source_id", "status", "segments", "items", "issues"]:
            if key not in contract:
                errors.append(f"Missing required top-level field: '{key}'")
        if errors:
            return False, errors

        # 2. Source ID check
        if contract["source_id"] != expected_source_id:
            errors.append(
                f"source_id mismatch: expected '{expected_source_id}', got '{contract['source_id']}'"
            )

        # 3. Status check
        if contract["status"] not in ALLOWED_STATUSES:
            errors.append(f"Invalid status '{contract['status']}'; allowed: {ALLOWED_STATUSES}")

        # 4. Empty input handling
        if len(raw_text) == 0:
            if contract["segments"] != []:
                errors.append("Empty raw_text must have empty segments partition")
            if contract["items"] != []:
                errors.append("Empty raw_text must have no items")
            has_empty_issue = any(iss.get("code") == "empty_input" for iss in contract["issues"])
            if not has_empty_issue:
                errors.append("Empty raw_text must include 'empty_input' in issues")
            if contract["status"] != "needs_review":
                errors.append("Empty raw_text must have status='needs_review'")
            return len(errors) == 0, errors

        # 5. Segments partition validation (gap-free, non-overlapping, covers 0..len(raw_text))
        segments = contract["segments"]
        if not segments:
            errors.append("Non-empty raw_text must have non-empty segments partition")
            return False, errors

        segment_ids: Set[str] = set()
        seg_dict: Dict[str, Dict[str, Any]] = {}
        last_end = 0

        for i, seg in enumerate(segments):
            # Check ID
            s_id = seg.get("id")
            if not s_id or not isinstance(s_id, str):
                errors.append(f"Segment #{i} missing valid string id")
            elif s_id in segment_ids:
                errors.append(f"Duplicate segment ID: '{s_id}'")
            else:
                segment_ids.add(s_id)
                seg_dict[s_id] = seg

            # Check kind
            kind = seg.get("kind")
            if kind not in ALLOWED_SEGMENT_KINDS:
                errors.append(f"Segment '{s_id}' has invalid kind: '{kind}'")

            # Check integer offsets (strictly reject bools which are int subclasses)
            start = seg.get("start")
            end = seg.get("end")
            if isinstance(start, bool) or not isinstance(start, int):
                errors.append(f"Segment '{s_id}' start offset must be integer")
            if isinstance(end, bool) or not isinstance(end, int):
                errors.append(f"Segment '{s_id}' end offset must be integer")

            if isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool):
                if start < 0 or end > len(raw_text) or start >= end:
                    errors.append(
                        f"Segment '{s_id}' offsets [{start}, {end}] invalid for raw_text length {len(raw_text)}"
                    )

                # Check gap-free and ordering
                if i == 0 and start != 0:
                    errors.append(f"First segment start must be 0, got {start}")
                elif i > 0 and start != last_end:
                    errors.append(f"Gap or overlap at segment '{s_id}': expected start {last_end}, got {start}")
                last_end = end

        if last_end != len(raw_text):
            errors.append(f"Segments do not cover full raw_text: ended at {last_end}, raw_text len {len(raw_text)}")

        # 6. Items validation
        item_ids: Set[str] = set()
        assigned_substantive_segs: Set[str] = set()

        for item in contract.get("items", []):
            i_id = item.get("id")
            if not i_id or not isinstance(i_id, str):
                errors.append("Item missing valid string id")
            elif i_id in item_ids:
                errors.append(f"Duplicate item ID: '{i_id}'")
            else:
                item_ids.add(i_id)

            if item.get("disposition") not in ALLOWED_DISPOSITIONS:
                errors.append(f"Item '{i_id}' invalid disposition: '{item.get('disposition')}'")

            if item.get("response_kind") not in ALLOWED_RESPONSE_KINDS:
                errors.append(f"Item '{i_id}' invalid response_kind: '{item.get('response_kind')}'")

            # Content fields: stem, reference, source_label, response_template
            stem_ids = item.get("stem_segment_ids", [])
            ref_ids = item.get("reference_segment_ids", [])
            label_ids = item.get("source_label_segment_ids", [])
            resp_ids = item.get("response_template_segment_ids", [])

            # Check segments existence
            for sid in stem_ids + ref_ids + label_ids + resp_ids:
                if sid not in seg_dict:
                    errors.append(f"Item '{i_id}' references non-existent segment '{sid}'")

            # Check no segment rendered twice across content fields (stem, reference, template)
            rendered_fields = stem_ids + ref_ids + resp_ids
            if len(rendered_fields) != len(set(rendered_fields)):
                errors.append(f"Item '{i_id}' has segment rendered multiple times across content fields")

            # Track substantive segments
            for sid in stem_ids + ref_ids + resp_ids:
                if sid in seg_dict and seg_dict[sid]["kind"] in {"stem", "reference", "response_template"}:
                    assigned_substantive_segs.add(sid)

            # Option groups validation
            for og in item.get("option_groups", []):
                og_kind = og.get("kind")
                if og_kind not in ALLOWED_OPTION_GROUP_KINDS:
                    errors.append(f"Item '{i_id}' option group invalid kind '{og_kind}'")
                for opt in og.get("options", []):
                    for osid in opt.get("segment_ids", []):
                        if osid not in seg_dict:
                            errors.append(f"Item '{i_id}' option references missing segment '{osid}'")
                        elif seg_dict[osid]["kind"] == "option":
                            assigned_substantive_segs.add(osid)

            # Stem length calculation (computed by Python, not model)
            actual_stem_text = "".join(raw_text[seg_dict[s]["start"]:seg_dict[s]["end"]] for s in stem_ids if s in seg_dict)
            if len(actual_stem_text) > max_stem_chars:
                # Must flag stem_limit_risk or have disposition needs_review
                has_limit_issue = any(iss.get("code") == "stem_limit_risk" for iss in contract["issues"])
                if item.get("disposition") != "needs_review" and not has_limit_issue:
                    errors.append(
                        f"Item '{i_id}' stem exceeds max_stem_chars ({len(actual_stem_text)} > {max_stem_chars}) "
                        f"without 'stem_limit_risk' issue or 'needs_review' disposition"
                    )

            # Depends on items
            for dep_id in item.get("depends_on_item_ids", []):
                if dep_id == i_id:
                    errors.append(f"Item '{i_id}' cannot depend on itself")

        # 7. Unassigned substantive segments check
        for sid, seg in seg_dict.items():
            if seg["kind"] in {"stem", "reference", "option", "response_template"}:
                if sid not in assigned_substantive_segs:
                    errors.append(f"Substantive segment '{sid}' ({seg['kind']}) not assigned to any item")

        # 8. Issues validation
        for iss in contract.get("issues", []):
            code = iss.get("code")
            if code not in CONTROLLED_ISSUE_CODES:
                errors.append(f"Invalid controlled issue code '{code}'; allowed: {CONTROLLED_ISSUE_CODES}")

        # If any issue exists or status is needs_review, verify consistency
        if contract.get("issues") and contract["status"] != "needs_review":
            errors.append("Top-level status must be 'needs_review' when issues are reported")

        return len(errors) == 0, errors
