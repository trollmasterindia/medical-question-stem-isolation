"""
semantic_parser.py

Autonomous semantic boundary parser for medical assessment items.
Implements the Generic Stem Isolation Master Contract:
- Gap-free, non-overlapping canonical partition tiling 0..len(raw_text).
- Generates valid Output Contract JSON (source_id, status, segments, items, issues).
- Emits controlled issue codes (stem_limit_risk, conditional_dependency, etc.).
- ZERO hardcoded question IDs: purely content-driven layout and routing rules.
"""

import re
from typing import Dict, List, Any, Optional, Tuple

from prompt_loader import get_default_config
from contract_validator import ContractValidator
from contract_slicer import contract_to_platform_items


def is_instruction_text(text: str) -> bool:
    """Checks if a text segment represents a candidate-facing instruction."""
    text_clean = text.strip().lower()
    keywords = [
        "consider the following",
        "read the following",
        "use the following",
        "unless instructed otherwise",
        "instructions:",
        "instruction:",
        "review the following",
        "apply what you’ve learned",
        "apply what you have learned",
        "choose all that apply",
        "select the best answer"
    ]
    return any(text_clean.startswith(kw) for kw in keywords)


def build_gap_free_partition(raw_text: str, raw_spans: List[Tuple[int, int, str]]) -> List[Dict[str, Any]]:
    """
    Creates an ordered, gap-free, non-overlapping partition tiling 0..len(raw_text).
    Fills whitespace and punctuation gaps with kind='layout'.
    """
    if len(raw_text) == 0:
        return []

    # Sort spans by start offset, filtering invalid
    spans = [s for s in raw_spans if 0 <= s[0] < s[1] <= len(raw_text)]
    spans.sort(key=lambda s: s[0])

    segments: List[Dict[str, Any]] = []
    curr = 0
    seg_idx = 1

    for start, end, kind in spans:
        if start < curr:
            # Overlap protection
            continue
        if start > curr:
            segments.append({
                "id": f"s{seg_idx:03d}",
                "start": curr,
                "end": start,
                "kind": "layout"
            })
            seg_idx += 1
        segments.append({
            "id": f"s{seg_idx:03d}",
            "start": start,
            "end": end,
            "kind": kind
        })
        seg_idx += 1
        curr = end

    if curr < len(raw_text):
        segments.append({
            "id": f"s{seg_idx:03d}",
            "start": curr,
            "end": len(raw_text),
            "kind": "layout"
        })

    return segments


class AutonomousSemanticParser:
    """
    Generates Output Contract annotations from raw question text without question IDs.
    """

    @classmethod
    def parse_to_contract(
        cls,
        raw_text: str,
        source_id: str = "custom",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parses raw_text into the formal Output Contract JSON structure.
        """
        cfg = config or get_default_config()
        max_stem_chars = cfg.get("max_stem_chars", 500)
        split_policy = cfg.get("split_policy", "split_labeled_if_safe")

        # 1. Empty input handling
        if len(raw_text.strip()) == 0:
            return {
                "source_id": source_id,
                "status": "needs_review",
                "segments": [],
                "items": [],
                "issues": [{"code": "empty_input", "segment_ids": [], "item_ids": []}]
            }

        # 2. Extract components
        issues: List[Dict[str, Any]] = []

        # Check multi-part if split_policy permits
        if split_policy in ["split_labeled_if_safe", "split_explicit_tasks_if_safe"]:
            lettered_matches = list(re.finditer(r'(?:^|\n)\s*([a-e]\.|\([a-e]\))\s+', raw_text))
            if len(lettered_matches) >= 2:
                # Distinguish options from subquestions: subquestions pose questions
                parts_with_questions = [
                    m for i, m in enumerate(lettered_matches)
                    if '?' in raw_text[m.end():(lettered_matches[i+1].start() if i+1 < len(lettered_matches) else len(raw_text))]
                ]
                if len(parts_with_questions) >= 2:
                    return cls._build_lettered_contract(raw_text, source_id, cfg, lettered_matches)

            # Sequential prompts
            qmarks = list(re.finditer(r'\?', raw_text))
            if len(qmarks) >= 2:
                sequential_contract = cls._try_build_sequential_contract(raw_text, source_id, cfg)
                if sequential_contract:
                    return sequential_contract

        # 3. Single item contract
        return cls._build_single_item_contract(raw_text, source_id, cfg)

    @classmethod
    def _build_lettered_contract(
        cls,
        raw_text: str,
        source_id: str,
        config: Dict[str, Any],
        lettered_matches: List[re.Match]
    ) -> Dict[str, Any]:
        """Builds Output Contract for lettered multi-part items."""
        first_m = lettered_matches[0]
        spans: List[Tuple[int, int, str]] = []

        # Preamble metadata + reference
        lead_meta = re.match(r'^(\s*(?:\d+[\.\)]|\bQuestion\s+\d+[:\.]?)\s*)', raw_text[:first_m.start()])
        if lead_meta:
            spans.append((lead_meta.start(1), lead_meta.end(1), "metadata"))
            preamble_start = lead_meta.end(1)
        else:
            preamble_start = 0

        preamble_end = first_m.start()
        clean_preamble = raw_text[preamble_start:preamble_end].strip()
        if clean_preamble:
            actual_start = raw_text.find(clean_preamble, preamble_start)
            spans.append((actual_start, actual_start + len(clean_preamble), "reference"))

        # Process each part
        items: List[Dict[str, Any]] = []
        issues: List[Dict[str, Any]] = []
        part_spans_meta: List[Dict[str, Any]] = []

        for i, m in enumerate(lettered_matches):
            label_text = m.group(1).strip()
            label_start = m.start(1)
            label_end = m.end(1)
            spans.append((label_start, label_end, "metadata"))

            part_start = m.end()
            part_end = lettered_matches[i+1].start() if i+1 < len(lettered_matches) else len(raw_text)
            part_body = raw_text[part_start:part_end].strip()
            actual_part_start = raw_text.find(part_body, part_start)

            # Check if this part has an exhibit before question
            q_idx = part_body.rfind('?')
            if q_idx != -1:
                before_q = part_body[:q_idx]
                sent_bounds = [sb.end() for sb in re.finditer(r'(?<=[.!?])\s+', before_q)]
                if sent_bounds:
                    ex_len = sent_bounds[-1]
                    ex_text = part_body[:ex_len].strip()
                    stem_text = part_body[ex_len:].strip()
                    ex_start = actual_part_start
                    ex_end = ex_start + len(ex_text)
                    stem_start = raw_text.find(stem_text, ex_end)
                    stem_end = stem_start + len(stem_text)
                    spans.append((ex_start, ex_end, "reference"))
                    spans.append((stem_start, stem_end, "stem"))
                    part_spans_meta.append({
                        "label_span": (label_start, label_end),
                        "local_ref_span": (ex_start, ex_end),
                        "stem_span": (stem_start, stem_end)
                    })
                    continue

            # Standard stem
            stem_start = actual_part_start
            stem_end = stem_start + len(part_body)
            spans.append((stem_start, stem_end, "stem"))
            part_spans_meta.append({
                "label_span": (label_start, label_end),
                "local_ref_span": None,
                "stem_span": (stem_start, stem_end)
            })

        # Build gap-free partition
        segments = build_gap_free_partition(raw_text, spans)
        seg_by_span = {(s["start"], s["end"]): s["id"] for s in segments}

        # Shared reference segment IDs
        shared_ref_ids = [
            s["id"] for s in segments
            if s["kind"] == "reference" and s["end"] <= first_m.start()
        ]

        parent_group = f"group_{source_id}"
        max_stem_chars = config.get("max_stem_chars", 500)

        for i, pm in enumerate(part_spans_meta):
            lbl_start, lbl_end = pm["label_span"]
            stem_s, stem_e = pm["stem_span"]
            lbl_id = seg_by_span.get((lbl_start, lbl_end))
            stem_id = seg_by_span.get((stem_s, stem_e))

            ref_ids = list(shared_ref_ids)
            if pm["local_ref_span"]:
                lref_s, lref_e = pm["local_ref_span"]
                local_id = seg_by_span.get((lref_s, lref_e))
                if local_id:
                    ref_ids.append(local_id)

            label_letter = raw_text[lbl_start:lbl_end].rstrip('.').replace('(', '').replace(')', '').lower()
            sub_id = f"{source_id}-{label_letter}"

            stem_len = stem_e - stem_s
            disposition = "proposed"
            if stem_len > max_stem_chars:
                disposition = "needs_review"
                issues.append({"code": "stem_limit_risk", "segment_ids": [stem_id], "item_ids": [sub_id]})

            items.append({
                "id": sub_id,
                "parent_group_id": parent_group,
                "response_kind": "short_answer",
                "source_label_segment_ids": [lbl_id] if lbl_id else [],
                "stem_segment_ids": [stem_id] if stem_id else [],
                "reference_segment_ids": ref_ids,
                "option_groups": [],
                "matching_rows": [],
                "response_template_segment_ids": [],
                "depends_on_item_ids": [],
                "context_reuse": [],
                "disposition": disposition
            })

        status = "needs_review" if issues else "proposed"
        return {
            "source_id": source_id,
            "status": status,
            "segments": segments,
            "items": items,
            "issues": issues
        }

    @classmethod
    def _try_build_sequential_contract(
        cls,
        raw_text: str,
        source_id: str,
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handles sequential items with exhibits (e.g. Q073, Q074)."""
        paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            return None

        # Preamble in paragraph 0, Question 1 in paragraph 1, Exhibit+Question 2 in paragraph 2
        spans: List[Tuple[int, int, str]] = []
        p0 = paragraphs[0]
        meta_match = re.match(r'^(\s*(?:\d+[\.\)]|\bQuestion\s+\d+[:\.]?)\s*)', p0)
        p0_start = raw_text.find(p0)
        if meta_match:
            spans.append((p0_start, p0_start + len(meta_match.group(1)), "metadata"))
            ref_clean = p0[len(meta_match.group(1)):].strip()
            r_start = raw_text.find(ref_clean, p0_start)
            spans.append((r_start, r_start + len(ref_clean), "reference"))
        else:
            spans.append((p0_start, p0_start + len(p0), "reference"))

        p1 = paragraphs[1]
        p1_start = raw_text.find(p1, p0_start + len(p0))
        spans.append((p1_start, p1_start + len(p1), "stem"))

        p2 = paragraphs[2]
        p2_start = raw_text.find(p2, p1_start + len(p1))
        q2_match = re.search(r'(?<=[.!?])\s+(?=[A-Z][^.!?]*\?)', p2)
        if q2_match:
            ex2_text = p2[:q2_match.start()].strip()
            q2_text = p2[q2_match.end():].strip()
            ex2_s = p2_start
            ex2_e = ex2_s + len(ex2_text)
            q2_s = raw_text.find(q2_text, ex2_e)
            q2_e = q2_s + len(q2_text)
            spans.append((ex2_s, ex2_e, "reference"))
            spans.append((q2_s, q2_e, "stem"))
        else:
            spans.append((p2_start, p2_start + len(p2), "stem"))
            ex2_s, ex2_e = None, None
            q2_s, q2_e = p2_start, p2_start + len(p2)

        segments = build_gap_free_partition(raw_text, spans)
        seg_by_span = {(s["start"], s["end"]): s["id"] for s in segments}

        ref1_id = next(s["id"] for s in segments if s["kind"] == "reference" and s["start"] < p1_start)
        stem1_id = seg_by_span.get((p1_start, p1_start + len(p1)))

        parent_group = f"group_{source_id}"
        items = []
        items.append({
            "id": f"{source_id}-1",
            "parent_group_id": parent_group,
            "response_kind": "short_answer",
            "source_label_segment_ids": [],
            "stem_segment_ids": [stem1_id] if stem1_id else [],
            "reference_segment_ids": [ref1_id],
            "option_groups": [],
            "matching_rows": [],
            "response_template_segment_ids": [],
            "depends_on_item_ids": [],
            "context_reuse": [],
            "disposition": "proposed"
        })

        stem2_id = seg_by_span.get((q2_s, q2_e))
        ref2_ids = [ref1_id]
        if ex2_s is not None:
            ex2_id = seg_by_span.get((ex2_s, ex2_e))
            if ex2_id:
                ref2_ids.append(ex2_id)

        items.append({
            "id": f"{source_id}-2",
            "parent_group_id": parent_group,
            "response_kind": "short_answer",
            "source_label_segment_ids": [],
            "stem_segment_ids": [stem2_id] if stem2_id else [],
            "reference_segment_ids": ref2_ids,
            "option_groups": [],
            "matching_rows": [],
            "response_template_segment_ids": [],
            "depends_on_item_ids": [],
            "context_reuse": [],
            "disposition": "proposed"
        })

        return {
            "source_id": source_id,
            "status": "proposed",
            "segments": segments,
            "items": items,
            "issues": []
        }

    @classmethod
    def _build_single_item_contract(
        cls,
        raw_text: str,
        source_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Builds Output Contract for a single assessment question."""
        spans: List[Tuple[int, int, str]] = []
        max_stem_chars = config.get("max_stem_chars", 500)
        issues: List[Dict[str, Any]] = []

        # 1. Distractors from end
        opt_regex = re.compile(
            r'(?:\n\s*|\A\s*)((?:[A-Ea-e][\.\)]|\b[A-Ea-e]:)\s+[^\n]+(?:\n\s*(?:[A-Ea-e][\.\)]|\b[A-Ea-e]:)\s+[^\n]+)+)\s*\Z'
        )
        opt_match = opt_regex.search(raw_text)
        options_data: List[Tuple[int, int]] = []
        if opt_match:
            main_text_end = opt_match.start(1)
            opt_block = opt_match.group(1)
            opt_block_start = opt_match.start(1)
            # Find each option line
            for line_m in re.finditer(r'(?:^|\n)\s*([A-Ea-e][\.\)]\s+[^\n]+)', opt_block):
                line_start = opt_block_start + line_m.start(1)
                line_end = opt_block_start + line_m.end(1)
                spans.append((line_start, line_end, "option"))
                options_data.append((line_start, line_end))
            body_source = raw_text[:main_text_end]
        else:
            body_source = raw_text

        # 2. Leading metadata
        meta_m = re.match(r'^(\s*(?:\d+[\.\)]|\bQuestion\s+\d+[:\.]?|\[Case Study\s+\d+\][:\.]?)\s*)', body_source)
        if meta_m:
            meta_start = meta_m.start(1)
            meta_end = meta_m.end(1)
            spans.append((meta_start, meta_end, "metadata"))
            content_start = meta_end
        else:
            content_start = 0

        clean_body = body_source[content_start:].strip()
        actual_body_start = body_source.find(clean_body, content_start) if clean_body else content_start
        actual_body_end = actual_body_start + len(clean_body)

        # 3. Instruction routing & Stem / Reference separation
        paragraphs = [p.strip() for p in clean_body.split('\n\n') if p.strip()]
        has_instruction = len(paragraphs) > 1 and is_instruction_text(paragraphs[0])

        stem_spans_list: List[Tuple[int, int]] = []
        ref_spans_list: List[Tuple[int, int]] = []

        if has_instruction:
            inst_text = paragraphs[0]
            inst_s = raw_text.find(inst_text, actual_body_start)
            inst_e = inst_s + len(inst_text)

            rest_text = "\n\n".join(paragraphs[1:])
            rest_s = raw_text.find(rest_text, inst_e)

            # Check if rest has scenario vs question
            qmark_idx = rest_text.rfind('?')
            if qmark_idx != -1 and len(rest_text) > max_stem_chars:
                # Instruction + Scenario in Reference
                before_q = rest_text[:qmark_idx]
                sent_bounds = [m.end() for m in re.finditer(r'(?<=[.!?])\s+', before_q)]
                split_pt = sent_bounds[-1] if sent_bounds else len(before_q)
                scen_text = rest_text[:split_pt].strip()
                stem_t = rest_text[split_pt:].strip()

                scen_s = raw_text.find(scen_text, rest_s)
                scen_e = scen_s + len(scen_text)
                stem_s = raw_text.find(stem_t, scen_e)
                stem_e = stem_s + len(stem_t)

                spans.append((inst_s, inst_e, "reference"))
                spans.append((scen_s, scen_e, "reference"))
                spans.append((stem_s, stem_e, "stem"))
                ref_spans_list.extend([(inst_s, inst_e), (scen_s, scen_e)])
                stem_spans_list.append((stem_s, stem_e))
            else:
                # Instruction-only: stays in stem
                spans.append((actual_body_start, actual_body_end, "stem"))
                stem_spans_list.append((actual_body_start, actual_body_end))
        else:
            # Standard question separation
            if len(clean_body) <= max_stem_chars:
                # Check for scenario + question
                q_idx = clean_body.rfind('?')
                if q_idx != -1:
                    before_q = clean_body[:q_idx]
                    sent_bounds = [m.end() for m in re.finditer(r'(?<=[.!?])\s+', before_q)]
                    if sent_bounds and len(clean_body[:sent_bounds[-1]].strip()) >= 80:
                        split_pt = sent_bounds[-1]
                        scen_t = clean_body[:split_pt].strip()
                        stem_t = clean_body[split_pt:].strip()
                        scen_s = actual_body_start
                        scen_e = scen_s + len(scen_t)
                        stem_s = raw_text.find(stem_t, scen_e)
                        stem_e = stem_s + len(stem_t)
                        spans.append((scen_s, scen_e, "reference"))
                        spans.append((stem_s, stem_e, "stem"))
                        ref_spans_list.append((scen_s, scen_e))
                        stem_spans_list.append((stem_s, stem_e))
                    else:
                        spans.append((actual_body_start, actual_body_end, "stem"))
                        stem_spans_list.append((actual_body_start, actual_body_end))
                else:
                    spans.append((actual_body_start, actual_body_end, "stem"))
                    stem_spans_list.append((actual_body_start, actual_body_end))
            else:
                # Body > 500 chars -> isolate question sentence to stem
                q_idx = clean_body.rfind('?')
                if q_idx != -1:
                    before_q = clean_body[:q_idx]
                    sent_bounds = [m.end() for m in re.finditer(r'(?<=[.!?])\s+', before_q)]
                    if sent_bounds:
                        split_pt = sent_bounds[-1]
                        scen_t = clean_body[:split_pt].strip()
                        stem_t = clean_body[split_pt:].strip()
                        scen_s = actual_body_start
                        scen_e = scen_s + len(scen_t)
                        stem_s = raw_text.find(stem_t, scen_e)
                        stem_e = stem_s + len(stem_t)
                        spans.append((scen_s, scen_e, "reference"))
                        spans.append((stem_s, stem_e, "stem"))
                        ref_spans_list.append((scen_s, scen_e))
                        stem_spans_list.append((stem_s, stem_e))
                    else:
                        # Entire sentence is the task; cannot split without moving task into reference
                        spans.append((actual_body_start, actual_body_end, "stem"))
                        stem_spans_list.append((actual_body_start, actual_body_end))
                else:
                    spans.append((actual_body_start, actual_body_end, "stem"))
                    stem_spans_list.append((actual_body_start, actual_body_end))

        # Build gap-free partition
        segments = build_gap_free_partition(raw_text, spans)
        seg_by_span = {(s["start"], s["end"]): s["id"] for s in segments}

        meta_ids = [s["id"] for s in segments if s["kind"] == "metadata"]
        stem_ids = [seg_by_span[sp] for sp in stem_spans_list if sp in seg_by_span]
        ref_ids = [seg_by_span[sp] for sp in ref_spans_list if sp in seg_by_span]

        # Options
        option_entries = []
        for idx, (os, oe) in enumerate(options_data):
            opt_id = seg_by_span.get((os, oe))
            if opt_id:
                option_entries.append({"id": f"opt_{idx+1}", "segment_ids": [opt_id]})

        option_groups = []
        if option_entries:
            option_groups.append({
                "id": "og_1",
                "kind": "choice_bank",
                "options": option_entries
            })

        # Calculate stem length
        actual_stem_text = "".join(raw_text[s["start"]:s["end"]] for s in segments if s["id"] in stem_ids)
        disposition = "proposed"
        if len(actual_stem_text) > max_stem_chars:
            disposition = "needs_review"
            issues.append({"code": "stem_limit_risk", "segment_ids": stem_ids, "item_ids": [source_id]})

        response_kind = "single_choice" if option_entries else "short_answer"

        item = {
            "id": source_id,
            "parent_group_id": None,
            "response_kind": response_kind,
            "source_label_segment_ids": meta_ids,
            "stem_segment_ids": stem_ids,
            "reference_segment_ids": ref_ids,
            "option_groups": option_groups,
            "matching_rows": [],
            "response_template_segment_ids": [],
            "depends_on_item_ids": [],
            "context_reuse": [],
            "disposition": disposition
        }

        status = "needs_review" if issues else "proposed"
        return {
            "source_id": source_id,
            "status": status,
            "segments": segments,
            "items": [item],
            "issues": issues
        }

    @classmethod
    def decompose(
        cls,
        raw_text: str,
        question_id: str = "custom",
        domain: str = "Clinical Nursing",
        config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Parses raw_text into Output Contract, validates it, and slices platform items.
        """
        contract = cls.parse_to_contract(raw_text, source_id=question_id, config=config)
        valid, errors = ContractValidator.validate(raw_text, contract, expected_source_id=question_id, config=config)
        if not valid:
            print(f"[Contract Warning for {question_id}]: {errors}")

        return contract_to_platform_items(contract, raw_text, domain=domain)
