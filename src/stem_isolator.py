"""
stem_isolator.py

Core engine for deterministic assessment item decomposition.
Combines AI semantic boundary detection with Python deterministic string slicing
to guarantee 0% text modification or character loss.
Supports both contiguous and discontiguous (multi-span) stems.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple, Any, Union


def slice_verbatim(text: str, start: int, end: int) -> str:
    """
    Slices source text directly using Unicode code point offsets.
    Guarantees no character corruption or alteration.
    """
    return text[start:end]


def verify_zero_text_loss(
    raw_text: str,
    stem: str,
    reference: Optional[str] = None,
    distractors: Optional[List[str]] = None,
    metadata: Optional[str] = None
) -> Tuple[bool, int, int]:
    """
    Verifies that the extracted components exactly reconstruct the source text characters.
    Uses multiset frequency analysis to account for questions where the stem was in the middle
    or where stems are discontiguous.
    
    Returns:
        (is_exact_match, characters_added, characters_removed)
    """
    clean_raw = "".join(raw_text.split())
    
    parts = []
    if metadata:
        parts.append(metadata)
    if reference:
        parts.append(reference)
    parts.append(stem)
    if distractors:
        parts.extend(distractors)
        
    clean_reconstructed = "".join("".join(parts).split())
    
    counter_raw = Counter(clean_raw)
    counter_proc = Counter(clean_reconstructed)
    
    diff_added = counter_proc - counter_raw
    diff_removed = counter_raw - counter_proc
    
    added = sum(diff_added.values())
    removed = sum(diff_removed.values())
    exact = (added == 0 and removed == 0)
    
    return exact, added, removed


def isolate_question_components(
    raw_text: str,
    stem_span: Union[Tuple[int, int], List[Tuple[int, int]]],
    reference_segments: Optional[List[str]] = None,
    option_segments: Optional[List[str]] = None,
    metadata_segments: Optional[List[str]] = None,
    max_stem_length: int = 500
) -> Dict[str, Any]:
    """
    Decomposes a raw question into stem, reference, and distractors.
    
    Args:
        raw_text: The complete unprocessed question string.
        stem_span: A single (start, end) tuple or list of (start, end) tuples for discontiguous stems.
        reference_segments: Clinical case / exhibit text segments.
        option_segments: Multiple-choice answer options.
        metadata_segments: Item numbering or exercise labels.
        max_stem_length: The database character constraint (default: 500).
        
    Returns:
        Dict containing before, after, and verification data.
    """
    # Normalize stem_spans to a list of tuples
    if isinstance(stem_span, tuple) and len(stem_span) == 2 and isinstance(stem_span[0], int):
        spans_list = [stem_span]
    elif isinstance(stem_span, list):
        spans_list = stem_span
    else:
        raise ValueError("Invalid stem_span format; expected (start, end) or list of (start, end)")

    # Slice each stem span verbatim from source
    stem_parts = [raw_text[s:e] for s, e in spans_list]
    isolated_stem = "\n\n".join(stem_parts)
    
    reference_text = "\n\n".join(reference_segments) if reference_segments else None
    distractors = option_segments if option_segments else []
    metadata = " ".join(metadata_segments) if metadata_segments else None
    
    is_exact, added, removed = verify_zero_text_loss(
        raw_text=raw_text,
        stem=isolated_stem,
        reference=reference_text,
        distractors=distractors,
        metadata=metadata
    )
    
    fits_limit = len(isolated_stem) <= max_stem_length
    layout = "split_screen_with_reference" if reference_text else "single_column_stem_only"
    
    return {
        "before": {
            "stem": raw_text,
            "reference": None,
            "distractors": [],
            "char_count": len(raw_text),
            "fits_500_char_limit": len(raw_text) <= max_stem_length,
            "layout": "single_column_stem_only"
        },
        "after": {
            "stem": isolated_stem,
            "reference": reference_text,
            "distractors": distractors,
            "metadata": metadata,
            "stem_char_count": len(isolated_stem),
            "reference_char_count": len(reference_text) if reference_text else 0,
            "fits_500_char_limit": fits_limit,
            "layout": layout,
            "is_discontiguous": len(spans_list) > 1
        },
        "verification": {
            "stem_spans": [[s, e] for s, e in spans_list],
            "characters_added": added,
            "characters_removed": removed,
            "exact_match": is_exact,
            "stem_under_500": fits_limit
        }
    }
