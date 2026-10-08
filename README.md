# Medical Assessment Item Stem Isolation & Reviewer Engine

> **Production AI boundary detector and layout engine combining Live AI Output Contracts with Python verbatim string slicing.**  
> Built for psychometric item-writing standards, 500-character database constraints, dual-layout student UI rendering, and **exact source order and content preservation (zero text loss)**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Test Suite](https://img.shields.io/badge/tests-all%20passing-brightgreen.svg)]()
[![Zero Text Loss](https://img.shields.io/badge/text%20preservation-order%20verified-success.svg)]()
[![Stems Under 500c](https://img.shields.io/badge/stems%20%E2%89%A4%20500c-strictly%20enforced-success.svg)]()

---

## ⚡ Quickstart in a New Session

When pulling this repository in a fresh workspace or asking an AI assistant to review questions:

```bash
# 1. Clone the repository
git clone https://github.com/trollmasterindia/medical-question-stem-isolation.git
cd medical-question-stem-isolation

# 2. Run the review pipeline on question files using Live AI
# Make sure GEMINI_API_KEY (or GOOGLE_API_KEY) is set in your environment
python3 review.py --open

# 3. Run full unit test suite
python3 -m unittest discover -s tests -v
```

### Reviewing Specific Questions
You can review any specific questions by ID:
```bash
python3 review.py --ids Q001 Q025 Q063 Q080
```

### Reviewing Unseen / Arbitrary Question Files
You can process ANY arbitrary, unannotated medical question file without needing entries in `expected_splits.json`:
```bash
python3 review.py --file path/to/any_question.txt --open
```

### Passing Model and Credentials Consistently
```bash
# Forward model name and API key directly
python3 review.py --model gemini-2.5-flash --api-key YOUR_API_KEY
```

---

## 🏗️ Architecture: Live AI Contract + Python Verbatim Slicing

To guarantee zero hallucination, zero text alteration, and strict adherence to database constraints, the pipeline implements the following architecture:

```
[Immutable Raw Question Text] + [Opaque source_id] + [Configuration]
                            │
                            ▼
              Live AI Boundary Detector
     • Master system instruction: prompts/medical_stem_isolation_system_prompt.txt
     • Returns canonical partition tiling 0..len(raw_text)
     • Never accesses benchmark answer keys or question-specific rules
     • Bounded retry on API error or contract validation failure
                            │
                            ▼
           Deterministic ContractValidator
     • Rejects booleans in offsets, overlapping spans, dangling IDs
     • Ensures gap-free partition covering entire source exactly
     • Enforces controlled issue codes and review routing
                            │
                            ▼
             Python Verbatim Slicing Adapter
     • Derives every clinical text component: raw_text[start:end]
     • Zero synthetic spaces or newlines inserted
     • Verifies character sequence order (clean_raw == clean_proc)
     • Enforces review gating: blocks automatic PASS for needs_review
     • Preserves matching rows, response templates, dependencies, context reuse
                            │
                            ▼
     Interactive Before vs After Viewer & Exact Contract Exports
     • docs/index.html (Interactive layout comparator with JSON modals)
     • data/review_output.json (Platform items with computed metrics)
     • data/master_contract_output.json (Exact model-returned contracts)
```

---

## 📐 Production Principles & Integration Notes

1. **Live AI as Default and Only Production Detector:**  
   Runtime execution invokes `AIBoundaryDetector`. All runtime calls and fallbacks to `AutonomousSemanticParser` are removed. Offline parser code is isolated in `src/experimental/autonomous_parser.py`.
2. **Deterministic Failure & Review Routing:**  
   If API credentials are missing, network calls fail, or the response fails schema validation, the pipeline performs bounded retries. If retries fail, it returns a `needs_review` failure contract retaining the complete original raw text. It never substitutes regex or benchmark answers.
3. **No Leakage:**  
   Inference receives only the immutable raw text, opaque source ID, and config. It has zero access to `expected_splits.json`.
4. **Credential & Secret Protection:**  
   Execution tracks `engine`, `model`, `prompt_hash`, and `source_hash`. API keys and credentials are never written to logs or contract exports.
5. **Exact Contract Export:**  
   The contract exported to `data/master_contract_output.json` is the exact contract returned by the live model.
6. **Order-Aware Verification:**  
   Verification asserts both multiset character frequency equality and exact character sequence order (`clean_raw == clean_proc`). Hardcoded pass claims and constant 100% strings are eliminated; all metrics are computed dynamically.

---

## 📂 Repository Structure

```
├── README.md                                  # Complete documentation & usage guide
├── review.py                                  # CLI reviewer entry point (Live AI default)
├── prompts/
│   ├── medical_stem_isolation_system_prompt.txt # Master system instruction prompt
│   └── extraction_prompt.md                   # Legacy reference schema
├── docs/
│   ├── implementation-notes.txt               # Integration guidelines & requirements
│   └── index.html                             # Interactive comparison viewer
├── src/
│   ├── ai_boundary_detector.py                # Production Live AI detector (retries, hashes, validation)
│   ├── contract_validator.py                  # Strict schema, offset, and tiling validator
│   ├── contract_slicer.py                     # Deterministic Python slicing, order checks, review gating
│   ├── prompt_loader.py                       # Loads master system prompt & default config
│   ├── question_processor.py                  # Batch and single-item orchestration
│   ├── html_generator.py                      # Interactive Before vs After HTML viewer generator
│   ├── stem_isolator.py                       # Low-level slicing utilities
│   └── experimental/
│       └── autonomous_parser.py               # Isolated offline parser (tests/experimental only)
├── data/
│   ├── review_output.json                     # Output items with calculated metrics
│   ├── master_contract_output.json            # Exact model-returned contracts
│   ├── expected_splits.json                   # Isolated evaluation benchmark
│   └── inputs/                                # Raw input text files (Q001.txt ... Q100.txt)
└── tests/
    ├── test_production_ai_pipeline.py         # Requirement 9 unit tests with mock responses
    ├── test_master_contract.py                # Master contract validation and tiling tests
    └── test_stem_isolation.py                 # Core isolation and layout tests
```

---

## 📄 License

Distributed under the MIT License. Clinical teaching questions are derived from published educational open-access resources (WisTech Open/Open RN and OpenStax under CC BY 4.0).

