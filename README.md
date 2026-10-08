# Medical Assessment Item Stem Isolation & Reviewer Engine

> **Deterministic medical assessment question parser and layout engine combining AI semantic boundary identification with Python verbatim string slicing.**  
> Built for psychometric item-writing standards, 500-character database constraints, dual-layout student UI rendering, and **100% exact text preservation (0% character loss)**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Test Suite](https://img.shields.io/badge/tests-all%20passing-brightgreen.svg)]()
[![Zero Text Loss](https://img.shields.io/badge/text%20preservation-100%25-success.svg)]()
[![Stems Under 500c](https://img.shields.io/badge/stems%20%E2%89%A4%20500c-100%25-success.svg)]()

---

## ⚡ Quickstart in a New Session

When pulling this repository in a fresh workspace or asking an AI assistant to review questions:

```bash
# 1. Clone the repository
git clone https://github.com/trollmasterindia/medical-question-stem-isolation.git
cd medical-question-stem-isolation

# 2. Run the review pipeline on the benchmark question set
python3 review.py --open

# 3. Run unit tests
python3 -m unittest discover -s tests
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

### Execution Modes
- `--mode auto` (Default): Uses the autonomous, content-driven semantic boundary engine to parse questions independently with ZERO hardcoded question IDs.
- `--mode ai`: Connects to live LLMs (e.g. Gemini 2.5 Flash) via `google.genai`, directly loading system instructions and schema from `prompts/extraction_prompt.md`.
- `--mode benchmark`: Evaluates benchmark items against annotated dataset spans.

```bash
# Run with live Gemini model using extraction prompt
python3 review.py --mode ai --api-key YOUR_GEMINI_KEY

# Run benchmark evaluation
python3 review.py --mode benchmark
```

---

## 💡 How to Ask an AI Assistant in a New Session

Simply prompt the agent:
> *"I have cloned this repo. Please review these questions [provide IDs, question text, or files] using `review.py`. Ensure stems are <= 500 characters, reference is properly assigned, and show me the Before vs After HTML output and JSON."*

The pipeline loads [`prompts/extraction_prompt.md`](file:///Users/nipunmehra/.gemini/antigravity-ide/scratch/medical-question-stem-isolation/prompts/extraction_prompt.md) and executes [`review.py`](file:///Users/nipunmehra/.gemini/antigravity-ide/scratch/medical-question-stem-isolation/review.py).

---

## 🏗️ Architecture: "AI Eyes, Python Scissors"

To guarantee zero text hallucination, zero character loss, and 100% reproducible parsing, semantic analysis is strictly decoupled from string extraction:

```
[Raw Question Text (Verbatim Input)]
                 │
                 ▼
       AI: Semantic Boundary Detection
    • Reads question and identifies role spans [start, end]
    • Classifies components: [Reference] [Stem] [Option] [Metadata]
    • Flags multi-part subquestions and exhibits
                 │
                 ▼
       Python: Deterministic String Slicing
    • Executes: raw_text[stem_start:stem_end]
    • Strictly isolates stem (<= 500 characters)
    • Allocates reference and distractors
                 │
                 ▼
       Multiset Character Verification
    • Frequency count analysis: Counter(raw) == Counter(extracted)
    • Asserts: Characters Added == 0, Characters Removed == 0
                 │
                 ▼
       Interactive Before vs After Viewer & JSON Export
    • docs/index.html (Interactive layout comparator with JSON modals)
    • data/review_output.json (Complete machine-readable JSON)
```

---

## 📐 Layout & Routing Rules

### 1. The 500-Character Stem Constraint
- Every item stem **must be $\le 500$ characters**.
- Patient vignettes, histories, lab tables, and clinical exhibits are extracted to `reference`.

### 2. Dual Student UI Layout
- **Split-Screen Layout (when `reference` exists):**
  - **Left Pane:** Clinical scenario / exhibit / protocol.
  - **Right Top:** Focused question stem ($\le 500$ characters).
  - **Right Bottom:** Distractor options (A, B, C, D).
- **Single-Column Layout (when `reference` is empty):**
  - **Top:** Question stem ($\le 500$ characters).
  - **Bottom:** Distractor options.

### 3. The Instruction Routing Rule
- **Scenario-Linked Instructions (e.g. Q003, Q095):** When an instruction introduces background materials (e.g., *"Consider the following scenario..."*), it is placed in the **Reference** pane above the vignette.
- **Instruction-Only Items (e.g. Q063):** When a question has a test-taking instruction (e.g., *"Unless instructed otherwise, choose ALL correct answers..."*) but **NO reference scenario or exhibits**, the instruction remains with the **Stem** (total 136 characters). This avoids creating an artificial empty split screen and renders cleanly in a single-column layout.

### 4. Multi-Part Subquestion Decomposition (e.g. Q080, Q082, Q073, Q074, Q084, Q085)
- Parent clinical case studies containing sub-parts `(a)`, `(b)` or `1.`, `2.` are decomposed into **independent platform items** (e.g. `Q080-a`, `Q080-b`).
- Common clinical guidelines are inherited into each subquestion's reference.
- Progressive disclosures or specific patient exhibits (e.g., 17-kg vs 14-kg patient) attach only to their respective subquestion.

---

## 🖥️ Interactive Before vs After Viewer Features

The viewer located at [`docs/index.html`](file:///Users/nipunmehra/.gemini/antigravity-ide/scratch/medical-question-stem-isolation/docs/index.html) provides:
1. **Instant Search & Filters:**
   - Search by Question ID or clinical keyword.
   - Filter pills: `All`, `Split-Screen (With Reference)`, `Single-Page (Stem Only)`, and `Subquestions`.
2. **Side-by-Side Before vs After Layouts:**
   - **Before:** Shows the unparsed input item (giant single text block, character count badge).
   - **After:** Shows the isolated student exam interface (split-screen or single-column).
3. **Question-Level JSON Inspection:**
   - Click **"View JSON"** on any question card to open a modal with formatted JSON.
   - 1-click **"Copy JSON"** button to copy directly to your clipboard.
4. **Export All JSON:**
   - Download the complete verified [`data/review_output.json`](file:///Users/nipunmehra/.gemini/antigravity-ide/scratch/medical-question-stem-isolation/data/review_output.json) directly from the header.

---

## 📂 Repository Structure

```
├── README.md                      # Complete documentation & usage guide
├── review.py                      # Root CLI reviewer tool
├── prompts/
│   └── extraction_prompt.md       # Exact LLM system instructions & schema
├── src/
│   ├── prompt_loader.py           # Loads & parses prompts/extraction_prompt.md (system prompt + schema)
│   ├── ai_boundary_detector.py    # AI boundary detector (Gemini/LLM integration + offline fallback)
│   ├── semantic_parser.py         # Autonomous content-driven boundary parser (zero hardcoded IDs)
│   ├── stem_isolator.py           # Python scissors: string slicing & multiset verification
│   ├── question_processor.py      # Core parser, rule evaluator & subquestion decomposer
│   └── html_generator.py          # Modern Before vs After HTML viewer generator
├── data/
│   ├── review_output.json         # Standardized, verified JSON output for all questions
│   ├── expected_splits.json       # Benchmark spans for 100 medical items
│   ├── inputs/                    # 100 raw benchmark input text files (Q001.txt ... Q100.txt)
│   └── stem_isolation_comparison.html
├── docs/
│   └── index.html                 # Interactive comparison viewer (GitHub Pages ready)
└── tests/
    └── test_stem_isolation.py     # Automated test suite (zero text loss, <= 500c stems)
```

---

## 📄 License

Distributed under the MIT License. Clinical teaching questions are derived from published educational open-access resources (WisTech Open/Open RN and OpenStax under CC BY 4.0).
