# Robust RAG Pipeline — Two Implementations

This repository contains two implementations of a **Robust Retrieval-Augmented Generation (RAG)** pipeline, built as part of an internship project exploring the limitations of standard RAG and how to defend against them.

Both implementations contrast a **Basic RAG** against a **Robust RAG** pipeline across multiple scenarios. The goal is not production-readiness — it is understanding. Specifically: why naive RAG fails, what trustworthy retrieval actually requires, and how the research translates into working code.

The foundation for both implementations is the research paper:
> **"Certifiably Robust RAG against Retrieval Corruption"** — arxiv 2405.15556

---

## The Problem with Standard RAG

A typical RAG pipeline does two things: retrieve the most semantically similar documents to a query, then feed them directly to the LLM and ask it to answer. It is simple, fast, and dangerously naive.

It places complete trust in whatever it retrieves — making it vulnerable to outdated, mismatched, or adversarial content. Both implementations demonstrate this with a stress-test scenario where a malicious or contradictory document is deliberately planted in the knowledge base.

---

## Implementation 1 — Custom Verification Pipeline

**File:** `gemini.py`

This implementation takes a practical, custom approach to the problem. Rather than following the paper's pseudocode directly, it builds a verification layer around three ideas: how relevant is this document, how trustworthy is it based on its metadata, and what happens when nothing is good enough.

### How it works

**Wide Retrieval** — Instead of fetching the top 3 documents, a larger candidate pool is retrieved to ensure the correct source has a chance to surface.

**Relevance Scoring** — The LLM acts as a judge, scoring each document on how well it genuinely answers the query on a scale of 1–10. Not just semantic similarity — actual answer quality.

**Trust Scoring** — Each document's metadata is interrogated:
- `status`: active or obsolete?
- `audience`: intended for the right group?
- `version`: how recent?

These produce a trust score that runs parallel to relevance.

**Combined Scoring & Selection** — The two scores are weighted and combined. The top document becomes the sole source of truth. One source, explicitly chosen, for a reason you can audit.

**Fallback Mechanism** — If the top combined score still falls below a defined confidence threshold (default: 7.0), the system discards all documents and falls back to general knowledge — flagging that it did so. Explicitly, not silently.

### Deliverables

- Console output with real-time scoring tables and decisions
- `rag_comparative_analysis.xlsx` — a full audit trail exported to Excel, with separate sheets per scenario and a complete breakdown of every document's scores and the final decision

### Scenarios

| Scenario | Query | Key Result |
|---|---|---|
| 1 — Conflicting Info | Spending limit for a senior manager's client dinner | Malicious memo outranked and ignored; correct policy selected |
| 2 — Knowledge Gap | Process for returning a laptop after a project ends | No relevant documents found; fallback triggered |

---

## Implementation 2 — Pseudocode Implementation

**File:** `advanced_2.py`

This implementation follows the research paper's pseudocode directly and faithfully. Each stage maps to a specific step described in the paper.

### How it works

**Wide Retrieval** — 10 candidate documents are fetched to ensure comprehensive coverage.

**Stage 1 — "I Don't Know" Filtering** — Each document is tested in isolation: can it answer the query on its own, or does it lead to an "I don't know"? Documents that cannot stand alone are immediately discarded.

**Stage 2 — Passage Re-ranking** — Surviving documents are scored for relevance by the LLM on a scale of 1–10. The top 5 highest-scoring documents are selected for the next stage.

**Stage 3 — Factual Consistency Check** — The top-ranked documents are cross-referenced against each other. The LLM checks whether each document is factually consistent with the rest of the set. Contradictory documents are discarded.

**Synthesized Generation** — If documents pass all three stages, the LLM synthesizes a single cohesive answer from the vetted, consistent sources. This simulates the paper's consensus-based "secure decoding" mechanism.

### A note on secure decoding aggregation

The final mechanism described in the paper — Secure Decoding Aggregation — involves averaging next-token probability vectors across multiple model outputs at each generation step. This was not implemented because current LLM APIs, including the Gemini API, do not expose the raw token-level probability distributions (logits) required to perform this operation. The synthesized generation step is a practical approximation of the same principle.

### Scenarios

| Scenario | Query | Key Result |
|---|---|---|
| 1 — Direct Contradiction | ISO currency code for Japan | Consistent documents identified; correct answer synthesized |
| 2 — Incomplete Information | Access protocol for internal system | All surviving documents contradictory; system correctly refuses to answer |

---

## Project Setup

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for Implementation 1
pip install google-generativeai chromadb sentence-transformers python-dotenv pandas openpyxl

# Install dependencies for Implementation 2
pip install google-generativeai chromadb sentence-transformers python-dotenv
```

### Configuration

Create a `.env` file in the project root:

```
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

Get your key from [Google AI Studio](https://aistudio.google.com).

### Running

```bash
# Implementation 1
python gemini.py

# Implementation 2
python advanced_2.py
```

---

## What This Is and What It Isn't

These are demonstration scripts, not production systems. The trust scoring weights are manually defined. The LLM judge is non-deterministic without temperature control. The sleep timers exist to manage rate limits. There are frameworks that handle all of this far more robustly and at scale.

The point was never the output. It was understanding why naive RAG fails, what trustworthy retrieval actually means, and how the ideas in a research paper translate — or don't — into working code given real API constraints.

That understanding is the deliverable.
