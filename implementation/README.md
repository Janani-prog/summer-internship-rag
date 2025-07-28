# Robust RAG Demonstration

This project provides a working demonstration of a **Robust Retrieval-Augmented Generation (RAG)** pipeline, contrasting it with a standard **Basic RAG** to showcase the importance of multi-stage document verification. The implementation is a practical application of the core principles discussed in the research paper "[Certifiably Robust RAG against Retrieval Corruption](https://arxiv.org/abs/2405.15556)".

The script runs two distinct scenarios to highlight how the Robust RAG pipeline intelligently handles both **conflicting/malicious information** and queries with **no relevant documents**.

---
## The Mechanism of Robust RAG

The fundamental problem with simple RAG systems is their vulnerability to **retrieval corruption**. If malicious, outdated, or irrelevant documents are retrieved, a basic pipeline can be easily misled. Robust RAG is designed to defend against this vulnerability.

The core strategy, as outlined in the research paper, is **"isolate-then-aggregate"**.

### The Ideal "Isolate-then-Aggregate" Strategy (from the Research Paper)

Imagine you ask a critical question. The ideal Robust RAG pipeline works like a skilled manager consulting a team of experts, rather than just an assistant who staples all their reports together.

**1. Isolate (Get Individual Opinions):**
Instead of lumping all retrieved documents together, the system treats each one as an individual expert. It asks the AI the same question based on *each document separately*.

For example, if the query is "What is the name of the highest mountain?" and it retrieves three documents, it will generate three isolated, independent answers:
* Answer from Document 1: "Mount Everest"
* Answer from Document 2 (malicious): "Fuji!"
* Answer from Document 3: "Everest is the highest mountain"

This isolation is critical because it ensures a malicious document can only corrupt its own individual response, without influencing the others.

**2. Aggregate (Find the Consensus):**
Next, the system takes all these individual answers and securely aggregates them to find the most consistent and frequent answer, effectively holding a vote. It identifies "Everest" as the clear consensus and discards "Fuji" as an outlier, leading to the correct and robust final answer.

### Our Practical Implementation

Due to API and resource constraints, our script implements a highly effective and practical version of this strategy. Instead of generating and aggregating full text responses, **it isolates documents for scoring and then aggregates the scores** to select a single, definitive source of truth.

This is how our pipeline maps to the core principle:

| Paper Concept | Our Implementation |
| :--- | :--- |
| **1. Isolate** | **Individual Document Scoring:** Each document is "isolated" and evaluated on its own merit through two distinct filters: **Relevance Scoring** (how well it answers the query) and **Trust Scoring** (is the source metadata authoritative?). |
| **2. Aggregate** | **Combined Score & Selection:** The scores from the isolation phase are "aggregated" into a final combined score. The pipeline then selects the single document with the highest score as the definitive source of truth, effectively finding the "expert" with the best credentials and the most relevant information. |

This implementation successfully achieves the same goal: it makes an informed, defensible decision based on a cross-verification of sources, ensuring the final answer is generated from the most trustworthy and relevant document possible.

---
## Features
* **Side-by-Side Comparison:** Directly contrasts the outcomes of a Basic and a Robust RAG pipeline.
* **Two Scenarios:**
    1.  **Conflicting Information:** Demonstrates how the Robust RAG rejects a malicious-looking memo in favor of an official policy.
    2.  **Irrelevant Information:** Shows the pipeline intelligently identifying that no retrieved documents are relevant and triggering a safe fallback mechanism.
* **Detailed Excel Report:** Automatically generates a `rag_comparative_analysis.xlsx` file with a detailed, side-by-side comparison and scoring breakdown for each scenario, providing a clear audit trail.

---
## Setup & Installation

Follow these steps to set up and run the project.

### 1. Prerequisites
* Python 3.8+
* Access to the Google Gemini API.

### 2. Clone or Download
Clone this repository or download the `final_rag_demonstration_professional.py` file to your local machine.

### 3. Set Up Virtual Environment
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install Dependencies
Install the required Python packages.
```bash
pip install google-generativeai chromadb sentence-transformers python-dotenv pandas openpyxl
```

### 5. Configure Environment Variables
Create a file named .env in the same directory as the script and add your Gemini API key:
```env
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### How to Run
Execute the script from your terminal:
```bash
python final_rag_demonstration_professional.py
```

---
## Understanding the Output

## Understanding the Output

### Console Output
The script will run two scenarios sequentially. For each scenario, it displays a step-by-step execution log for both the Basic and Robust RAG pipelines, including a detailed analysis table for the robust method's scoring.

![Console Output](screenshots)

### Excel Report: `rag_comparative_analysis.xlsx`
This file is the key deliverable for analysis. It contains two sheets, one for each scenario, providing a clear, side-by-side comparison of the pipelines and a detailed scoring breakdown for the robust method.

### Excel Report: rag_comparative_analysis.xlsx
This file is the key deliverable for analysis. It contains two sheets, one for each scenario. Each sheet provides a clear, side-by-side comparison:

- **Basic RAG Analysis:** Shows the documents it retrieved and the final response it gave.
- **Robust RAG Analysis:** Provides a summary of the outcome (document selected or fallback) and its final response, followed by a detailed table breaking down the Relevance, Trust, Status, Audience, and Version scores for every candidate document.

---
## Connection to the Research Paper

This project serves as a practical demonstration of the concepts outlined in **"Certifiably Robust RAG against Retrieval Corruption"** (arXiv:2405.15556v1).

| **Paper Concept**                     | **Our Implementation**                                                                                                                                                                                                                 |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Vanilla RAG**                       | **Basic RAG Pipeline:** Concatenates top-k documents, demonstrating the vulnerability to corruption as described in the paper.                                                                                                         |
| **"Isolate–then–Aggregate" Strategy**| **Multi-Stage Scoring:** Our pipeline embodies this principle by "isolating" each document for individual scoring (Relevance & Trust) before "aggregating" those scores to select a single, definitive source.                      |
| **Retrieval Corruption Attack**        | **Scenario 1:** The malicious memo (`$75`) is a direct example of a retrieval corruption attack, which the Robust RAG successfully defends against.                                                                                    |
| **Handling Irrelevant Information**    | **Fallback Mechanism:** Our pipeline's ability to fallback when scores are low is a practical way to handle cases where benign but irrelevant documents are retrieved, similar to the paper's "I don't know" handling.                |

---
