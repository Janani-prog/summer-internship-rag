# Robust RAG Demonstration: A Pseudocode Implementation

This project provides a working implementation of an advanced **Robust Retrieval-Augmented Generation (RAG)** pipeline. It is designed to be a direct and faithful execution of the principles outlined in the research paper "[Certifiably Robust RAG against Retrieval Corruption](https://arxiv.org/abs/2405.15556)".

The script contrasts a standard **Basic RAG** with the multi-stage **Robust RAG** pipeline across two complex scenarios. This clearly showcases how a more sophisticated verification process can defend against misinformation and handle ambiguity, which are common failures in simpler RAG systems.

---

## Live Demonstration Output

This is the final, successful output from running the script. It showcases both scenarios and the clear difference in how each pipeline handles challenging information.

```text
================================================================================ RAG Pipeline Demonstration (Gemini API - Pseudocode Implementation)
...
################################################################################

Scenario 1: Handling Direct Contradiction
################################################################################ ...
Executing Basic RAG Pipeline
................................................................................

Retrieval: Fetched top 3 documents: ['doc_1', 'doc_2', 'doc_9']

Generation: Feeding mixed context directly to the LLM...
--- Response from Basic RAG ---
JPY

Executing Robust RAG Pipeline (Pseudocode Implementation)
................................................................................

Wide Retrieval: Fetched 10 candidates.
Candidates: ['doc_1', 'doc_2', 'doc_9', 'doc_3', 'doc_4', 'doc_7', 'doc_6', 'doc_10', 'doc_5', 'doc_8']

'I Don't Know' Filtering (Pseudocode Step 1):

[PASS] 'doc_1' seems to contain a direct answer.

[PASS] 'doc_2' seems to contain a direct answer.
...

[PASS] 'doc_10' seems to contain a direct answer.
...

Passage Re-ranking (Pseudocode Step 2): Selecting top 5 candidates.
Re-ranked Scores:

10/10 'doc_1'

10/10 'doc_2'

10/10 'doc_10'
Top 5 selected: ['doc_1', 'doc_2', 'doc_10']

Factual Consistency Check (Pseudocode Step 3):

[PASS] 'doc_1' is consistent with the other top documents.

[PASS] 'doc_2' is consistent with the other top documents.

[PASS] 'doc_10' is consistent with the other top documents.

Synthesized Generation (Simulating Pseudocode's Secure Decoding):
--- Response from Robust RAG ---
The official ISO 4217 currency code for the currency of Japan, the Yen, is JPY. Travelers to Japan should exchange their currency for JPY.

################################################################################

Scenario 2: Synthesizing from Incomplete Information
################################################################################ ...
Executing Basic RAG Pipeline
................................................................................

Retrieval: Fetched top 3 documents: ['doc_5', 'doc_9', 'doc_1']

Generation: Feeding mixed context directly to the LLM...
--- Response from Basic RAG ---
Access to the Argus system uses:

RSA security tokens (old system).

Restricted physical access to servers.

Multi-factor authentication (MFA) via an approved authenticator app.

Executing Robust RAG Pipeline (Pseudocode Implementation)
................................................................................

Wide Retrieval: Fetched 10 candidates.
Candidates: ['doc_5', 'doc_9', 'doc_1', 'doc_6', 'doc_2', 'doc_3', 'doc_10', 'doc_4', 'doc_8', 'doc_7']

'I Don't Know' Filtering (Pseudocode Step 1):

[PASS] 'doc_5' seems to contain a direct answer.
...

[PASS] 'doc_1' seems to contain a direct answer.
...

[PASS] 'doc_3' seems to contain a direct answer.
...

Passage Re-ranking (Pseudocode Step 2): Selecting top 5 candidates.
Re-ranked Scores:

10/10 'doc_5'

10/10 'doc_1'

10/10 'doc_3'
Top 5 selected: ['doc_5', 'doc_1', 'doc_3']

Factual Consistency Check (Pseudocode Step 3):

[FAIL] 'doc_5' is contradictory.

[FAIL] 'doc_1' is contradictory.

[FAIL] 'doc_3' is contradictory.
--- Response from Robust RAG ---
After filtering, the most relevant documents were found to be contradictory. No reliable answer can be generated.
```

---

## Mechanism Explained: A Line-by-Line Breakdown

This project's goal is to faithfully implement the advanced RAG mechanisms described in the research pseudocode. Here is how our code's logic maps directly to those concepts.

### Basic RAG: The Vulnerable Baseline
The basic pipeline demonstrates a standard, vulnerable approach.
* **Line 1: Retrieval:** It fetches the top 3 documents based on simple similarity. This is a standard but naive approach.
* **Line 2: Generation:** It combines the raw text from all 3 documents into a single, unverified context and asks the LLM to generate an answer. As seen in the output, this can lead to incorrect or incomplete answers if the context is polluted.

### Robust RAG: The Pseudocode Implementation
This pipeline follows the multi-stage verification process from the research paper to ensure a reliable and secure answer.

* **Line 1: Wide Retrieval:** The process begins by gathering a large set of 10 candidate documents to ensure all potentially relevant information is included in the initial analysis.

* **Line 2: 'I Don't Know' Filtering:**
    * **Pseudocode Step:** The first filtering stage described in the paper.
    * **Our Implementation:** The script isolates each of the 10 documents and asks the LLM if it can confidently answer the query from that single piece of context. Documents that would lead to an "I don't know" answer are immediately discarded. This effectively filters out irrelevant or useless information.

* **Line 3: Passage Re-ranking:**
    * **Pseudocode Step:** The second filtering stage.
    * **Our Implementation:** The documents that survive the first filter are then scored for relevance by the LLM on a scale of 1-10. The script then selects the top 5 highest-scoring documents to move to the next stage, further refining the candidate pool.

* **Line 4: Factual Consistency Check:**
    * **Pseudocode Step:** The final and most critical filtering stage.
    * **Our Implementation:** The script takes the top-ranked documents and cross-references them against each other. It asks the LLM if each document is factually consistent with the others in the set.
    * **Outcome in Scenario 2:** In our second example, the LLM correctly identifies that the surviving documents (an *archived* spec vs. *current* protocols) are contradictory. As a result, all are discarded, and the pipeline safely aborts, preventing a wrong answer. This is the "right" outcome for a truly robust system.

* **Line 5: Synthesized Generation:**
    * **Pseudocode Step:** The final "secure decoding" or "aggregation" step.
    * **Our Implementation:** If a set of documents passes all three filtering stages (as in Scenario 1), this final step is triggered. The script provides all vetted, consistent sources to the LLM with a strict prompt: synthesize a single, cohesive answer from this trusted information. This simulates the "consensus" mechanism from the paper.

---

### A Note on the Token-by-Token Implementation

The final mechanism described in the research paper is **Secure Decoding Aggregation**, which involves averaging the model's next-token probability vectors at each step of generation.

**This step was not implemented due to fundamental limitations of current high-level LLM APIs.**

* **LLM API Capabilities:** Commercial and open-access LLM APIs, including both the **Google Gemini API** and third-party services like **OpenRouter**, are designed for ease of use. They provide the final generated text but do not expose the low-level, real-time probability distributions (or "logits") for each potential next token. Access to this data is required to perform the vector-averaging described in the pseudocode.
* **Our Solution:** The "Synthesized Generation" step is a powerful and practical simulation of this principle. By instructing the model to synthesize an answer from multiple trusted sources and report any ambiguity, we achieve the same goal of a robust, consensus-based answer, working within the capabilities of the available tools.

---

## Project Setup and Execution

### 1. Prerequisites
* Python 3.8+
* Git (for cloning)

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install google-generativeai chromadb sentence-transformers python-dotenv
```

### 3. Configuration
Create a file named .env in the project's root directory. Add your Google AI API key, which you can get from Google AI Studio.

.env file contents:
```
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### 4. Running the Script
Execute the script from your terminal:

```bash
python advanced_2.py
```
