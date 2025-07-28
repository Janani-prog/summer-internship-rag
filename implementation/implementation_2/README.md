## Live Demonstration Output

This is the final, successful output from running the script. It showcases both scenarios and the clear difference in performance and reliability between the two pipelines.

```
================================================================================
RAG Pipeline Demonstration (Gemini API): Basic vs. Robust 
2025-07-28 11:51:46,468 - INFO - Google AI client initialized for model: gemini-1.5-flash-latest
...

################################################################################
# Scenario 1: Spending Limit Query
################################################################################

Executing Basic RAG Pipeline
................................................................................

1. Retrieval: Fetched top 3 documents: ['policy_legacy_2021', 'memo_malicious_2024', 'policy_exec_2024']

2. Generation: Feeding mixed context directly to the LLM...
--- Response from Basic RAG ---
$75

Executing Robust RAG Pipeline
................................................................................

1. Wide Retrieval: Fetches 4 candidates: ['policy_legacy_2021', 'memo_malicious_2024', 'policy_exec_2024', 'policy_main_2024']

2. Relevance & Trust Scoring:

3. Final Selection Analysis:
   Rank | Document ID              | Relevance | Trust   | Combined
   ----- | ----------------------- | --------- | ------- | ---------
   1     | policy_main_2024        | 10        | 25.5    | 14.65
   2     | memo_malicious_2024     | 10        | 10.6    | 10.18
   3     | policy_exec_2024        | 3         | 10.2    | 5.16
   4     | policy_legacy_2021      | 9         | -8.0    | 3.90
   ----- | ----------------------- | --------- | ------- | ---------

4. Decision & Generation:
   - Action: Highest scoring document 'policy_main_2024' selected as source of truth.
--- Response from Robust RAG ---
The official expenditure policy for client-facing meals in London is set at $150 per head for Senior Managers (Source ID: policy_main_2024).

Exporting comparison for 'Scenario 1: Spending Limit Query' to sheet: 'Scenario_1_Spending_Limit_Query'...
Successfully exported comparison report.

################################################################################
# Scenario 2: Hardware Failure Query
################################################################################

Executing Basic RAG Pipeline
................................................................................

1. Retrieval: Fetched top 3 documents: ['offboarding_policy', 'onboarding_hr_2025', 'onboarding_it_2025']

2. Generation: Feeding mixed context directly to the LLM...
--- Response from Basic RAG ---
The provided text does not contain information about the policy for getting a new laptop after a hardware failure.

Executing Robust RAG Pipeline
................................................................................

1. Wide Retrieval: Fetches 4 candidates: ['offboarding_policy', 'onboarding_hr_2025', 'onboarding_it_2025', 'laptop_options_draft']

2. Relevance & Trust Scoring:

3. Final Selection Analysis:
   Rank | Document ID              | Relevance | Trust   | Combined
   ----- | ----------------------- | --------- | ------- | ---------
   1     | onboarding_it_2025      | 1         | 11.6    | 4.18
   2     | onboarding_hr_2025      | 1         | 11.5    | 4.15
   3     | offboarding_policy      | 1         | 9.0     | 3.40
   4     | laptop_options_draft    | 1         | -18.3   | -4.79
   ----- | ----------------------- | --------- | ------- | ---------

4. Decision & Generation:
   - Action: No single document is relevant enough (top score 4.18 < 6.0). Triggering Fallback.
--- Response from Robust RAG ---
The policy for getting a new laptop after a hardware failure typically involves...
...[Detailed general knowledge answer]...

Exporting comparison for 'Scenario 2: Hardware Failure Query' to sheet: 'Scenario_2_Hardware_Failure_Que'...
Successfully exported comparison report.

Demonstration complete. Check 'rag_comparative_analysis.xlsx' for detailed reports.
```

---

## Mechanism Explained: Basic vs. Robust RAG

This project contrasts two different RAG methodologies.

### Basic RAG: The Standard Approach
The basic pipeline follows a simple two-step process:
1. **Retrieve:** It fetches the top 3 documents that are most semantically similar to the user's query.
2. **Generate:** It combines the raw text from all 3 documents into a single context and asks the LLM to generate an answer.

This approach is fast but vulnerable. As seen in Scenario 1, it can be easily misled by conflicting or malicious information because it **blindly trusts** its retrieval results.

### Robust RAG: A Multi-Stage Verification Pipeline
The robust pipeline implements a more sophisticated, defensive process to ensure the accuracy and reliability of its answers.

1. **Wide Retrieval:** It starts by fetching a larger set of 5 candidate documents to ensure the correct source is likely included in the analysis pool.

2. **Relevance Scoring (LLM as a Judge):** This is the first critical filter. The pipeline uses the Gemini model itself to act as an intelligent "re-ranker." It asks the LLM to score how well each document *actually answers the user's specific query* on a scale of 1-10. Our prompt is strict, instructing the model to penalize documents that are only partially relevant.

3. **Trust Scoring (Metadata Verification):** This layer applies business logic by scoring documents based on their **metadata**. Think of this as checking a document's "ID badge" for authenticity. It verifies:
   * **Status:** Is the document `active` or `obsolete`?
   * **Audience:** Is it intended for the correct audience (e.g., `senior_managers`)?
   * **Version:** Is it a recent document?
   
   This step is crucial for identifying the most *authoritative* source.

4. **Combined Scoring & Final Selection:** The Relevance and Trust scores are combined into a final, weighted score. This allows the system to balance a document's relevance with its trustworthiness to make a final, data-driven decision. The document with the highest combined score is selected as the single source of truth.

5. **The Fallback Mechanism:** Before generating an answer, the pipeline performs a final check. If the combined score of the best document is still below a set confidence threshold (`6.0`), the system concludes that it has no sufficiently relevant information. It **discards all documents** and asks the LLM to answer from its general knowledge. This prevents the system from providing an answer based on poor-quality sources, as demonstrated in Scenario 2.

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
pip install google-generativeai chromadb sentence-transformers python-dotenv pandas openpyxl
```

### 3. Configuration
Create a file named `.env` in the project's root directory. Add your Google AI API key, which you can get from Google AI Studio.

`.env` file contents:
```
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### 4. Running the Script
Execute the script from your terminal:

```bash
python advanced2.py
```

---

## Deliverables

### Console Output
The terminal will display a real-time log of the execution for both scenarios, including the retrieval and generation steps, the detailed scoring analysis table, and the final decision made by the Robust RAG pipeline.

### Excel Report: `rag_comparative_analysis.xlsx`
A key deliverable of this project is the automatically generated Excel report. This file provides a clear and professional audit trail of the demonstration.

* **Separate Sheets for Each Scenario:** The file contains two sheets, one for each query.
* **Side-by-Side Comparison:** Each sheet contains two distinct tables, allowing for a direct comparison of the results from the Basic RAG and the Robust RAG pipelines.
* **Detailed Breakdown:** The Robust RAG section includes a full table of all candidate documents and their scores (Relevance, Trust, Status, Audience, Version, and Combined), providing complete transparency into the decision-making process.
