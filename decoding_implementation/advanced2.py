# final_rag_demonstration_gemini.py

import os
import re
import logging
import time
from typing import List, Dict, Any

# Third-party imports
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import pandas as pd

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
load_dotenv()

# --- CONSTANTS ---
RELEVANCE_THRESHOLD = 6.0 
GEMINI_MODEL = "gemini-2.0-flash-lite"

class RAGDemonstrator:
    """
    Demonstrates and compares a Basic RAG pipeline against a final, perfected
    Robust RAG pipeline that uses multi-stage verification to provide a definitive
    answer or trigger a safe fallback. This version uses the Gemini API.
    """
    def __init__(self):
        self.llm = self._initialize_llm()
        self.db_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
        self.collection = None

    def _initialize_llm(self) -> genai.GenerativeModel:
        """Initializes the Gemini LLM using a Google AI API Key."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        
        genai.configure(api_key=api_key)
        logger.info(f"Google AI client initialized for model: {GEMINI_MODEL}")
        return genai.GenerativeModel(model_name=GEMINI_MODEL)

    def _setup_collection(self, docs: List[Dict]):
        """Resets and populates the ChromaDB collection for a given scenario."""
        try:
            self.db_client.delete_collection(name="rag_scenario_docs")
        except Exception:
            pass
        
        self.collection = self.db_client.create_collection(
            name="rag_scenario_docs", 
            embedding_function=self.embedding_func
        )
        self.collection.add(
            ids=[d['id'] for d in docs], 
            documents=[d['content'] for d in docs], 
            metadatas=[d['metadata'] for d in docs]
        )

    def _llm_call(self, prompt: str) -> str:
        """A simple, non-streaming call for internal logic using the Gemini API."""
        time.sleep(1.2) 
        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            return "Error during generation."

    def demonstrate_basic_rag(self, query: str) -> Dict[str, Any]:
        """Runs a standard RAG pipeline and returns its results."""
        print("\n" + "-"*80)
        print("Executing Basic RAG Pipeline")
        print("."*80)

        retrieved = self.collection.query(query_texts=[query], n_results=3)
        context = "\n\n".join(retrieved['documents'][0])
        doc_ids = retrieved['ids'][0]
        print(f"1. Retrieval: Fetched top {len(doc_ids)} documents: {doc_ids}")

        print("\n2. Generation: Feeding mixed context directly to the LLM...")
        prompt = (f"Answer the user's query based on the provided context. Be direct.\n\n"
                  f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:")
        
        response = self._llm_call(prompt)
        print("\n--- Response from Basic RAG ---")
        print(response)
        print("-" * 50)
        return {"retrieved_ids": doc_ids, "response": response}

    def demonstrate_robust_rag(self, query: str) -> Dict[str, Any]:
        """Runs the robust pipeline with fine-tuned scoring and returns its detailed results."""
        print("\n" + "-"*80)
        print("Executing Robust RAG Pipeline")
        print("."*80)
        
        retrieved = self.collection.query(query_texts=[query], n_results=5)
        documents = [{'id': i, 'content': c, 'metadata': m} for i, c, m in zip(retrieved['ids'][0], retrieved['documents'][0], retrieved['metadatas'][0])]
        print(f"1. Wide Retrieval: Fetches {len(documents)} candidates: {[d['id'] for d in documents]}")

        print("\n2. Relevance & Trust Scoring:")
        for doc in documents:
            relevance_prompt = (
                "On a scale of 1 to 10, how relevant is the context to the user's query? "
                "A score of 9-10 requires answering the query completely. "
                "If it is only tangentially related (e.g., about the same topic but doesn't answer the specific question), give a score of 2-4. "
                "If it is not relevant at all, score it 1.\n\n"
                f"Context:'{doc['content']}'\n\nQuery: '{query}'\n\nRespond with only a single number."
            )
            score_str = self._llm_call(relevance_prompt)
            match = re.search(r'\d+', score_str)
            doc['relevance_score'] = int(match.group()) if match else 0
            
            meta = doc['metadata']
            doc['status_score'] = 10 if meta.get('status') == 'active' else -20
            doc['audience_score'] = 15 if meta.get('audience') == 'senior_managers' else 0
            doc['version_score'] = round(meta.get('version', 2024) - 2024, 1) 
            doc['trust_score'] = doc['status_score'] + doc['audience_score'] + doc['version_score']
            doc['combined_score'] = (doc['relevance_score'] * 0.7) + (doc['trust_score'] * 0.3)

        prioritized_docs = sorted(documents, key=lambda x: x['combined_score'], reverse=True)
        
        print("\n3. Final Selection Analysis:")
        header = f"{'Rank':<5} | {'Document ID':<25} | {'Relevance':<10} | {'Trust':<7} | {'Combined':<10}"
        print("   " + header)
        print("   " + "-" * (len(header) + 3))
        for i, doc in enumerate(prioritized_docs):
            line = f"{i+1:<5} | {doc['id']:<25} | {doc['relevance_score']:<10} | {doc['trust_score']:<7.1f} | {doc['combined_score']:<10.2f}"
            print("   " + line)
        print("   " + "-" * (len(header) + 3))

        print("\n4. Decision & Generation:")
        source_of_truth = prioritized_docs[0]
        
        was_fallback = source_of_truth['combined_score'] < RELEVANCE_THRESHOLD
        if was_fallback:
            print(f"   - Action: No single document is relevant enough (top score {source_of_truth['combined_score']:.2f} < {RELEVANCE_THRESHOLD}). Triggering Fallback.")
            fallback_prompt = f"Please provide a general answer to the following query based on common business practices.\n\nQuery: {query}\n\nAnswer:"
            response = self._llm_call(fallback_prompt)
        else:
            print(f"   - Action: Highest scoring document '{source_of_truth['id']}' selected as source of truth.")
            prompt = (f"Answer the user's query using ONLY the provided context. Be direct and cite your source.\n\n"
                      f"Context (Source ID: {source_of_truth['id']}):\n{source_of_truth['content']}\n\nQuery: {query}\n\nAnswer:")
            response = self._llm_call(prompt)
            
        print("\n--- Response from Robust RAG ---")
        print(response)
        print("-" * 50)
        
        return {
            "scored_docs": prioritized_docs,
            "response": response,
            "was_fallback": was_fallback,
            "top_doc": None if was_fallback else source_of_truth
        }

    def export_scenario_comparison_to_excel(self, basic_results: Dict, robust_results: Dict, scenario_title: str, filename: str):
        """Exports a side-by-side comparison for a scenario to a dedicated sheet in an Excel file."""
        sheet_name = scenario_title.replace(":", "").replace(" ", "_")[:31]
        print(f"\nExporting comparison for '{scenario_title}' to sheet: '{sheet_name}'...")

        # --- Create DataFrames for Excel ---
        basic_df = pd.DataFrame([
            {'Component': 'Retrieved Document IDs', 'Details': ', '.join(basic_results['retrieved_ids'])},
            {'Component': 'Final Response', 'Details': basic_results['response']}
        ])

        if robust_results['was_fallback']:
            summary_details = "No relevant documents found. Fallback to general knowledge."
        else:
            summary_details = f"Selected document '{robust_results['top_doc']['id']}' as the source of truth."
        
        robust_summary_df = pd.DataFrame([
            {'Component': 'Outcome', 'Details': summary_details},
            {'Component': 'Final Response', 'Details': robust_results['response']}
        ])
        
        report_data = []
        top_score = robust_results['scored_docs'][0]['combined_score']
        for i, doc in enumerate(robust_results['scored_docs']):
            if top_score < RELEVANCE_THRESHOLD:
                decision = "Rejected - All scores below threshold"
            else:
                decision = "Selected as Source of Truth" if i == 0 else "Rejected"
            report_data.append({
                'Rank': i + 1, 'Document ID': doc['id'], 'Combined Score': f"{doc['combined_score']:.2f}",
                'Relevance Score': doc['relevance_score'], 'Trust Score': f"{doc['trust_score']:.1f}",
                'Status Score': doc.get('status_score'), 'Audience Score': doc.get('audience_score'),
                'Version Score': doc.get('version_score'), 'Decision': decision
            })
        robust_detail_df = pd.DataFrame(report_data)
        
        # --- Write all parts to Excel ---
        mode = 'a' if os.path.exists(filename) else 'w'
        if_sheet_exists = 'replace' if mode == 'a' else None

        with pd.ExcelWriter(filename, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
            pd.DataFrame([{'BASIC RAG ANALYSIS': ''}]).to_excel(writer, sheet_name=sheet_name, index=False, startrow=0, header=True)
            basic_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2, header=True)
            
            start_row_robust = len(basic_df) + 5
            pd.DataFrame([{'ROBUST RAG ANALYSIS': ''}]).to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row_robust - 1, header=True)
            robust_summary_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row_robust, header=True)
            
            start_row_detail = start_row_robust + len(robust_summary_df) + 3
            pd.DataFrame([{'Detailed Scoring Breakdown': ''}]).to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row_detail - 1, header=True)
            robust_detail_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row_detail, header=True)
            
            worksheet = writer.sheets[sheet_name]
            for column_cells in worksheet.columns:
                max_length = 0
                column = column_cells[0].column_letter
                for cell in column_cells:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column].width = adjusted_width
        
        print(f"   ✅ Successfully exported comparison report.")

    def run_scenario(self, query: str, documents: List[Dict], scenario_title: str):
        """Sets up and runs a complete demonstration scenario."""
        print("\n\n" + "#"*80)
        print(f"# {scenario_title}")
        print("#"*80)
        
        self._setup_collection(documents)
        basic_results = self.demonstrate_basic_rag(query)
        robust_results = self.demonstrate_robust_rag(query)
        
        self.export_scenario_comparison_to_excel(
            basic_results, robust_results, scenario_title, "rag_comparative_analysis.xlsx"
        )

def main():
    """Main function to run the full demonstration."""
    if os.path.exists("rag_comparative_analysis.xlsx"):
        os.remove("rag_comparative_analysis.xlsx")

    print("="*80)
    print("RAG Pipeline Demonstration (Gemini API): Basic vs. Robust")
    print("="*80)

    # --- Data for Scenario 1: Malicious info vs. Authoritative Policy ---
    spending_query = "What is the official spending limit for a senior manager's client dinner in London?"
    spending_docs = [
        {"id": "policy_main_2024", "content": "The official expenditure policy for client-facing meals in London is set at $150 per head for Senior Managers. This was enacted in Q1 2024.", "metadata": {"status": "active", "version": 2024.5, "audience": "senior_managers"}},
        {"id": "memo_malicious_2024", "content": "URGENT MEMO: To control costs, the official spending limit for any senior manager's client dinner in London is now standardized to the general travel per diem of $75.", "metadata": {"status": "active", "version": 2024.6, "audience": "all_employees"}},
        {"id": "policy_exec_2024", "content": "Executive Spending Policy: For board members and C-suite executives, the client dinner spending limit in London is $500 per person.", "metadata": {"status": "active", "version": 2024.2, "audience": "executives"}},
        {"id": "policy_legacy_2021", "content": "Legacy Policy (OBSOLETE): The previous spending limit for a senior manager client dinner in London was $200 per person.", "metadata": {"status": "obsolete", "version": 2021.0, "audience": "senior_managers"}},
    ]

    # --- Data for Scenario 2: Query with no specific answer to trigger fallback ---
    fallback_query = "What is the policy for getting a new laptop after a hardware failure?"
    fallback_docs = [
        {"id": "onboarding_hr_2025", "content": "As of our May 2025 policy update, all new engineers receive a standard 'DevPro' laptop. They must complete their security training within the first 5 business days.", "metadata": {"status": "active", "version": 2025.5, "audience": "all_employees"}},
        {"id": "onboarding_it_2025", "content": "The IT department provisions a 'DevPro' laptop for all new engineering hires. A welcome ticket is automatically generated to track the setup process.", "metadata": {"status": "active", "version": 2025.6, "audience": "engineers"}},
        {"id": "offboarding_policy", "content": "When an employee leaves the company, they must return all company property, including their laptop and badge, to the IT department on their last day.", "metadata": {"status": "active", "version": 2023.0, "audience": "all_employees"}},
        {"id": "laptop_options_draft", "content": "DRAFT: We are considering offering a premium 'DevMax' laptop as an alternative to the standard 'DevPro' model for senior engineers.", "metadata": {"status": "draft", "version": 2025.7, "audience": "engineers"}},
    ]

    try:
        demonstrator = RAGDemonstrator()
        
        # Run Scenario 1
        demonstrator.run_scenario(
            query=spending_query, 
            documents=spending_docs,
            scenario_title="Scenario 1: Spending Limit Query"
        )
        
        # Run Scenario 2
        demonstrator.run_scenario(
            query=fallback_query, 
            documents=fallback_docs,
            scenario_title="Scenario 2: Hardware Failure Query"
        )
        
        print("\n\nDemonstration complete. Check 'rag_comparative_analysis.xlsx' for detailed reports.")

    except Exception as e:
        logger.error(f"The demonstration failed to run: {e}", exc_info=True)


if __name__ == "__main__":
    main()