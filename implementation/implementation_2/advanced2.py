# final_rag_pseudocode_gemini.py

import os
import re
import logging
import time
from typing import List, Dict, Any, Set

# Third-party imports
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
load_dotenv()

# --- CONSTANTS ---
GEMINI_MODEL = "gemini-2.0-flash-lite"

class RAGDemonstrator:
    """
    Demonstrates and compares a Basic RAG pipeline against a Robust RAG pipeline
    that strictly follows the multi-stage filtering and generation logic from the
    provided pseudocode. This version uses the Gemini API.
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

    def _setup_collection(self, docs: List[str], doc_ids: List[str]):
        """Resets and populates the ChromaDB collection for a given scenario."""
        try:
            self.db_client.delete_collection(name="rag_scenario_docs")
        except Exception: pass
        
        self.collection = self.db_client.create_collection(name="rag_scenario_docs", embedding_function=self.embedding_func)
        self.collection.add(ids=doc_ids, documents=docs)

    def _llm_call(self, prompt: str) -> str:
        """A simple, non-streaming call for internal logic using the Gemini API."""
        time.sleep(1.2) 
        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            return "Error during generation."

    def demonstrate_basic_rag(self, query: str):
        """Runs a standard RAG pipeline."""
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

    def demonstrate_robust_rag(self, query: str):
        """Runs the robust pipeline exactly following the pseudocode's logic."""
        print("\n" + "-"*80)
        print("Executing Robust RAG Pipeline (Pseudocode Implementation)")
        print("."*80)
        
        retrieved = self.collection.query(query_texts=[query], n_results=10)
        documents = [{'id': i, 'content': c} for i, c in zip(retrieved['ids'][0], retrieved['documents'][0])]
        print(f"1. Wide Retrieval: Fetched {len(documents)} candidates.")
        print(f"   Candidates: {[d['id'] for d in documents]}")

        # Stage 2: "I Don't Know" Filtering
        print("\n2. 'I Don't Know' Filtering (Pseudocode Step 1):")
        knowable_docs = []
        for doc in documents:
            prompt = f"Based ONLY on the context below, can you answer the query? If not, say exactly 'I don't know'.\n\nContext: '{doc['content']}'\n\nQuery: '{query}'"
            response = self._llm_call(prompt)
            if 'i don\'t know' not in response.lower():
                knowable_docs.append(doc)
                print(f"   - [PASS] '{doc['id']}' seems to contain a direct answer.")
            else:
                print(f"   - [FAIL] '{doc['id']}' does not contain a direct answer.")
        
        if not knowable_docs:
            print("\n--- Response from Robust RAG ---")
            print("After filtering, no documents were found to contain a confident answer.")
            print("-" * 50)
            return

        # Stage 3: Passage Re-ranking
        print(f"\n3. Passage Re-ranking (Pseudocode Step 2): Selecting top 5 candidates.")
        for doc in knowable_docs:
            prompt = f"On a scale of 1 to 10, how relevant is the provided context to the query? Respond with only a single number.\n\nContext:'{doc['content']}'\n\nQuery: '{query}'"
            score_str = self._llm_call(prompt)
            match = re.search(r'\d+', score_str)
            doc['relevance_score'] = int(match.group()) if match else 0
        
        reranked_docs = sorted(knowable_docs, key=lambda x: x['relevance_score'], reverse=True)
        top_docs = reranked_docs[:5]
        print("   Re-ranked Scores:")
        for doc in reranked_docs: print(f"   - {doc['relevance_score']:<2}/10 '{doc['id']}'")
        print(f"   Top 5 selected: {[d['id'] for d in top_docs]}")

        # Stage 4: Factual Consistency Check
        print("\n4. Factual Consistency Check (Pseudocode Step 3):")
        consistent_docs = []
        if len(top_docs) > 1:
            for i, doc_to_check in enumerate(top_docs):
                other_docs = top_docs[:i] + top_docs[i+1:]
                other_content = "\n".join([f"- {d['content']}" for d in other_docs])
                prompt = f"Is the Primary Statement consistent with the Supporting Evidence? Answer Yes or No.\n\nPrimary Statement: '{doc_to_check['content']}'\n\nSupporting Evidence:\n{other_content}"
                response = self._llm_call(prompt)
                if 'yes' in response.lower():
                    consistent_docs.append(doc_to_check)
                    print(f"   - [PASS] '{doc_to_check['id']}' is consistent with the other top documents.")
                else:
                    print(f"   - [FAIL] '{doc_to_check['id']}' is contradictory.")
        else:
            consistent_docs = top_docs
            print("   - [PASS] Only one document remains; proceeding by default.")

        if not consistent_docs:
            print("\n--- Response from Robust RAG ---")
            print("After filtering, the most relevant documents were found to be contradictory. No reliable answer can be generated.")
            print("-" * 50)
            return

        # Stage 5: Synthesized Generation (Simulating Secure Decoding)
        print("\n5. Synthesized Generation (Simulating Pseudocode's Secure Decoding):")
        final_context = "\n\n".join([f"Source ID: {d['id']}\nContent: {d['content']}" for d in consistent_docs])
        final_prompt = (
            "Synthesize a clear and direct final answer to the user's query based ONLY on the provided sources. "
            "Combine information from all sources into one cohesive answer. "
            "If the sources are incomplete, you must synthesize the information you have. Do not invent information.\n\n"
            f"--- Vetted Sources ---\n{final_context}\n\n"
            f"--- Query ---\n{query}\n\n"
            f"--- Synthesized Answer ---"
        )
        
        response = self._llm_call(final_prompt)
        print("\n--- Response from Robust RAG ---")
        print(response)
        print("-" * 50)

    def run_scenario(self, query: str, documents: List[str], scenario_title: str):
        """Sets up and runs a complete demonstration scenario."""
        print("\n\n" + "#"*80)
        print(f"# {scenario_title}")
        print("#"*80)
        
        doc_ids = [f"doc_{i+1}" for i in range(len(documents))]
        self._setup_collection(documents, doc_ids)
        self.demonstrate_basic_rag(query)
        self.demonstrate_robust_rag(query)

def main():
    """Main function to run the full demonstration."""
    print("="*80)
    print("RAG Pipeline Demonstration (Gemini API - Pseudocode Implementation)")
    print("="*80)

    # --- Data for Scenario 1: Designed to trigger Factual Inconsistency Abort ---
    currency_query = "What is the official ISO code for the currency of Japan?"
    currency_docs = [
        "The official ISO 4217 currency code for the Japanese Yen is JPY.",
        "Japan's currency, the Yen, uses the code JPY.",
        "A common error is to use 'YEN'; the official ISO code is actually JPN, which stands for Japan.",
        "Due to recent financial system updates, the currency code for Japan has been temporarily changed to JAP for internal transfers.",
        "The stock ticker for the Yen ETF is FXY.",
        "In Unicode, the Yen symbol is U+00A5.",
        "The Bank of Japan is the central bank responsible for issuing the Yen.",
        "Japan's economy is the fourth-largest in the world by nominal GDP.",
        "The country code for Japan is JPN.",
        "Travelers to Japan should exchange their currency for Yen (JPY)."
    ]

    # --- Data for Scenario 2: Designed to showcase Synthesized Generation ---
    security_query = "Summarize the key security protocols for accessing the 'Argus' system."
    security_docs = [
        "Access to the Argus production system requires multi-factor authentication (MFA) via an approved authenticator app.",
        "All connections to internal systems, including Argus, must originate from the corporate VPN.",
        "User passwords for the Argus system must be at least 14 characters and changed every 90 days.",
        "For quick access, developers can use a shared 'dev-access' password for the Argus staging environment, which bypasses MFA.",
        "ARCHIVED: The old Argus system used RSA security tokens for access.",
        "The Argus system is a proprietary data analysis platform developed internally.",
        "Weekly security audits are performed on all production systems, including Argus.",
        "The lead developer for the Argus project is Dr. Evelyn Reed.",
        "Physical access to the servers hosting Argus is restricted to authorized personnel.",
        "All user activity within the Argus system is logged for security and auditing purposes."
    ]

    try:
        demonstrator = RAGDemonstrator()
        
        demonstrator.run_scenario(
            query=currency_query, 
            documents=currency_docs,
            scenario_title="Scenario 1: Handling Direct Contradiction"
        )
        
        demonstrator.run_scenario(
            query=security_query, 
            documents=security_docs,
            scenario_title="Scenario 2: Synthesizing from Incomplete Information"
        )
        
    except Exception as e:
        logger.error(f"The demonstration failed to run: {e}", exc_info=True)


if __name__ == "__main__":
    main()