import logging
from pathlib import Path
from datetime import datetime, timedelta
import re # Added for parsing citations
from typing import List, Tuple, Optional # Added for type hinting

# LlamaIndex core components
from llama_index.core import (
    VectorStoreIndex,
    Settings
)
# Added for Map-Reduce
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever

# Ollama LLM
from llama_index.llms.ollama import Ollama
# Ollama Embedding Model
from llama_index.embeddings.ollama import OllamaEmbedding
# LanceDB Vector Store
from llama_index.vector_stores.lancedb import LanceDBVectorStore

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Model and Database Settings ===

# (These must be the same as in 'vector_indexer.py')
EMBED_MODEL_NAME = "embeddinggemma:300m"
LLM_MODEL_NAME = "gemma3:4b"

LANCEDB_DIR = "./lancedb_storage"
LANCEDB_TABLE_NAME = "arxiv_papers"
BASE_DIR = Path(__file__).resolve().parent
LANCEDB_PATH = BASE_DIR / LANCEDB_DIR

# === RAG STRATEGY 1 (ORIGINAL - SIMPLE QUERY) ===

def generate_english_report(query: str, date_filter_str: str = None) -> str:
    """
    Queries the cumulative vector database (LanceDB) using a simple RAG pipeline.
    """
    
    logging.info(f"RAG reporter (Strategy 1) initialized...")
    logging.info(f"Using LLM: {LLM_MODEL_NAME}, Embed Model: {EMBED_MODEL_NAME}")
    
    try:
        # 1. Configure Settings
        Settings.llm = Ollama(model=LLM_MODEL_NAME, request_timeout=1000.0)
        Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)

        # 2. Load the vector store (LanceDB)
        vector_store = LanceDBVectorStore(
            uri=str(LANCEDB_PATH),
            table_name=LANCEDB_TABLE_NAME,
            mode="read_only"
        )

        # 3. Load the index
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        # 4. Build the Query Engine
        # Retrieve more documents if no date filter is applied
        top_k = 20 if date_filter_str else 50
        query_engine_kwargs = {"similarity_top_k": top_k}
        
        # Apply metadata filtering if a date string is provided
        if date_filter_str:
            where_clause = f"metadata.published_date LIKE '{date_filter_str}%'"
            query_engine_kwargs["vector_store_kwargs"] = {"where": where_clause}
            logging.info(f"Applying metadata filter: {where_clause}")

        query_engine = index.as_query_engine(**query_engine_kwargs)
        
        logging.info(f"Query engine ready. Sending query: '{query[:50]}...'")
        
        # 5. Run the query
        response = query_engine.query(query)
        
        logging.info("✅ English report generated successfully.")
        return str(response)
        
    except Exception as e:
        logging.error(f"❌ Error during RAG report generation: {e}")
        return ""


# === RAG STRATEGY 2 (NEW - MAP-REDUCE FOR LINKEDIN) ===

# --- MAP-REDUCE PROMPTS ---

# This prompt is used in the "Map" phase (called 5 times in this setup)
MAP_PROMPT_TEMPLATE = """
You are a senior AI researcher. Analyze the following batch of research paper 
abstracts (and their arXiv IDs). Your goal is to identify the 2-3 most 
significant innovations or novel techniques discussed.

For each innovation you find, briefly explain it and *you must* cite the 
supporting paper(s) using the format [arXiv:ID].

--- CONTEXT (Abstracts Batch) ---
{context_str}
---

Your Summary (Innovations and [arXiv:ID] citations):
"""

# This prompt is used in the "Reduce" phase (called 1 time)
REDUCE_PROMPT_TEMPLATE = """
You are an expert AI/Tech content creator for LinkedIn. Your task is to
synthesize the 5 analysis reports below (separated by '===') into a single,
engaging, and insightful LinkedIn post.

The post should highlight today's most exciting innovations from arXiv.
- Start with a strong hook.
- Clearly explain the key trends or breakthroughs.
- *Synthesize insights from at least 3 different reports* to show diversity.
- Maintain a professional and optimistic tone.
- *Crucially*, preserve the [arXiv:ID] citations from the reports 
  (e.g., "[arXiv:2401.1234]") when you mention a specific finding.

**CRITICAL RULES:**
1. The output MUST be *only* the LinkedIn post text itself.
2. You MUST wrap your entire response between two unique markers:
   `<<<POST_START>>>` at the beginning.
   `<<<POST_END>>>` at the end.
3. Do NOT add a "Sources" section (this will be added later).
4. **If you see the *exact same finding* (e.g., the same paper) mentioned in multiple reports, *do NOT repeat it*. Summarize only unique insights.**

--- CONTEXT (5 Analysis Reports) ---
{context_str}
---

Your Final LinkedIn Post (wrapped in markers):
"""


def _get_retriever(
    index: VectorStoreIndex,
    top_k: int,
    date_filter_str: str
) -> BaseRetriever:
    """Helper function to build the retriever with a date filter."""
    where_clause = f"metadata.published_date LIKE '{date_filter_str}%'"
    logging.info(f"Applying metadata filter: {where_clause}")
    
    return index.as_retriever(
        similarity_top_k=top_k,
        vector_store_kwargs={"where": where_clause}
    )

def _format_nodes_for_map(nodes: List[NodeWithScore]) -> str:
    """Helper function to format a chunk of nodes for the MAP prompt context."""
    node_texts = []
    for node in nodes:
        # Include metadata (arxiv_id) in the context for the LLM
        paper_id = node.metadata.get('arxiv_id', 'UNKNOWN_ID')
        text = node.get_content()
        node_texts.append(
            f"Paper [arXiv:{paper_id}]:\n{text}"
        )
    return "\n\n---\n\n".join(node_texts)


def generate_linkedin_post_map_reduce(
    base_query: str,
    date_filter_str: str
) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Generates a LinkedIn post using a "Map-Reduce" RAG strategy.
    
    Returns:
        Tuple[Optional[str], Optional[List[str]]]:
        (Generated Post Text (RAW, with markers), List of Cited arXiv IDs)
    """
    logging.info(f"RAG reporter (Strategy 2: Map-Reduce) initialized...")
    
    # --- Phase 0: Setup ---
    try:
        # 1. Configure Settings
        Settings.llm = Ollama(model=LLM_MODEL_NAME, request_timeout=1000.0)
        Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)

        # 2. Load the vector store (LanceDB)
        vector_store = LanceDBVectorStore(
            uri=str(LANCEDB_PATH),
            table_name=LANCEDB_TABLE_NAME,
            mode="read_only"
        )
        # 3. Load the index
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
    except Exception as e:
        logging.error(f"❌ Error during RAG setup: {e}")
        return None, None

    # --- Phase 1: Retrieve (Fetch Top 100) ---
    logging.info("Map-Reduce (1/4): Retrieving top 100 nodes...")
    top_k = 100
    chunk_size = 20 # (Process 100 nodes in 5 chunks of 20)
    
    try:
        retriever = _get_retriever(index, top_k, date_filter_str)
        nodes = retriever.retrieve(base_query)
        
        if not nodes:
            logging.warning("⚠️ No relevant nodes found for today's filter.")
            return None, None
            
        logging.info(f"Retrieved {len(nodes)} nodes.")
        
    except Exception as e:
        logging.error(f"❌ Error during RAG Retrieve phase: {e}")
        return None, None

    # --- Phase 2: Map (Summarize 5x 20 Chunks) ---
    logging.info(f"Map-Reduce (2/4): Mapping {len(nodes)} nodes in {chunk_size}-item chunks...")
    intermediate_summaries = []
    
    # Calculate number of chunks needed (e.g., 100 nodes / 20 chunk_size = 5 chunks)
    num_chunks = (len(nodes) + chunk_size - 1) // chunk_size
    
    try:
        for i in range(num_chunks):
            logging.info(f"  - Mapping chunk {i+1}/{num_chunks}...")
            chunk_nodes = nodes[i*chunk_size : (i+1)*chunk_size]
            
            # Format context (including [arXiv:ID]s) for the LLM
            map_context_str = _format_nodes_for_map(chunk_nodes)
            
            # Call LLM (Map Phase)
            prompt = MAP_PROMPT_TEMPLATE.format(context_str=map_context_str)
            response = Settings.llm.complete(prompt)
            
            intermediate_summaries.append(str(response))
            
    except Exception as e:
        logging.error(f"❌ Error during RAG Map phase: {e}")
        return None, None

    # --- Phase 3: Reduce (Combine 5 Summaries) ---
    logging.info("Map-Reduce (3/4): Reducing {len(intermediate_summaries)} intermediate summaries...")
    try:
        reduce_context_str = "\n\n========\n\n".join(intermediate_summaries)
        
        # Call LLM (Reduce Phase)
        prompt = REDUCE_PROMPT_TEMPLATE.format(context_str=reduce_context_str)
        final_response = Settings.llm.complete(prompt)
        
        # We now return the RAW text, which includes the markers
        final_post_text = str(final_response)
        
        logging.info("✅ Reduce phase complete. Final post generated (with markers).")
        
    except Exception as e:
        logging.error(f"❌ Error during RAG Reduce phase: {e}")
        return None, None

    # --- Phase 4: Parse Citations ---
    logging.info("Map-Reduce (4/4): Parsing citations from final post...")
    try:
        # Find all unique arXiv IDs (e.g., "2401.1234" or "2401.1234v1")
        # We parse from the raw text, just in case citations are outside markers
        citations = re.findall(r'\[arXiv:(\d{4}\.\d{4,}[\w\-]*)\]', final_post_text)
        unique_citations = sorted(list(set(citations)))
        
        logging.info(f"Found {len(unique_citations)} unique citations.")
        
        # Return the RAW post text AND the list of IDs
        return final_post_text, unique_citations
        
    except Exception as e:
        logging.error(f"❌ Error during citation parsing: {e}")
        return None, None


# If this script is run directly, it performs test runs.
if __name__ == "__main__":
    logging.info("RAG Reporter Module (rag_reporter.py) testing...")
    
    # Check if placeholder model names are still present
    if (LLM_MODEL_NAME == "<YOUR_LLM_MODEL_NAME>" or
        EMBED_MODEL_NAME == "<YOUR_EMBEDDING_MODEL_NAME>"):
        logging.error(f"Please update 'LLM_MODEL_NAME' and 'EMBED_MODEL_NAME' in '{__file__}'.")
    else:
        
        # --- TEST 1: New Map-Reduce Strategy ---
        logging.info("\n--- TEST 1: MAP-REDUCE STRATEGY (Strategy 2) ---")
        
        # (This test requires data from that day to exist in 'lancedb_storage')
        test_day = (datetime.now().date() - timedelta(days=1)).isoformat()
        # Base query for the retriever, before Map-Reduce
        test_query = "Today's innovations in AI" 
        
        post, citations = generate_linkedin_post_map_reduce(
            base_query=test_query,
            date_filter_str=test_day
        )
        
        if post:
            print("\n--- GENERATED POST (TEST) - (RAW WITH MARKERS) ---")
            print(post)
            print("\n--- CITED IDS (TEST) ---")
            print(citations)
        else:
            print("Map-Reduce post could not be generated (or no data found).")

        # --- TEST 2: Original 'generate_english_report' ---
        logging.info("\n--- TEST 2: ORIGINAL STRATEGY (Strategy 1) ---")
        test_day_orig = (datetime.now().date() - timedelta(days=3)).isoformat()
        TEST_QUERY_ORIG = "Summarize key trends in AI"
        
        daily_report = generate_english_report(TEST_QUERY_ORIG, date_filter_str=test_day_orig)
        
        if daily_report:
            print(daily_report)
        else:
            print("Original report could not be generated.")