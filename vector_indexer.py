import logging
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

# LlamaIndex core components
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings
)
# Ollama Embedding Model
from llama_index.embeddings.ollama import OllamaEmbedding
# LanceDB Vector Store
from llama_index.vector_stores.lancedb import LanceDBVectorStore

# Our other modules
import data_ingestor
import db_manager

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Model and Database Settings (Plan A2 - Module 3) ===

# Please enter the name of your embedding model running in Ollama
# (e.g., "all-minilm" or "nomic-embed-text")
EMBED_MODEL_NAME = "embeddinggemma:300m" 

# Directory to store the local vector database (Plan A2)
LANCEDB_DIR = "./lancedb_storage"
# Table name within LanceDB
LANCEDB_TABLE_NAME = "arxiv_papers"

# Project base directory
BASE_DIR = Path(__file__).resolve().parent
LANCEDB_PATH = BASE_DIR / LANCEDB_DIR

def get_vector_index() -> VectorStoreIndex:
    """
    Loads the existing LanceDB vector store or creates a new one.
    Configures the embedding model (Ollama) and storage (LanceDB).
    """
    logging.info(f"Preparing vector store. Path: {LANCEDB_PATH}")
    
    # 1. Configure the embedding model (Ollama)
    # We assign the model name globally via Settings.
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)
    
    # 2. Configure the vector store (LanceDB)
    vector_store = LanceDBVectorStore(
        uri=str(LANCEDB_PATH),
        table_name=LANCEDB_TABLE_NAME,
        mode="overwrite" # "a+" could also be used, but starting with "overwrite"
                         # ensures the most stable operation with LlamaIndex.
    )
    
    # 3. Create the StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 4. Load (or create) the index
    # We create a VectorStoreIndex object from the LanceDB store.
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    
    logging.info("✅ Vector store and index loaded/created successfully.")
    return index

def index_papers(papers_list: List[Dict], index: VectorStoreIndex):
    """
    Indexes a given list of paper dictionaries into the vector index.
    This is the core function for Plan A2 - Module 3.
    
    Args:
        papers_list (List[Dict]): The list of paper info dicts from data_ingestor.
        index (VectorStoreIndex): The index object to insert documents into.
    """
    logging.info(f"Indexing started: {len(papers_list)} new papers to process...")
    indexed_count = 0
    
    for paper in tqdm(papers_list, desc="Embedding and Indexing Papers"):
        try:
            # Per Plan A2 specification:
            # Main content (text) -> summary
            # Metadata -> All remaining information
            
            document_text = paper.get('summary', '')
            
            # Create the metadata dictionary
            metadata = {
                "arxiv_id": paper.get('arxiv_id'),
                "title": paper.get('title'),
                "authors": ", ".join(paper.get('authors', [])), # Convert list to string
                "published_date": paper.get('published_date'),
                "pdf_url": paper.get('pdf_url'),
                "categories": ", ".join(paper.get('categories', [])), # Convert list to string
            }
            
            # Create the LlamaIndex Document object
            doc = Document(text=document_text, metadata=metadata)
            
            # Add the document to the index (Embedding model runs here)
            index.insert(doc)
            
            # === On Success ===
            # If successfully added to the vector store, log it in SQLite
            db_manager.add_processed_paper(paper['arxiv_id'])
            
            indexed_count += 1
            
        except Exception as e:
            logging.error(f"❌ Error: {paper['arxiv_id']} could not be indexed. Error: {e}")
            # If an error occurs, it is NOT added to SQLite.
            # This allows it to be retried on the next run.

    logging.info(f"✅ Indexing complete. {indexed_count} / {len(papers_list)} papers processed successfully.")

# If this script is run directly, it performs a test indexing run.
if __name__ == "__main__":
    logging.info("Vector Indexer (vector_indexer.py) module testing...")
    
    # Placeholder for the user to fill in their model name
    PLACEHOLDER_NAME = "<EMBEDDING_MODEL_ADI>" 
    
    if EMBED_MODEL_NAME == PLACEHOLDER_NAME or EMBED_MODEL_NAME == "":
        logging.error(f"Please update the 'EMBED_MODEL_NAME' variable in '{__file__}' to a valid Ollama model.")
    else:
        # 1. Initialize/check databases
        db_manager.initialize_database()
        
        # 2. Fetch new papers
        logging.info("--- Stage 1: Fetching new papers ---")
        # BUG FIX: fetch_new_papers does not take 'max_results'.
        # We fetch all, then slice the list for testing.
        new_papers = data_ingestor.fetch_new_papers(days_ago=7, use_manual_pagination=False)
        
        if new_papers:
            # Limit to 50 papers for this test run
            test_papers = new_papers[:50]
            logging.info(f"Fetched {len(new_papers)} papers, limiting to {len(test_papers)} for the test run.")
            
            # 3. Load/create index
            logging.info("--- Stage 2: Preparing vector index ---")
            vector_index = get_vector_index()
            
            # 4. Index new papers (using the limited list)
            logging.info("--- Stage 3: Indexing new papers ---")
            index_papers(test_papers, vector_index)
            
            logging.info("\nTest complete. The 'lancedb_storage' directory should be created.")
            logging.info("If you run the test again, 'data_ingestor' should find 0 new papers (if no new papers were published).")
            
        else:
            logging.info("No new papers found to index. (System is up-to-date)")