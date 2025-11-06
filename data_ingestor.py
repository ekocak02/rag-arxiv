import arxiv
from arxiv import Client
import logging
import db_manager
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Plan A2 - Relevant Categories
AI_CATEGORIES = [
    'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'stat.ML', 'cs.IR', 'cs.DB'
]

# --- Helper Function (Refactored for DRY) ---

def _build_date_filtered_query(days_ago: int) -> str:
    """
    Creates an enhanced arXiv API query string with a date filter.
    
    Args:
        days_ago (int): How many days back to search.
        
    Returns:
        str: The formatted API query string.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_ago)

    # Format: YYYYMMDDHHMMSS
    start_date_str = start_date.strftime("%Y%m%d%H%M%S")
    end_date_str = end_date.strftime("%Y%m%d%H%M%S")
    
    logging.info(f"📅 API Date Range (UTC): {start_date_str} to {end_date_str}")
    
    categories_query = " OR ".join([f"cat:{cat}" for cat in AI_CATEGORIES])
    date_query = f"submittedDate:[{start_date_str} TO {end_date_str}]"
    
    # Final query: Combine categories and date with 'AND'
    # (Parentheses prioritize the category query)
    query = f"({categories_query}) AND {date_query}"
    
    logging.info(f"API Query (first 100 chars): {query[:100]}...")
    return query

# --- Helper Function (Refactored for DRY) ---

def _process_paper_result(result: arxiv.Result) -> Optional[Dict]:
    """
    Processes a single arxiv.Result object into a dictionary.
    Returns None if the paper is already processed or fails parsing.
    
    Args:
        result (arxiv.Result): An individual paper result from the API.
        
    Returns:
        Optional[Dict]: A dictionary with paper info, or None.
    """
    try:
        entry_id = result.get_short_id()
        
        # Check if this paper was already processed
        if db_manager.is_paper_processed(entry_id):
            return None
        
        # Make the published date 'naive' for DB storage
        pub_date = result.published.replace(tzinfo=None)

        paper_info = {
            'arxiv_id': entry_id,
            'entry_id_full': result.entry_id,
            'title': result.title,
            'authors': [author.name for author in result.authors],
            # Clean up summaries by removing newlines
            'summary': result.summary.replace('\n', ' '),
            'published_date': pub_date.strftime("%Y-%m-%d %H:M:%S"),
            'pdf_url': result.pdf_url,
            'categories': result.categories,
        }
        return paper_info
        
    except Exception as e:
        logging.warning(f"⚠️  Inner error while processing paper ({result.entry_id}): {e}")
        return None

# --- Main Functions ---

def fetch_new_papers(days_ago: int = 7, use_manual_pagination: bool = False) -> List[Dict]:
    """
    Fetches AI/Data Science papers published in the last N days that are not in the DB.
    
    TWO METHODS:
    1. Auto (default): Uses the library's built-in automatic pagination.
    2. Manual: Uses our own pagination control (more stable during arXiv API issues).
    
    Args:
        days_ago (int): How many days back to search (default: 7).
        use_manual_pagination (bool): Use manual pagination if True.
    
    Returns:
        List[Dict]: A list of new paper information dictionaries.
    """
    
    if use_manual_pagination:
        return fetch_new_papers_manual(days_ago)
    else:
        return fetch_new_papers_auto(days_ago)


def fetch_new_papers_auto(days_ago: int = 7) -> List[Dict]:
    """
    Fetches papers using automatic pagination (ENHANCED QUERY).
    Reduces API load by adding a 'submittedDate' filter.
    """
    logging.info(f"🔍 Querying arXiv API for new papers from the last {days_ago} days...")
    logging.info("📌 Mode: Automatic Pagination (enhanced with 'submittedDate' Filter)")
    
    query = _build_date_filtered_query(days_ago)

    # Create the 'Search' object with the ENHANCED QUERY
    search = arxiv.Search(
        query=query,
        max_results=float('inf'), # Get all results (already filtered by date)
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    new_papers_list = []
    processed_count = 0
    total_inspected = 0
    
    logging.info("⬇️  Fetching papers from arXiv API (auto mode)...\n")

    try:
        # The generator handles pagination implicitly using the default Client.
        for result in search.results():
            total_inspected += 1
            
            paper_info = _process_paper_result(result)
            
            if paper_info:
                new_papers_list.append(paper_info)
            else:
                # Paper was either in DB or had an error
                if result: # Check if result is not None
                   if db_manager.is_paper_processed(result.get_short_id()):
                       processed_count += 1
                
            if (total_inspected % 50 == 0) and (total_inspected > 0):
                logging.info(
                    f"📊 Inspected: {total_inspected} | "
                    f"Skipped (in DB): {processed_count} | "
                    f"Found (New): {len(new_papers_list)}"
                )
                
    except arxiv.UnexpectedEmptyPageError:
        logging.warning("⚠️  arXiv API returned an unexpected empty page (normal behavior).")
        logging.info(f"✅ Collected {len(new_papers_list)} papers so far, finishing...")
    
    except Exception as e:
        logging.error(f"❌ Critical error during arXiv fetch (auto): {e}")

    print_summary(total_inspected, processed_count, len(new_papers_list))
    return new_papers_list


def fetch_new_papers_manual(days_ago: int = 7) -> List[Dict]:
    """
    Fetches papers using manual pagination (ENHANCED QUERY).
    More stable for large result sets or API instability.
    """
    logging.info(f"🔍 Querying arXiv API for new papers from the last {days_ago} days...")
    logging.info("📌 Mode: Manual Pagination (enhanced with 'submittedDate' Filter)")
    
    query = _build_date_filtered_query(days_ago)

    new_papers_list = []
    processed_count = 0
    total_inspected = 0
    
    # 1. 'Search' object
    search = arxiv.Search(
        query=query,
        max_results=float('inf'), # Get all results (already filtered by date)
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    # 2. 'Client' object
    client = Client(
        page_size = 100,      
        delay_seconds = 31,    # 5-second delay is polite to the API and sufficient.
        num_retries = 5       # 5 retries is sufficient
    )
    
    logging.info(f"⬇️  Fetching papers with manual pagination (Client(page_size={client.page_size}))...")

    try:
        for result in client.results(search):
            total_inspected += 1
            
            paper_info = _process_paper_result(result)
            
            if paper_info:
                new_papers_list.append(paper_info)
            else:
                # Paper was either in DB or had an error
                if result:
                    if db_manager.is_paper_processed(result.get_short_id()):
                        processed_count += 1
                
            if (total_inspected % 50 == 0) and (total_inspected > 0):
                logging.info(
                    f"📊 Inspected: {total_inspected} | "
                    f"Skipped (in DB): {processed_count} | "
                    f"Found (New): {len(new_papers_list)}"
                )
                
    except arxiv.UnexpectedEmptyPageError:
        logging.warning("⚠️  arXiv API returned an unexpected empty page (normal behavior).")
        logging.info(f"✅ Collected {len(new_papers_list)} papers so far, finishing...")
    
    except Exception as e:
        logging.error(f"❌ Critical error during arXiv fetch (manual): {e}")
    
    print_summary(total_inspected, processed_count, len(new_papers_list))
    return new_papers_list


def print_summary(total_inspected, processed_count, new_paper_count):
    """Prints summary statistics"""
    logging.info("\n" + "="*70)
    logging.info("DATA INGESTION SUMMARY")
    logging.info("="*70)
    logging.info(f"Total paper metadata inspected from API: {total_inspected}")
    logging.info(f"├─ Already in DB: {processed_count} ")
    logging.info(f"└─ NEW PAPERS Found: {new_paper_count}")
    logging.info("="*70 + "\n")


if __name__ == "__main__":
    """
    Test both methods for comparison.
    """
    logging.info("Data Ingestor (data_ingestor.py) module testing...\n")
    
    db_manager.initialize_database()
    
    # METHOD 1: Automatic (default)
    print("\n" + "🔹"*35)
    print("METHOD 1: Automatic Pagination")
    print("🔹"*35 + "\n")
    papers_auto = fetch_new_papers(days_ago=7, use_manual_pagination=False)
    
    # METHOD 2: Manual (more stable)
    print("\n" + "🔹"*35)
    print("METHOD 2: Manual Pagination")
    print("🔹"*35 + "\n")
    papers_manual = fetch_new_papers(days_ago=7, use_manual_pagination=True)
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Automatic Method Found: {len(papers_auto)} papers")
    print(f"Manual Method Found: {len(papers_manual)} papers")
    print("="*70)