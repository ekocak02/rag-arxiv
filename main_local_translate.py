import logging
import datetime
from pathlib import Path

# Import all modules from Plan A2
import db_manager
import data_ingestor
import vector_indexer
import rag_reporter
import local_translator

# Main logging configuration (All modules will inherit this)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)

# === MAIN SETTINGS ===

# The BASE RAG query to be used for daily reporting
# This query will be dynamically updated in the loop
BASE_RAG_QUERY = """
Based on the documents (research paper abstracts) provided for THIS SPECIFIC DAY, 
summarize the key trends and findings in AI and Data Science.

Identify 2-3 main themes for this day. For each theme:
1. Provide a detailed explanation of the theme and its significance.
2. Mention at least 1-2 relevant paper titles (or arXiv IDs) that support this theme.

Keep the summary detailed, professional, and insightful.
"""

# Directory to store the output reports
OUTPUT_DIR = "haftalik_raporlar" # (Directory name left as is)
BASE_DIR = Path(__file__).resolve().parent

def save_report(report_content: str, full_filename: str) -> Path:
    """
    Saves the given content to the reports directory with the specified filename.
    """
    try:
        output_path = BASE_DIR / OUTPUT_DIR
        # Create directory if it doesn't exist
        output_path.mkdir(exist_ok=True) 
        
        file_path = output_path / full_filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        logging.info(f"✅ Report saved successfully: {file_path}")
        return file_path
        
    except Exception as e:
        logging.error(f"❌ Failed to save report to file: {e}")
        return None

# === MAIN WORKFLOW (HYBRID STRATEGY) ===

def main_workflow():
    """
    Executes the full Plan A2 workflow (Modules 2-5) using a 
    hybrid DAILY LOOP strategy.
    """
    logging.info("="*50)
    logging.info("Plan A2 - Weekly RAG Reporting System Initialized")
    logging.info("STRATEGY: Hybrid (Fetch 7 Days of Data, Report Daily)")
    logging.info("="*50)

    # Use a single timestamp for all reports from this run
    report_run_date = datetime.date.today().isoformat()
    DAYS_AGO = 7 # Fetch the last 7 days

    # --- Step 1/6: Data Ingestion (Module 2) (FETCH 7 DAYS AT ONCE) ---
    logging.info("[1/6] Initializing database and fetching new papers (total 7 days)...")
    db_manager.initialize_database()
    
    new_papers = data_ingestor.fetch_new_papers(
        days_ago=DAYS_AGO, 
        use_manual_pagination=True # Recommended for stability
    )
    
    if not new_papers:
        logging.info("✅ System is up-to-date. No new papers found to index or report.")
        logging.info("Process complete.")
        return

    logging.info(f"Found {len(new_papers)} new papers to process.")

    # --- Step 2/6: Indexing (Module 3) (INDEX 7 DAYS AT ONCE) ---
    logging.info("[2/6] Loading vector index and indexing new papers...")
    try:
        vector_index = vector_indexer.get_vector_index()
        vector_indexer.index_papers(new_papers, vector_index)
        logging.info("Indexing complete.")
    except Exception as e:
        logging.error(f"❌ Critical error during vector indexing: {e}")
        return # Do not continue if indexing fails

    # --- Step 3/6: RAG Report Generation (Module 4) (DAILY LOOP) ---
    logging.info("[3/6] Running RAG query in daily loop (English Report)...")
    
    # 1. Dynamically calculate the 7 days to report on
    today = datetime.date.today()
    dates_to_report = []
    # (Excluding today, go back 7 days starting from yesterday)
    for i in range(1, DAYS_AGO + 1): 
        day = today - datetime.timedelta(days=i)
        dates_to_report.append(day.isoformat()) # "YYYY-MM-DD"
    
    # (Sorting from oldest to newest is more logical for the report)
    dates_to_report.reverse() 
    logging.info(f"Reporting on {len(dates_to_report)} days: {dates_to_report}")

    all_english_reports = []
    
    # 2. Loop through each day and generate a filtered report
    for report_day in dates_to_report:
        logging.info(f"\n--- Generating Daily Report: {report_day} ---")
        
        # Create a day-specific prompt
        day_specific_query = (
            f"## Daily Report: {report_day}\n\n"
            f"(Context for this section is limited to papers from: {report_day})\n"
            f"{BASE_RAG_QUERY}"
        )
        
        # Call the RAG engine with the DAILY FILTER
        english_report = rag_reporter.generate_english_report(
            day_specific_query,
            date_filter_str=report_day # "YYYY-MM-DD"
        )
        
        if not english_report or english_report.strip() == "":
            logging.warning(f"⚠️ {report_day} RAG engine failed to generate a report (or no data).")
            # Add a placeholder (to maintain format)
            all_english_reports.append(
                f"## Daily Report: {report_day}\n\n"
                f"*(No significant trends or data found for this date [{report_day}].) *"
            )
            continue
        
        all_english_reports.append(english_report)
        logging.info(f"✅ English draft report created for {report_day}.")

    if not all_english_reports:
        logging.error("❌ Failed to generate a report for any day.")
        return

    # 3. Combine all daily reports into a single weekly report
    final_english_report = "\n\n---\n\n".join(all_english_reports)
    logging.info("All daily English reports successfully combined.")
    logging.debug(f"Combined Report (First 100char): {final_english_report[:100]}...")

    # --- Step 4/6: Save English Report ---
    logging.info("[4/6] Saving combined English report (EN) to disk...")
    en_filename = f"EN_Weekly_Report_{report_run_date}.md"
    en_header = f"# Weekly AI/Data Science arXiv Report ({report_run_date})\n\n"
    en_header += "This report was generated by the local RAG system (Plan A2) using abstracts submitted to arXiv in the last 7 days.\n"
    en_header += f"Strategy: Daily Summary ({dates_to_report[0]} to {dates_to_report[-1]})\n\n---\n\n"
    save_report(en_header + final_english_report, en_filename)

    # --- Step 5/6: Translation (Module 5) ---
    logging.info("[5/6] Translating combined report to Turkish using local model...")
    turkish_report = local_translator.translate_to_turkish(final_english_report)
    
    # --- CRITICAL BUG FIX ---
    # Check for the English error string, not the old Turkish one.
    if "[TRANSLATION FAILED]" in turkish_report:
        logging.error("❌ An error occurred during local translation. Report not saved.")
        return
        
    logging.info("Translation complete.")

    # --- Step 6/6: Save Turkish Report ---
    logging.info("[6/6] Saving final Turkish report (TR) to Markdown (.md) file...")
    tr_filename = f"TR_Weekly_Report_{report_run_date}.md"
    
    # (NOTE: This header content remains in Turkish as it is the
    #  desired content for the final Turkish-language file.)
    tr_header = f"# Haftalık AI/Data Science arXiv Raporu ({report_run_date})\n\n"
    tr_header += "Bu rapor, son 7 günde arXiv'e eklenen makalelerin özetleri kullanılarak lokal RAG sistemi (Plan A2) ile oluşturulmuştur.\n"
    tr_header += f"Strateji: Günlük Özet ({dates_to_report[0]} - {dates_to_report[-1]})\n\n---\n\n"
    
    save_report(tr_header + turkish_report, tr_filename)
    
    logging.info("="*50)
    logging.info("✅ Plan A2 - Weekly Reporting (Hybrid Strategy) Completed Successfully.")
    logging.info(f"Output files saved to '{OUTPUT_DIR}' directory.")
    logging.info("="*50)

# === END OF WORKFLOW ===


if __name__ == "__main__":
    main_workflow()