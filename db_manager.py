import sqlite3
from pathlib import Path
import logging
from datetime import datetime

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine the database path based on the project's root directory
# This script creates the database file in the same directory.
BASE_DIR = Path(__file__).resolve().parent
DB_NAME = "processed_papers.db"
DB_PATH = BASE_DIR / DB_NAME

def initialize_database():
    """
    Initializes the SQLite database and creates the 'papers' table if it doesn't exist.
    Requirement for Plan A2 - Module 2.
    """
    conn = None  # Define conn outside try for the finally block
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create the 'papers' table
        # entry_id: The arXiv paper ID (PRIMARY KEY)
        # processed_at: Timestamp of processing (useful for tracking)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            entry_id TEXT PRIMARY KEY,
            processed_at TEXT NOT NULL
        )
        """)
        
        conn.commit()
        logging.info(f"✅ Database initialized/verified successfully: {DB_PATH}")
        
    except sqlite3.Error as e:
        logging.error(f"❌ Database error in initialize_database: {e}")
    finally:
        if conn:
            conn.close()

def is_paper_processed(entry_id: str) -> bool:
    """
    Checks if the given 'entry_id' already exists in the database.
    Requirement for Plan A2 - Module 2.
    
    Args:
        entry_id (str): The arXiv paper ID to check.
        
    Returns:
        bool: True if the paper has been processed, False otherwise.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM papers WHERE entry_id = ?", (entry_id,))
        exists = cursor.fetchone() is not None
        
    except sqlite3.Error as e:
        logging.error(f"❌ Database error in is_paper_processed: {e}")
        # In case of an error, assume it's not processed to allow a retry.
        return False
    finally:
        if conn:
            conn.close()
            
    return exists

def add_processed_paper(entry_id: str):
    """
    Saves a newly processed paper's ID to the database.
    Requirement for Plan A2 - Module 3.
    
    Args:
        entry_id (str): The arXiv paper ID to save.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        
        # Insert the ID and the processing timestamp
        cursor.execute("INSERT INTO papers (entry_id, processed_at) VALUES (?, ?)", 
                       (entry_id, current_time))
        conn.commit()
        logging.debug(f"Record added: {entry_id}")
        
    except sqlite3.IntegrityError:
        # This occurs if 'entry_id' already exists as a PRIMARY KEY.
        # This is an expected situation in a retry scenario, not a critical error.
        logging.warning(f"Notice: {entry_id} already exists in the database.")
    except sqlite3.Error as e:
        logging.error(f"❌ Database error in add_processed_paper: {e}")
    finally:
        if conn:
            conn.close()

# If this script is run directly, initialize the database.
if __name__ == "__main__":
    logging.info("Database Manager script run directly, initializing...")
    initialize_database()