# Local RAG Pipeline for Daily arXiv Summaries

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is an end-to-end automated RAG (Retrieval-Augmented Generation) pipeline. Its primary purpose is to gain foundational knowledge in RAG systems by building a practical automation.

The system automatically fetches the latest AI and Data Science papers from arXiv, processes them into a local vector database, uses a local LLM to generate daily trend-analysis reports, and finally translates these reports into Turkish. The entire process runs 100% locally using Ollama, requiring no API keys.

## 🚀 Features

* **End-to-End Automation:** A single script (`main_local_translate.py`) runs the entire workflow.
* **100% Local:** Uses [Ollama](https://ollama.com/) for both embeddings and LLM generation. No external API keys or costs.
* **Vector Storage:** Employs [LanceDB](https://lancedb.github.io/lancedb/) as a serverless, local-first vector store.
* **Daily Trend Analysis:** Instead of just listing papers, it uses a RAG prompt to identify and summarize key *themes* for each day.
* **Metadata Filtering:** The RAG query filters the vector store by publication date (`metadata.published_date LIKE '...%'`) to generate true daily reports.
* **Idempotent:** A local SQLite database (`processed_papers.db`) tracks processed papers to prevent re-indexing.
* **Local Translation:** Includes a module to translate the final English report into Turkish using a local Helsinki-NLP model.

## 🛠️ Tech Stack

* **Core:** Python 3.10+
* **LLM & Embeddings:** [LlamaIndex](https://www.llamaindex.ai/), [Ollama](https://ollama.com/) (e.g., `gemma3:4b`, `embeddinggemma:300m`)
* **Vector Database:** [LanceDB](https://lancedb.github.io/lancedb/)
* **Translation:** [Hugging Face Transformers](https://huggingface.co/transformers), [NLTK](https://www.nltk.org/)
* **Data Fetching:** [arxiv](https://pypi.org/project/arxiv/)
* **Utilities:** `tqdm` (for progress bars)

## 🏗️ Architecture & Workflow

The pipeline is orchestrated by `main_local_translate.py` and follows a hybrid strategy: it fetches 7 days of data at once but generates reports in a daily loop.

1.  **Fetch (Data Ingestor):**
    * Queries the arXiv API for new papers from the last 7 days (`data_ingestor.py`).

2.  **Deduplicate (DB Manager):**
    * Checks each `entry_id` against the SQLite DB (`processed_papers.db`).
    * Already processed papers are skipped (`db_manager.py`).

3.  **Index (Vector Indexer):**
    * For new papers, the abstract is embedded using Ollama (`embeddinggemma:300m`).
    * The vector and its metadata (title, authors, `published_date`, etc.) are stored in LanceDB (`vector_indexer.py`).
    * Processed paper IDs are saved to the SQLite DB.

4.  **Generate (RAG Reporter):**
    * The main script loops through each of the past 7 days.
    * For each day, it runs a RAG query against the LLM (`gemma3:4b`).
    * The query is **filtered** using the `date_filter_str` parameter, forcing the RAG engine to only use documents from that specific day (`rag_reporter.py`).
    * The LLM generates a thematic summary (e.g., "Theme 1: ...", "Theme 2: ...") for that day.

5.  **Translate (Local Translator):**
    * All daily English reports are combined into a single large markdown file.
    * This final report is translated into Turkish using the `Helsinki-NLP` model, carefully preserving markdown formatting (`local_translator.py`).

6.  **Save:**
    * The final English (`EN_Weekly_Report_...md`) and Turkish (`TR_Haftalik_Rapor_...md`) reports are saved to the `haftalik_raporlar/` directory.

## 🔧 Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-project-name.git](https://github.com/your-username/your-project-name.git)
    cd your-project-name
    ```

2.  **Install Ollama:**
    * Download and install [Ollama](https://ollama.com/) for your operating system.

3.  **Pull Ollama Models:**
    * You must pull the embedding and LLM models specified in the scripts.
    ```bash
    ollama pull embeddinggemma:300m
    ollama pull gemma3:4b
    ```
    *(Note: If you use different models, update the `EMBED_MODEL_NAME` and `LLM_MODEL_NAME` constants in `vector_indexer.py` and `rag_reporter.py`.)*

4.  **Create a Virtual Environment & Install Dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # (On Windows: .venv\Scripts\activate)
    pip install -r requirements.txt
    ```

5.  **Download NLTK Data:**
    * The `local_translator.py` script requires the `punkt` tokenizer. It will attempt to download it automatically on the first run.

## Usage

To run the entire pipeline (fetch, index, report, and translate), simply execute the main script:

```bash
python main_local_translate.py
```

The script will log its progress through all 6 steps. Final reports will be saved in the `haftalik_raporlar/` directory.

## 🗺️ Roadmap / Future Work

* **Docker Integration:** Containerize the entire application with `docker-compose`. This will package the Python environment and the Ollama service into a single, reproducible unit.
* **Web Interface:** Add a simple FastAPI backend and a Streamlit frontend to view the generated reports from a browser.
* **Full-Text Processing:** Extend the `data_ingestor` to download and parse PDFs, allowing the RAG pipeline to index the full text of papers, not just the abstracts.