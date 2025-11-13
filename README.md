Local RAG Pipeline for Daily arXiv LinkedIn Posts

This project is an end-to-end automated RAG (Retrieval-Augmented Generation) pipeline. Its primary purpose is to gain foundational knowledge in RAG systems by building a practical automation.

The system automatically fetches the latest AI and Data Science papers from arXiv, processes them into a local vector database, uses a local LLM to generate a focused, daily LinkedIn-ready post summarizing the day's key innovations, and finally translates this post into Turkish. The entire process runs 100% locally using Ollama, requiring no API keys.

🚀 Features

End-to-End Automation: A single script (main_linkedin_report.py) runs the entire workflow.

100% Local: Uses Ollama for both embeddings and LLM generation. No external API keys or costs.

Vector Storage: Employs LanceDB as a serverless, local-first vector store.

Advanced Map-Reduce RAG: Instead of a simple query, it uses a multi-step Map-Reduce strategy to synthesize insights from many papers into a single, cohesive post.

Metadata Filtering: The RAG query filters the vector store by publication date (metadata.published_date LIKE '...%') to generate true daily reports.

Idempotent: A local SQLite database (processed_papers.db) tracks processed papers to prevent re-indexing.

Local Translation: Includes a module to translate the final English post into Turkish using a local Helsinki-NLP model.

🛠️ Tech Stack

Core: Python 3.11+

LLM & Embeddings: LlamaIndex, Ollama (e.g., gemma3:4b, embeddinggemma:300m)

Vector Database: LanceDB

Translation: Hugging Face Transformers, NLTK

Data Fetching: arxiv

Utilities: tqdm (for progress bars)

🏗️ Architecture & Workflow

The pipeline is orchestrated by main_linkedin_report.py and follows a daily fetch, daily post strategy.

Fetch (Data Ingestor):

Queries the arXiv API for new papers from the last 1 day (DAYS_AGO = 1).

Deduplicate (DB Manager):

Checks each entry_id against the SQLite DB (processed_papers.db).

Already processed papers are skipped.

Index (Vector Indexer):

For new papers, the abstract is embedded using Ollama (embeddinggemma:300m).

The vector and its metadata (title, authors, published_date, etc.) are stored in LanceDB.

Processed paper IDs are saved to the SQLite DB.

Generate (RAG Reporter - Map-Reduce):

This is the core RAG step, executed once per day.

Retrieve: Fetches the Top-K (e.g., 100) most relevant papers from that day using the BASE_RAG_QUERY.

Map: The LLM summarizes these 100 papers in small, parallel batches (e.g., 5 batches of 20 papers) using the MAP_PROMPT_TEMPLATE.

Reduce: The LLM takes these 5 intermediate summaries and synthesizes them into a single, cohesive LinkedIn post, following the REDUCE_PROMPT_TEMPLATE.

Parse: The main script extracts the final post content by looking for <<<POST_START>>> and <<<POST_END>>> markers in the LLM's raw output.

Translate (Local Translator):

The single, generated English post is translated into Turkish using the Helsinki-NLP model, carefully preserving markdown formatting.

Save:

The final English (EN_post_...md) and Turkish (TR_post_...md) reports are saved to the daily_linkedin_post/ directory.

🔧 Setup & Installation

Clone the Repository:

git clone [https://github.com/ekocak02/rag-arxiv.git](https://github.com/ekocak02/rag-arxiv.git)
cd rag-arxiv


Install Ollama:

Download and install Ollama for your operating system.

Pull Ollama Models:

You must pull the embedding and LLM models specified in the scripts.

ollama pull embeddinggemma:300m
ollama pull gemma3:4b


(Note: If you use different models, update the EMBED_MODEL_NAME and LLM_MODEL_NAME constants in vector_indexer.py and rag_reporter.py.)

Create a Virtual Environment & Install Dependencies:

python3.11 -m venv .venv
source .venv/bin/activate  # (On Windows: .venv\Scripts\activate)
pip install -r requirements.txt


Download NLTK Data:

The local_translator.py script requires the punkt tokenizer. It will attempt to download it automatically on the first run.

Usage

To run the entire pipeline (fetch, index, report, and translate), simply execute the main script:

python main_linkedin_report.py


The script will log its progress through all 5 steps. Final posts will be saved in the daily_linkedin_post/ directory.

🗺️ Roadmap / Future Work

Docker Integration: Containerize the entire application with docker-compose.

Web Interface: Add a simple FastAPI backend and a Streamlit frontend to view the generated posts.

Full-Text Processing: Extend the data_ingestor to download and parse PDFs, allowing the RAG pipeline to index the full text of papers, not just the abstracts.