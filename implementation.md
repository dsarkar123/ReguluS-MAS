# Implementation Guide: ReguluS-MAS

This document provides a step-by-step guide to set up and run the ReguluS-MAS project. This guide is based on the current implementation of the system.

## 1. Project Overview

ReguluS-MAS is a Retrieval-Augmented Generation (RAG) system designed to process and answer questions about regulatory circulars from the Monetary Authority of Singapore (MAS). It consists of two main pipelines:

*   **Ingestion Pipeline:** Parses a MAS notice from a PDF, enriches the content using Google's Gemini AI model, and stores it in a local ChromaDB vector database.
*   **Retrieval Pipeline:** Takes a user's question, searches the ChromaDB database for relevant text chunks, uses the Gemini model to re-rank them for relevance, and generates a final answer with citations.

## 2. Prerequisites

Before you begin, ensure you have the following:

*   **Python 3.8+**
*   **Google AI API Key:** The system uses Google's Generative AI for embedding, content generation, and re-ranking.
    1.  Obtain an API key from the [Google AI Studio](https://aistudio.google.com/app/apikey).
    2.  Set it as an environment variable in your terminal:
        ```bash
        export GOOGLE_API_KEY="your_api_key_here"
        ```

## 3. Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    The project's dependencies are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## 4. How to Run the System

The system is operated via the command line using `main.py`. It has two main commands: `ingest` and `query`.

### Ingestion Pipeline (`ingest`)

This process takes a PDF of a MAS notice, processes it, and adds it to the local vector database.

**Step 1: Place Your PDF**

Place the MAS notice PDF file you want to ingest into a directory. For example, create a `data/` directory and place it there. The parser expects a specific filename format to extract metadata: `MAS Notice <ID>_dated <Date>_effective <Date>.pdf`.

*Example Filename:* `MAS Notice 758_dated 18 Dec 2024_effective 26 Dec 2024.pdf`

**Step 2: Run the Ingestion Command**

Execute the following command in your terminal, replacing the path with the actual path to your PDF file:

```bash
python main.py ingest "data/MAS Notice 758_dated 18 Dec 2024_effective 26 Dec 2024.pdf"
```

**What Happens During Ingestion:**

1.  **Parsing:** The PDF is parsed into structured text chunks. A `_structured.json` file is created in the `data/` directory.
2.  **Enrichment:** Each text chunk is sent to the Gemini AI to generate a summary and a hypothetical question. Embeddings are created for the original text, summary, and question. An `_enriched.json` file is created in the `data/` directory.
3.  **Storage:** The enriched data and embeddings are stored in a local ChromaDB database located in a new `db/` directory at the root of the project.

You only need to run the ingestion process once for each new document.

### Retrieval Pipeline (`query`)

Once documents have been ingested, you can ask questions about them using the `query` command.

**Step 1: Run the Query Command**

Execute the following command in your terminal, replacing the text in quotes with your question:

```bash
python main.py query "What are the requirements for financial institutions regarding customer data?"
```

**What Happens During Retrieval:**

1.  **Search:** The system embeds your query and searches the ChromaDB database for the most relevant document chunks.
2.  **Re-rank:** The initial results are passed to the Gemini model, which scores them based on relevance to your specific query.
3.  **Synthesis:** The top-ranked, re-ranked documents are used as context for the Gemini model to generate a comprehensive final answer, complete with citations pointing to the source document.

The final answer will be printed directly to your console.
