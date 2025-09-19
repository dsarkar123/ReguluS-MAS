# ReguluS-MAS
Regulatory Intelligence System for MAS
### **Technical Specification: Generalized RAG System for MAS Regulatory Circulars**

**Project:** ReguluS-MAS (Regulatory Intelligence System for MAS)
**Version:** 2.0
**Date:** September 19, 2025

#### **1. Project Overview & Goal**

The goal of this project is to build a scalable and flexible Retrieval-Augmented Generation (RAG) system. This system will ingest, process, and accurately answer questions on any regulatory circular or notice issued by the Monetary Authority of Singapore (MAS). The system must handle multiple documents, distinguish between them, and provide precise, citable answers that reference the specific source document.

-----

#### **2. System Architecture**

The architecture remains a two-pipeline system: a generic **Ingestion Pipeline** that can be run for any new document, and a multi-document aware **Retrieval Pipeline**.

-----

#### **3. Ingestion Pipeline Specification (Generic)**

This pipeline is designed as a reusable workflow that can process any MAS circular provided as a PDF.

##### **Component 3.1: Document Deconstruction (Generalized Parser)**

  * **Objective:** To parse any given MAS circular PDF into a hierarchical JSON structure, dynamically identifying its logical layout.
  * **Input:** A PDF file (e.g., `MAS Notice 758_dated 18 Dec 2024_effective 26 Dec 2024.pdf`).
  * **Output:** A structured JSON file (e.g., `MAS_758_structured.json`).
  * **Key Logic:**
    1.  **Metadata Extraction:** Before parsing, extract key metadata from the filename or the document's header/footer using regex.
          * `notice_id`: "MAS Notice 758"
          * `publication_date`: "18 Dec 2024"
          * `effective_date`: "26 Dec 2024"
    2.  **Adaptive Parsing:** Use the **`PyMuPDF`** library. Instead of hardcoding section names like "Article", the parser must be heuristic-based. It should identify structural elements by looking for common MAS patterns:
          * Headings (e.g., "PART I", "ANNEX A").
          * Numbered paragraphs (e.g., "1.", "2.", "3.1", "3.2").
          * Sub-paragraphs (e.g., "(a)", "(b)", "(i)").
          * Definition sections (often at the beginning).
    3.  **JSON Structure:** Create a nested JSON. Each node must contain:
          * `node_id`: A unique identifier (e.g., `MAS_758_Part_I_Para_3_a`).
          * `node_type`: The identified type (e.g., `Part`, `Paragraph`, `Sub-paragraph`, `Definition`).
          * `text`: The verbatim text content.
          * `parent_id`: The `node_id` of the parent element.
  * **Libraries:** `PyMuPDF`, `re` (for regex).

##### **Component 3.2: Enrichment & Embedding**

  * **Objective:** To enrich each node with summaries and questions, then generate multiple embeddings. This component remains largely the same but operates on the generic parser's output.
  * **Input:** The structured JSON from the parser (e.g., `MAS_758_structured.json`).
  * **Output:** A list of data objects for the vector database.
  * **Key Logic:**
    1.  Traverse the input JSON tree.
    2.  For each node, generate summaries and hypothetical questions using an LLM (e.g., `claude-3-sonnet-20240229` for a balance of cost and performance) via **`langchain`**. The prompts remain the same.
    3.  Use a sentence-transformer model (e.g., `all-mpnet-base-v2`) to create three vector embeddings: `vector_content`, `vector_summary`, and `vector_question`.
  * **Libraries:** `langchain`, `sentence-transformers`.

##### **Component 3.3: Vector Storage (Multi-Document Schema)**

  * **Objective:** To store the data in a vector database, with a schema that supports multiple documents and filtering.
  * **Input:** The list of enriched data objects.
  * **Technology:** **Pinecone** (or similar vector DB).
  * **Schema / Data Structure:**
      * Use a **single Pinecone index** for all MAS documents.
      * For each processed node, upsert a record with:
          * **ID:** The unique `node_id`.
          * **Vectors:** The three generated embeddings.
          * **Metadata:** This is the most critical change. The metadata must be robust for filtering.
            ```json
            {
              "notice_id": "MAS Notice 758",
              "publication_date": "2024-12-18",
              "effective_date": "2024-12-26",
              "original_text": "...",
              "summary_text": "...",
              "question_text": "...",
              "hierarchy": { "part": "I", "paragraph": "3", "sub_paragraph": "a" },
              "parent_id": "MAS_758_Part_I_Para_3"
            }
            ```
  * **Libraries:** `pinecone-client`.

-----

#### **4. Retrieval Pipeline Specification (Multi-Document Aware)**

This pipeline is enhanced to handle queries that may or may not specify a particular document.

##### **Component 4.1: Query Handling & Filtered Search**

  * **Objective:** To perform a multi-vector search that can be filtered by a specific document if required.
  * **Input:**
      * `user_query` (string)
      * `document_filter` (optional string, e.g., "MAS Notice 758")
  * **Output:** A ranked list of candidate `node_id`s.
  * **Key Logic:**
    1.  Generate an embedding for the `user_query`.
    2.  Construct a query for Pinecone.
    3.  **Apply Filter:** If a `document_filter` is provided, add a metadata filter to the Pinecone query.
        ```python
        # Example Pinecone query logic
        query_filter = {}
        if document_filter:
            query_filter = {"notice_id": {"$eq": document_filter}}

        results = index.query(
            vector=query_embedding,
            top_k=10,
            filter=query_filter,
            include_metadata=True
        )
        ```
    4.  Perform the multi-vector search as specified in v1.0, combining and scoring the results.

##### **Component 4.2: Contextual Expansion & Re-Ranking**

  * **Objective:** To enrich results with context and re-rank for relevance. This component's logic is unchanged but benefits greatly from the accurate, filtered results of the previous step.
  * **Input:** Ranked list of candidate `node_id`s.
  * **Output:** Final, re-ranked list of text chunks.
  * **Key Logic:**
    1.  For the top `N` candidates, fetch their full data (including all metadata).
    2.  Fetch their parent nodes using `parent_id` to add context.
    3.  Use a **Cross-Encoder model** (e.g., from `sentence-transformers`) to re-rank the expanded set of chunks against the `user_query`.
  * **Libraries:** `sentence-transformers`.

##### **Component 4.3: Synthesis & Dynamic Citation**

  * **Objective:** To generate a final answer that dynamically cites the correct source document.
  * **Input:** Top `M` re-ranked text chunks and the `user_query`.
  * **Output:** A final answer string with precise citations.
  * **Key Logic:**
    1.  Construct the final LLM prompt. The prompt must be updated to handle multiple sources.
    2.  The prompt will explicitly use the `notice_id` and `hierarchy` from the metadata for citations.
        ```python
        final_prompt = f"""
        You are an expert on regulations from the Monetary Authority of Singapore (MAS). Answer the user's question based ONLY on the provided context sections from various MAS notices. For each piece of information you use, you MUST cite the specific source notice and paragraph (e.g., "According to MAS Notice 758, Part I, Paragraph 3..."). If the answer is not in the context, state that clearly.

        User Question: "{user_query}"

        Context Sections:
        ---
        Context 1 (Source: {metadata1['notice_id']}, {metadata1['hierarchy']['part']}, Paragraph {metadata1['hierarchy']['paragraph']}):
        {text_chunk_1}
        ---
        Context 2 (Source: {metadata2['notice_id']}, {metadata2['hierarchy']['part']}, Paragraph {metadata2['hierarchy']['paragraph']}):
        {text_chunk_2}
        ---
        """
        ```
    3.  Send this prompt to the LLM and return the response.
  * **Libraries:** `langchain`.
