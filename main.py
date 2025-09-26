import argparse
import os
import sys

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from ingestion.parser import parse_mas_notice
from ingestion.enrichment import enrich_and_embed
from ingestion.vector_storage import store_vectors_chroma
from retrieval.retriever import Retriever

def run_ingestion(pdf_path):
    """
    Runs the full ingestion pipeline for a given PDF file.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    # Create the data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    print(f"--- Starting Ingestion for {pdf_path} ---")

    # 1. Parsing
    print("\nStep 1: Parsing PDF...")
    base_filename = os.path.basename(pdf_path).replace('.pdf', '')
    structured_output_path = f"data/{base_filename}_structured.json"

    json_output = parse_mas_notice(pdf_path)
    with open(structured_output_path, 'w') as f:
        f.write(json_output)
    print(f"Parsing complete. Structured data saved to {structured_output_path}")

    # 2. Enrichment
    print("\nStep 2: Enriching data with Google AI...")
    enriched_output_path = f"data/{base_filename}_enriched.json"
    enrich_and_embed(structured_output_path, enriched_output_path)
    print(f"Enrichment complete. Enriched data saved to {enriched_output_path}")

    # 3. Vector Storage
    print("\nStep 3: Storing vectors in ChromaDB...")
    store_vectors_chroma(enriched_output_path)
    print("Vector storage complete.")

    print("\n--- Ingestion Pipeline Finished ---")

def run_query(query):
    """
    Runs the full retrieval pipeline for a given query.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable must be set for querying.")
        return

    print(f"--- Running Query: '{query}' ---")

    try:
        retriever = Retriever()
        final_answer = retriever.full_retrieval(query)

        print("\n--- Final Answer ---")
        print(final_answer)
    except Exception as e:
        print(f"An error occurred during query execution: {e}")

def main():
    parser = argparse.ArgumentParser(description="A RAG system for MAS Circulars.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Run the full ingestion pipeline for a PDF.")
    ingest_parser.add_argument("pdf_path", type=str, help="The path to the MAS notice PDF file.")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question to the RAG system.")
    query_parser.add_argument("query_text", type=str, help="The question you want to ask.")

    args = parser.parse_args()

    if args.command == "ingest":
        # Check for GOOGLE_API_KEY before starting ingestion as it's needed for enrichment
        if not os.environ.get("GOOGLE_API_KEY"):
            print("Error: GOOGLE_API_KEY environment variable must be set for the enrichment step.")
            return
        run_ingestion(args.pdf_path)
    elif args.command == "query":
        run_query(args.query_text)

if __name__ == "__main__":
    main()
