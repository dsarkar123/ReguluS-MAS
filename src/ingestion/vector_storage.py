import os
import json
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

def store_vectors_chroma(enriched_data_path, collection_name="mas_notices"):
    """
    Stores enriched data and embeddings in a local ChromaDB collection.

    Args:
        enriched_data_path (str): Path to the enriched JSON file.
        collection_name (str): The name of the ChromaDB collection.
    """
    # 1. Initialize a persistent ChromaDB client
    if not os.path.exists('db'):
        os.makedirs('db')
    client = chromadb.PersistentClient(path="db")

    # 2. Define the embedding function from Google
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set for ChromaDB.")

    embedding_function = GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name="models/text-embedding-004"
    )

    # 3. Get or create the collection with the specified embedding function
    print(f"Getting or creating collection: {collection_name}")
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    print(f"Collection '{collection_name}' ready.")

    # 3. Load the enriched data
    with open(enriched_data_path, 'r') as f:
        enriched_nodes = json.load(f)

    if not enriched_nodes:
        print("No enriched nodes to process. Exiting.")
        return

    # 4. Prepare data for ChromaDB
    # ChromaDB's `add` method is very flexible. We can pass lists of
    # ids, embeddings, metadatas, and documents.

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    print(f"Preparing {len(enriched_nodes)} nodes for ChromaDB...")

    for node in enriched_nodes:
        # For simplicity, we will use the content embedding for now.
        # The other embeddings (summary, question) are in the metadata.
        # We also store the original text as the 'document'.

        # ChromaDB requires metadata values to be strings, numbers, or booleans.
        # The 'parent_id' can be None, which is not allowed. We'll convert it to a string.
        node['metadata']['parent_id'] = str(node['metadata'].get('parent_id', 'None'))

        ids.append(node['id'])
        embeddings.append(node['values']['content'])
        metadatas.append(node['metadata'])
        documents.append(node['metadata']['original_text'])

    # 5. Add the data to the collection
    # Using `upsert` is safer as it will add new documents and update existing ones.
    if ids:
        print(f"Adding {len(ids)} documents to the collection...")
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print("Data successfully added to ChromaDB.")
        print(f"Collection count: {collection.count()}")
    else:
        print("No valid data to add to the collection.")


if __name__ == '__main__':
    enriched_file = 'data/MAS_758_enriched.json'

    if os.path.exists(enriched_file):
        store_vectors_chroma(enriched_file)
    else:
        print(f"Error: Enriched data file not found at {enriched_file}")
        print("Please run the enrichment.py script first.")
