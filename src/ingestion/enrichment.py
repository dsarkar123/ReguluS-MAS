import google.generativeai as genai
import json
import os
import time
import re

def enrich_and_embed(structured_data_path, output_path):
    """
    Enriches structured data with summaries, questions, and embeddings
    using the Google AI SDK in a batch-efficient manner.

    Args:
        structured_data_path (str): Path to the structured JSON file.
        output_path (str): Path to save the enriched data.
    """
    # 1. Configure the Google AI SDK
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return

    genai.configure(api_key=api_key)

    # 2. Load the structured data
    with open(structured_data_path, 'r') as f:
        data = json.load(f)

    enriched_nodes = []

    # Initialize models
    generative_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    embedding_model = "models/text-embedding-004"

    print(f"Starting enrichment for {len(data['content'])} nodes...")

    for i, node in enumerate(data['content']):
        text = node.get('text', '')
        if not text:
            continue

        print(f"  Processing node {i+1}/{len(data['content'])} ({node['node_id']})...")

        try:
            # 3. Generate Summary and Question in a single call
            prompt = f"""
            You are a helpful AI assistant. Analyze the following text from a Monetary Authority of Singapore (MAS) notice.

            TEXT: "{text}"

            Based on the text, provide the following in a valid JSON format with two keys: "summary" and "hypothetical_question".
            1.  "summary": A concise summary of the key requirements, obligations, and definitions.
            2.  "hypothetical_question": A specific, practical question a compliance officer might ask.
            """

            response = generative_model.generate_content(prompt)

            # Clean up the response to extract only the JSON part
            cleaned_response = response.text.strip().replace('```json', '').replace('```', '')

            generated_data = json.loads(cleaned_response)
            summary = generated_data.get("summary", "Error: No summary found.")
            hypothetical_question = generated_data.get("hypothetical_question", "Error: No question found.")

            # 4. Generate Embeddings in a single batch call
            texts_to_embed = [text, summary, hypothetical_question]
            embedding_result = genai.embed_content(
                model=embedding_model,
                content=texts_to_embed,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = embedding_result['embedding']

            # 5. Assemble the enriched data object
            enriched_node = {
                "id": node['node_id'],
                "values": {
                    "content": embeddings[0],
                    "summary": embeddings[1],
                    "question": embeddings[2],
                },
                "metadata": {
                    "original_text": text,
                    "summary": summary,
                    "hypothetical_question": hypothetical_question,
                    "notice_id": data['metadata']['notice_id'],
                    "publication_date": data['metadata']['publication_date'],
                    "effective_date": data['metadata']['effective_date'],
                    "node_type": node['node_type'],
                    "parent_id": node.get('parent_id')
                }
            }

            enriched_nodes.append(enriched_node)
            time.sleep(1) # Add a small delay to respect rate limits

        except Exception as e:
            print(f"    An error occurred while processing node {node['node_id']}: {e}")
            continue

    # 6. Save the enriched data
    with open(output_path, 'w') as f:
        json.dump(enriched_nodes, f, indent=4)

    print(f"Enrichment complete. Enriched data saved to {output_path}")


if __name__ == '__main__':
    structured_file = 'data/MAS Notice 758_dated 18 Dec 2024_effective 26 Dec 2024_structured.json'
    enriched_file = 'data/MAS_758_enriched.json'

    if os.path.exists(structured_file):
        enrich_and_embed(structured_file, enriched_file)
    else:
        print(f"Error: Structured data file not found at {structured_file}")
        print("Please run the parser.py script first.")