import google.generativeai as genai
import json
import os
import time

def enrich_and_embed(structured_data_path, output_path):
    """
    Enriches structured data with summaries, questions, and embeddings
    using the Google AI SDK.

    Args:
        structured_data_path (str): Path to the structured JSON file.
        output_path (str): Path to save the enriched data.
    """
    # 1. Configure the Google AI SDK
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set the environment variable and try again.")
        # We will request this from the user in the next step.
        # For now, we exit gracefully.
        return

    genai.configure(api_key=api_key)

    # 2. Load the structured data
    with open(structured_data_path, 'r') as f:
        data = json.load(f)

    enriched_nodes = []

    # Initialize models
    generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    embedding_model = "models/text-embedding-004"

    print(f"Starting enrichment for {len(data['content'])} nodes...")

    for i, node in enumerate(data['content']):
        text = node.get('text', '')
        if not text:
            continue

        print(f"  Processing node {i+1}/{len(data['content'])} ({node['node_id']})...")

        try:
            # 3. Generate Summary and Hypothetical Question
            summary_prompt = f"""
            You are a legal expert specializing in financial regulations.
            Summarize the following text from a Monetary Authority of Singapore (MAS) notice.
            Focus on the key requirements, obligations, and definitions.
            The summary should be concise and clear.

            TEXT: "{text}"
            """
            summary_response = generative_model.generate_content(summary_prompt)
            summary = summary_response.text
            time.sleep(1) # Basic rate limiting

            question_prompt = f"""
            You are a compliance officer at a bank in Singapore.
            Based on the following text from a MAS notice, what is one specific, hypothetical question
            you might ask to clarify your obligations or to test your understanding of the rule?
            The question should be practical and relevant to a banking context.

            TEXT: "{text}"
            """
            question_response = generative_model.generate_content(question_prompt)
            hypothetical_question = question_response.text
            time.sleep(1) # Basic rate limiting

            # 4. Generate Embeddings
            # We embed the original text, the summary, and the hypothetical question.
            content_embedding = genai.embed_content(model=embedding_model, content=text)['embedding']
            summary_embedding = genai.embed_content(model=embedding_model, content=summary)['embedding']
            question_embedding = genai.embed_content(model=embedding_model, content=hypothetical_question)['embedding']

            # 5. Assemble the enriched data object
            enriched_node = {
                "id": node['node_id'], # Pinecone uses 'id'
                "values": { # This structure can be adjusted for Pinecone
                    "content": content_embedding,
                    "summary": summary_embedding,
                    "question": question_embedding,
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

        except Exception as e:
            print(f"    An error occurred while processing node {node['node_id']}: {e}")
            # Continue to the next node
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
