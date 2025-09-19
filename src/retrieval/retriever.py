import os
import re
import chromadb
import google.generativeai as genai
import json
import time

class Retriever:
    def __init__(self, collection_name="mas_notices", db_path="db"):
        # Initialize clients and models
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        print(f"Connected to ChromaDB. Collection '{collection_name}' has {self.collection.count()} documents.")

        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=self.api_key)
        self.embedding_model = "models/text-embedding-004"
        self.generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def _search(self, user_query, n_results=10, doc_filter=None):
        """Internal method to perform the initial vector search using ChromaDB."""
        print(f"Embedding query: '{user_query}'")
        query_embedding = genai.embed_content(
            model=self.embedding_model,
            content=user_query
        )['embedding']

        print(f"Searching collection for top {n_results} results...")
        query_params = {
            'query_embeddings': [query_embedding],
            'n_results': n_results
        }
        if doc_filter:
            query_params['where'] = doc_filter

        results = self.collection.query(**query_params)
        return results

    def _expand_context(self, search_results):
        """Expands context by fetching parent nodes."""
        print("Expanding context by fetching parent nodes...")
        # Using a dictionary to ensure uniqueness of documents
        expanded_docs = {}
        parent_ids_to_fetch = set()

        # Add original results to the dict and gather parent IDs
        for i, metadata in enumerate(search_results['metadatas'][0]):
            doc_id = search_results['ids'][0][i]
            expanded_docs[doc_id] = {'metadata': metadata, 'text': search_results['documents'][0][i]}
            parent_id = metadata.get('parent_id')
            if parent_id and parent_id != 'None':
                parent_ids_to_fetch.add(parent_id)

        # Fetch parent documents if any
        if parent_ids_to_fetch:
            print(f"Fetching {len(parent_ids_to_fetch)} parent documents...")
            # Note: ChromaDB's get() might not return items in the same order as the ids list.
            parents_data = self.collection.get(ids=list(parent_ids_to_fetch))
            for i, parent_id in enumerate(parents_data['ids']):
                if parent_id not in expanded_docs:
                     expanded_docs[parent_id] = {'metadata': parents_data['metadatas'][i], 'text': parents_data['documents'][i]}

        # Return a list of unique documents
        return list(expanded_docs.values())

    def _rerank_with_gemini(self, query, documents, top_n=5):
        """Re-ranks documents based on relevance to the query using Gemini."""
        print(f"Re-ranking {len(documents)} documents with Gemini...")
        ranked_docs = []
        for doc in documents:
            prompt = f"""
            Score the relevance of the following document to the user's query.
            The score should be an integer from 1 (not relevant) to 10 (highly relevant).
            Return ONLY the integer score.

            User Query: "{query}"
            ---
            Document (Source: {doc['metadata']['notice_id']}, Type: {doc['metadata']['node_type']}):
            "{doc['text']}"
            """
            try:
                response = self.generative_model.generate_content(prompt)
                time.sleep(1) # Basic rate limiting
                score = int(re.search(r'\d+', response.text).group())
                ranked_docs.append({'doc': doc, 'score': score})
                print(f"  Scored document {doc['metadata']['notice_id']} ({doc['metadata']['node_type']}) with relevance: {score}")
            except Exception as e:
                print(f"  Could not score document: {e}. Assigning score of 0.")
                ranked_docs.append({'doc': doc, 'score': 0})

        ranked_docs.sort(key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in ranked_docs[:top_n]]

    def synthesize_answer(self, query, ranked_documents):
        """Generates a final answer using Gemini, with citations."""
        print("Synthesizing final answer...")

        context_str = ""
        for i, doc in enumerate(ranked_documents):
            meta = doc['metadata']
            context_str += f"--- Context {i+1} (Source: {meta['notice_id']}, Type: {meta['node_type']}, Original ID: {meta.get('parent_id', 'N/A')}) ---\n"
            context_str += doc['text'] + "\n\n"

        final_prompt = f"""
        You are an expert on MAS regulations. Answer the user's question based ONLY on the provided context sections.
        For each piece of information you use, you MUST cite the specific source notice and paragraph (e.g., "According to MAS Notice 758, ...").
        If the answer is not in the context, state that clearly.

        User Question: "{query}"

        Context Sections:
        {context_str}
        """

        response = self.generative_model.generate_content(final_prompt)
        return response.text

    def full_retrieval(self, user_query, n_results=10, top_n_rerank=3, doc_filter=None):
        """Orchestrates the full retrieval pipeline."""
        # 1. Search
        search_results = self._search(user_query, n_results, doc_filter)
        if not search_results or not search_results['documents'][0]:
            return "Could not find any relevant documents."

        # Create a list of document objects from the search results
        initial_docs = [{'metadata': search_results['metadatas'][0][i], 'text': search_results['documents'][0][i]} for i in range(len(search_results['ids'][0]))]

        # 2. Re-rank
        reranked_docs = self._rerank_with_gemini(user_query, initial_docs, top_n=top_n_rerank)

        # 3. Synthesize
        final_answer = self.synthesize_answer(user_query, reranked_docs)

        return final_answer


if __name__ == '__main__':
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: Please set the GOOGLE_API_KEY environment variable to run this example.")
    else:
        try:
            retriever = Retriever()
            sample_query = "What is the minimum cash balance requirement for banks?"

            print("\n--- Running Full RAG Pipeline ---")
            final_answer = retriever.full_retrieval(sample_query)

            print("\n\n--- Final Answer ---")
            print(final_answer)

        except Exception as e:
            print(f"An error occurred during the full pipeline execution: {e}")
