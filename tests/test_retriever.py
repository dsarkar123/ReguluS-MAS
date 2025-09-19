import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add src to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.retrieval.retriever import Retriever

class TestRetriever(unittest.TestCase):

    def setUp(self):
        """Set up a mock environment for each test."""
        self.env_patch = patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'})
        self.env_patch.start()

        self.configure_patch = patch('src.retrieval.retriever.genai.configure')
        self.mock_configure = self.configure_patch.start()

        self.chroma_patch = patch('src.retrieval.retriever.chromadb.PersistentClient')
        self.mock_chroma_client = self.chroma_patch.start()
        self.mock_collection = MagicMock()
        self.mock_collection.count.return_value = 1
        self.mock_chroma_client.return_value.get_collection.return_value = self.mock_collection

        self.model_patch = patch('src.retrieval.retriever.genai.GenerativeModel')
        self.mock_generative_model_class = self.model_patch.start()
        self.mock_generative_model_instance = MagicMock()
        self.mock_generative_model_class.return_value = self.mock_generative_model_instance

        self.retriever = Retriever()

    def tearDown(self):
        """Clean up all patches after each test."""
        self.env_patch.stop()
        self.configure_patch.stop()
        self.chroma_patch.stop()
        self.model_patch.stop()

    @patch('src.retrieval.retriever.genai.embed_content')
    def test_search(self, mock_embed_content):
        """Tests the internal search method."""
        mock_embed_content.return_value = {'embedding': [0.1] * 768}

        mock_query_results = {
            'ids': [['node_1']],
            'documents': [['This is a test document.']],
            'metadatas': [[{'notice_id': 'TEST_001'}]],
            'distances': [[0.5]]
        }
        self.mock_collection.query.return_value = mock_query_results

        results = self.retriever._search("test query", n_results=1)

        mock_embed_content.assert_called_once_with(model=self.retriever.embedding_model, content="test query")
        self.mock_collection.query.assert_called_once()
        self.assertEqual(results['ids'][0][0], 'node_1')

    def test_rerank_with_gemini(self):
        """Tests the re-ranking logic."""
        mock_response_10 = MagicMock()
        mock_response_10.text = "10"
        mock_response_5 = MagicMock()
        mock_response_5.text = "5"

        self.mock_generative_model_instance.generate_content.side_effect = [mock_response_10, mock_response_5]

        # Add the 'node_type' key to the metadata
        docs = [
            {'text': 'highly relevant doc', 'metadata': {'notice_id': 'doc1', 'node_type': 'paragraph'}},
            {'text': 'less relevant doc', 'metadata': {'notice_id': 'doc2', 'node_type': 'paragraph'}},
        ]

        reranked = self.retriever._rerank_with_gemini("query", docs, top_n=2)

        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]['metadata']['notice_id'], 'doc1')
        self.assertEqual(reranked[1]['metadata']['notice_id'], 'doc2')

    def test_synthesize_answer(self):
        """Tests the final answer synthesis."""
        mock_response = MagicMock()
        mock_response.text = "This is the synthesized answer."
        self.mock_generative_model_instance.generate_content.return_value = mock_response

        docs = [
            {'text': 'doc1 text', 'metadata': {'notice_id': 'MAS 1', 'node_type': 'para', 'parent_id': 'None'}},
        ]

        answer = self.retriever.synthesize_answer("query", docs)

        self.mock_generative_model_instance.generate_content.assert_called_once()
        call_args = self.mock_generative_model_instance.generate_content.call_args
        prompt = call_args[0][0]
        self.assertIn("User Question: \"query\"", prompt)
        self.assertIn("Context 1 (Source: MAS 1", prompt)
        self.assertIn("doc1 text", prompt)
        self.assertEqual(answer, "This is the synthesized answer.")

if __name__ == '__main__':
    unittest.main()
