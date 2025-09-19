import unittest
from unittest.mock import patch, MagicMock
import os
import json
import sys

# Add src to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ingestion.parser import parse_mas_notice

class TestParser(unittest.TestCase):

    @patch('src.ingestion.parser.fitz.open')
    def test_parser_with_mock_pdf(self, mock_fitz_open):
        """
        Tests the parser logic with a mocked PDF document.
        """
        # 1. Setup the mock
        mock_doc = MagicMock()
        mock_page = MagicMock()

        # Simulate the text extraction to return some sample lines
        mock_page.get_text.return_value = """
        1. This is the first paragraph.
        (a) This is a sub-paragraph.
        2. This is the second paragraph.
        """

        mock_doc.__iter__.return_value = [mock_page] # Make the document iterable
        mock_fitz_open.return_value = mock_doc

        # A dummy path, since fitz.open is mocked, it won't be used.
        dummy_pdf_path = "data/MAS Notice 758_dated 18 Dec 2024_effective 26 Dec 2024.pdf"

        # 2. Run the parser
        json_output = parse_mas_notice(dummy_pdf_path)
        data = json.loads(json_output)

        # 3. Assertions
        # Test metadata extraction from filename
        metadata = data.get("metadata", {})
        self.assertEqual(metadata.get("notice_id"), "MAS Notice 758")
        self.assertEqual(metadata.get("publication_date"), "18 Dec 2024")
        self.assertEqual(metadata.get("effective_date"), "26 Dec 2024")

        # Test content chunking
        content = data.get("content", [])
        self.assertEqual(len(content), 3)

        # Check node 1
        self.assertEqual(content[0]['node_type'], 'paragraph')
        self.assertIn("1. This is the first paragraph.", content[0]['text'])

        # Check node 2 (sub-paragraph)
        self.assertEqual(content[1]['node_type'], 'sub-paragraph')
        self.assertIn("(a) This is a sub-paragraph.", content[1]['text'])
        self.assertEqual(content[1]['parent_id'], content[0]['node_id']) # Check parent relationship

        # Check node 3
        self.assertEqual(content[2]['node_type'], 'paragraph')
        self.assertIn("2. This is the second paragraph.", content[2]['text'])
        self.assertIsNone(content[2]['parent_id']) # Should be a new top-level node


if __name__ == '__main__':
    unittest.main()
