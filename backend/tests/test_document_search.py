"""
Simple test for document search functionality.
Tests the basic document-to-document search using mean averaging.
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.document_search import DocumentSearchService, search_documents


class TestDocumentSearch(unittest.TestCase):
    """Test document search functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_doc_path = os.path.join(self.temp_dir, "test_doc.txt")
        
        # Create a simple test document
        with open(self.test_doc_path, 'w') as f:
            f.write("This is a test document about legal matters and court proceedings.")
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_chunk_text(self):
        """Test text chunking functionality."""
        service = DocumentSearchService()
        
        # Test with short text
        short_text = "Short text"
        chunks = service._chunk_text(short_text, chunk_size=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short_text)
        
        # Test with longer text
        long_text = "A" * 2500  # 2500 characters
        chunks = service._chunk_text(long_text, chunk_size=1000, overlap=200)
        self.assertGreater(len(chunks), 1)
        
        # Check overlap
        if len(chunks) > 1:
            overlap_region = chunks[0][-200:]
            self.assertEqual(overlap_region, chunks[1][:200])
    
    @patch('services.document_search.generate_embedding')
    def test_generate_document_embedding(self, mock_embedding):
        """Test document embedding generation with mean averaging."""
        # Mock the embedding function to return consistent vectors
        mock_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5] + [0.0] * 763  # 768 dim vector
        
        service = DocumentSearchService()
        text = "This is a test document with some content that should be chunked."
        
        embedding = service._generate_document_embedding(text)
        
        # Should return a 768-dimensional vector
        self.assertEqual(len(embedding), 768)
        
        # Should be called for each chunk
        self.assertGreater(mock_embedding.call_count, 0)
    
    @patch('services.document_search.extract_text_from_file')
    @patch('services.document_search.generate_embedding')
    def test_search_similar_documents(self, mock_embedding, mock_extract):
        """Test the main search functionality."""
        # Mock text extraction
        mock_extract.return_value = "Test document content"
        
        # Mock embedding generation
        mock_embedding.return_value = [0.1] * 768
        
        # Mock ChromaDB collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['Document 1 content', 'Document 2 content']],
            'metadatas': [[{'title': 'Doc 1'}, {'title': 'Doc 2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        service = DocumentSearchService()
        service.collection = mock_collection
        
        results = service.search_similar_documents(self.test_doc_path, top_k=2)
        
        # Should return 2 results
        self.assertEqual(len(results), 2)
        
        # Check result structure
        for result in results:
            self.assertIn('id', result)
            self.assertIn('document', result)
            self.assertIn('metadata', result)
            self.assertIn('distance', result)
            self.assertIn('similarity', result)
        
        # Check similarity calculation
        self.assertEqual(results[0]['similarity'], 0.9)  # 1 - 0.1
        self.assertEqual(results[1]['similarity'], 0.8)  # 1 - 0.2


if __name__ == '__main__':
    unittest.main()
