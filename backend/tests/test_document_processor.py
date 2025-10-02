import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path to import document_processor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processor import (
    process_documents,
    extract_text_from_file,
    chunk_text,
    generate_embedding
)


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for document processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_documents_dir = os.path.join(self.temp_dir, "test_documents")
        os.makedirs(self.test_documents_dir)
        
        # Create test text files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        # Clean up any ChromaDB files created during tests
        chroma_dir = "./chroma_db"
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
    
    def create_test_files(self):
        """Create test files for testing."""
        # Create a simple text file
        with open(os.path.join(self.test_documents_dir, "test.txt"), "w") as f:
            f.write("This is a test document for unit testing.")
        
        # Create a Word document (simulated)
        with open(os.path.join(self.test_documents_dir, "test.docx"), "w") as f:
            f.write("This is a test Word document.")
    
    def test_chunk_text_basic(self):
        """Test basic text chunking functionality."""
        text = "This is a test document. " * 50  # Create long text
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)
        
        # Check that chunks don't exceed size limit
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)
    
    def test_chunk_text_short(self):
        """Test chunking with text shorter than chunk size."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
    
    def test_chunk_text_empty(self):
        """Test chunking with empty text."""
        chunks = chunk_text("", chunk_size=100, chunk_overlap=20)
        self.assertEqual(len(chunks), 1)  # Empty string returns one empty chunk
    
    def test_extract_text_from_file_txt(self):
        """Test text extraction from text file."""
        text = extract_text_from_file(os.path.join(self.test_documents_dir, "test.txt"))
        self.assertIn("This is a test document for unit testing.", text)
    
    def test_extract_text_from_file_nonexistent(self):
        """Test text extraction from non-existent file."""
        text = extract_text_from_file("nonexistent.txt")
        self.assertEqual(text, "")
    
    
    @patch('document_processor.genai.embed_content')
    def test_generate_embedding_success(self, mock_embed):
        """Test successful embedding generation."""
        mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3] * 256}  # 768 dimensions
        
        embedding = generate_embedding("test text", "models/embedding-001")
        
        self.assertEqual(len(embedding), 768)
        self.assertEqual(embedding[:3], [0.1, 0.2, 0.3])
        mock_embed.assert_called_once()
    
    @patch('document_processor.genai.embed_content')
    def test_generate_embedding_failure(self, mock_embed):
        """Test embedding generation failure."""
        mock_embed.side_effect = Exception("API Error")
        
        embedding = generate_embedding("test text", "models/embedding-001")
        
        self.assertEqual(len(embedding), 768)
        self.assertEqual(embedding, [0.0] * 768)
    
    @patch('document_processor.genai.configure')
    @patch('document_processor.chromadb.Client')
    @patch('document_processor.generate_embedding')
    def test_process_documents_success(self, mock_generate_embedding, mock_chromadb_client, mock_configure):
        """Test successful document processing."""
        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb_client.return_value = mock_client
        
        # Mock embedding generation
        mock_generate_embedding.return_value = [0.1] * 768
        
        # Create a test document
        test_file = os.path.join(self.test_documents_dir, "test.txt")
        
        result = process_documents(
            documents_path=self.test_documents_dir,
            chunk_size=50,
            chunk_overlap=10,
            embedding_model="gemini",
            api_key="test_api_key",
            collection_name="test_collection"
        )
        
        self.assertEqual(result["processed_docs"], 1)
        self.assertGreater(result["total_chunks"], 0)
        self.assertEqual(result["collection_name"], "test_collection")
        mock_configure.assert_called_once_with(api_key="test_api_key")
        mock_collection.add.assert_called_once()
    
    @patch('document_processor.genai.configure')
    @patch('document_processor.chromadb.Client')
    def test_process_documents_no_documents(self, mock_chromadb_client, mock_configure):
        """Test processing with no valid documents."""
        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb_client.return_value = mock_client
        
        # Create empty directory
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir)
        
        result = process_documents(
            documents_path=empty_dir,
            embedding_model="gemini",
            api_key="test_api_key"
        )
        
        self.assertEqual(result["message"], "No valid documents found")
        self.assertEqual(result["processed_docs"], 0)
    
    def test_process_documents_no_api_key(self):
        """Test processing without API key."""
        with self.assertRaises(ValueError) as context:
            process_documents(
                documents_path=self.test_documents_dir,
                embedding_model="gemini",
                api_key=None
            )
        
        self.assertIn("API key is required", str(context.exception))
    
    def test_process_documents_unsupported_model(self):
        """Test processing with unsupported embedding model."""
        with self.assertRaises(ValueError) as context:
            process_documents(
                documents_path=self.test_documents_dir,
                embedding_model="unsupported",
                api_key="test_key"
            )
        
        self.assertIn("Unsupported embedding model", str(context.exception))




if __name__ == '__main__':
    unittest.main()
