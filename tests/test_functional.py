"""
Essential functional tests for document processor.
Tests vector storage and basic search functionality.
"""
import unittest
import tempfile
import os
import shutil
import sys
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processor import process_documents
import chromadb
from chromadb.config import Settings


class TestVectorStorageFunctionality(unittest.TestCase):
    """Test actual vector storage and retrieval functionality."""
    
    def setUp(self):
        """Set up test environment with real ChromaDB."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_docs_dir = os.path.join(self.temp_dir, "test_docs")
        os.makedirs(self.test_docs_dir)
        
        # Create test documents
        self.create_test_documents()
        
        # Use a separate ChromaDB instance for testing
        self.chroma_dir = os.path.join(self.temp_dir, "test_chroma")
        os.makedirs(self.chroma_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_documents(self):
        """Create test documents with known content."""
        # Document 1: Legal contract
        with open(os.path.join(self.test_docs_dir, "contract.pdf"), "w") as f:
            f.write("""
            SOFTWARE LICENSE AGREEMENT
            
            This agreement governs the use of our software product.
            The licensee agrees to use the software only for authorized purposes.
            Any unauthorized distribution is strictly prohibited.
            
            TERMS:
            1. License is non-transferable
            2. Software may not be reverse engineered
            3. All rights reserved by the licensor
            """)
        
        # Document 2: Privacy policy
        with open(os.path.join(self.test_docs_dir, "privacy.docx"), "w") as f:
            f.write("""
            PRIVACY POLICY
            
            We collect personal information to provide our services.
            Your data is protected and will not be shared with third parties.
            You can request data deletion at any time.
            
            DATA COLLECTION:
            - Email addresses for communication
            - Usage statistics for service improvement
            - Payment information for billing purposes
            """)
        
        # Document 3: Technical documentation
        with open(os.path.join(self.test_docs_dir, "tech_doc.doc"), "w") as f:
            f.write("""
            API DOCUMENTATION
            
            Our REST API provides endpoints for data management.
            Authentication is required for all API calls.
            Rate limiting applies to prevent abuse.
            
            ENDPOINTS:
            - GET /api/data - Retrieve user data
            - POST /api/data - Create new records
            - PUT /api/data - Update existing records
            - DELETE /api/data - Remove records
            """)
    
    
    def test_vector_storage_with_mock_api(self):
        """Test vector storage with mocked API."""
        # Mock the Gemini API and text extraction
        with unittest.mock.patch('document_processor.genai.configure'), \
             unittest.mock.patch('document_processor.genai.embed_content') as mock_embed, \
             unittest.mock.patch('document_processor.extract_text_from_file') as mock_extract:
            
            # Configure mock to return consistent embeddings and text
            mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3] * 256}  # 768 dimensions
            
            # Mock different content for different files
            def mock_extract_side_effect(file_path):
                if "contract" in file_path:
                    return "SOFTWARE LICENSE AGREEMENT\nThis agreement governs the use of our software product."
                elif "privacy" in file_path:
                    return "PRIVACY POLICY\nWe collect personal information to provide our services."
                else:
                    return "API DOCUMENTATION\nOur REST API provides endpoints for data management."
            
            mock_extract.side_effect = mock_extract_side_effect
            
            # Process documents
            result = process_documents(
                documents_path=self.test_docs_dir,
                chunk_size=200,
                chunk_overlap=50,
                embedding_model="gemini",
                api_key="test_key",
                collection_name="test_mock_storage"
            )
            
            # Verify processing completed
            self.assertEqual(result["processed_docs"], 3)
            self.assertGreater(result["total_chunks"], 0)
            
            # Verify vectors are stored in ChromaDB
            client = chromadb.PersistentClient(path="./chroma_db")
            
            collection = client.get_collection("test_mock_storage")
            
            # Check collection has data
            count = collection.count()
            self.assertGreater(count, 0)
            
            # Verify document content is stored correctly
            all_docs = collection.get()
            
            # Check that we have the expected content
            documents_text = ' '.join(all_docs['documents'])
            self.assertIn("SOFTWARE LICENSE", documents_text)
            self.assertIn("PRIVACY POLICY", documents_text)
            self.assertIn("API DOCUMENTATION", documents_text)
            
            # Verify metadata is stored
            for metadata in all_docs['metadatas']:
                self.assertIn('file_type', metadata)
                self.assertIn('chunk_size', metadata)
    
    def test_search_functionality(self):
        """Test basic search functionality."""
        with patch('document_processor.genai.configure'), \
             patch('document_processor.genai.embed_content') as mock_embed, \
             patch('document_processor.extract_text_from_file') as mock_extract:
            
            mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3] * 256}
            
            # Mock different content for different files
            def mock_extract_side_effect(file_path):
                if "contract" in file_path:
                    return "SOFTWARE LICENSE AGREEMENT\nThis agreement governs the use of our software product."
                elif "privacy" in file_path:
                    return "PRIVACY POLICY\nWe collect personal information to provide our services."
                else:
                    return "API DOCUMENTATION\nOur REST API provides endpoints for data management."
            
            mock_extract.side_effect = mock_extract_side_effect
            
            # Process documents
            result = process_documents(
                documents_path=self.test_docs_dir,
                chunk_size=200,
                chunk_overlap=50,
                embedding_model="gemini",
                api_key="test_key",
                collection_name="test_search"
            )
            
            # Test search functionality
            client = chromadb.PersistentClient(path="./chroma_db")
            
            collection = client.get_collection("test_search")
            
            # Test search with proper embedding (use the same embedding as stored)
            test_embedding = [0.1, 0.2, 0.3] * 256  # 768 dimensions
            search_results = collection.query(
                query_embeddings=[test_embedding],
                n_results=3,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Verify search returns results
            self.assertGreater(len(search_results['documents'][0]), 0)
    
    def test_persistence_across_sessions(self):
        """Test that stored vectors persist across ChromaDB sessions."""
        with patch('document_processor.genai.configure'), \
             patch('document_processor.genai.embed_content') as mock_embed, \
             patch('document_processor.extract_text_from_file') as mock_extract:
            
            mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3] * 256}
            
            # Mock different content for different files
            def mock_extract_side_effect(file_path):
                if "contract" in file_path:
                    return "SOFTWARE LICENSE AGREEMENT\nThis agreement governs the use of our software product."
                elif "privacy" in file_path:
                    return "PRIVACY POLICY\nWe collect personal information to provide our services."
                else:
                    return "API DOCUMENTATION\nOur REST API provides endpoints for data management."
            
            mock_extract.side_effect = mock_extract_side_effect
            
            result = process_documents(
                documents_path=self.test_docs_dir,
                chunk_size=200,
                chunk_overlap=50,
                embedding_model="gemini",
                api_key="test_key",
                collection_name="test_persistence_unique"
            )
            
            initial_count = result["total_chunks"]
        
        # Simulate new session by creating new ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Verify data persists
        collection = client.get_collection("test_persistence_unique")
        persisted_count = collection.count()
        
        self.assertEqual(persisted_count, initial_count)


if __name__ == '__main__':
    # Set up test environment
    os.environ['CHROMA_DB_PATH'] = './chroma_db'
    
    # Run tests
    unittest.main()
