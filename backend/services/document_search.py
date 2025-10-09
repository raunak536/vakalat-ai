"""
Simple document-to-document search service using mean averaging of embeddings.
This service takes a document path and returns similar documents from ChromaDB.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
import google.generativeai as genai
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
# Import from existing document processor (make path relative)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from document_processor import extract_text_from_file 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CHROMA_DB_PATH = str(PROJECT_ROOT / "chroma_db")


def _generate_embedding_worker(args):
    """
    Worker function for multiprocessing embedding generation.
    This function needs to be at module level for multiprocessing to work.
    
    Args:
        args: Tuple of (chunk_text, model_name, api_key)
        
    Returns:
        Embedding vector or None if error
    """
    chunk_text, model_name = args
    
    try:
        # Configure Gemini API for this worker process
        result = genai.embed_content(
            model=model_name,
            content=chunk_text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Error generating embedding in worker: {e}")
        return None


class DocumentSearchService:
    """
    Simple document search service that finds similar documents using mean averaging.
    
    The approach:
    1. Extract text from input document
    2. Split into chunks and generate embeddings
    3. Compute mean of all chunk embeddings
    4. Search ChromaDB for similar documents
    """
    
    def __init__(self, chroma_db_path: str = DEFAULT_CHROMA_DB_PATH, collection_name: str = "legal_documents", max_workers: Optional[int] = None):
        """
        Initialize the document search service.
        
        Args:
            chroma_db_path: Path to ChromaDB directory
            collection_name: Name of the collection to search
            max_workers: Maximum number of parallel workers (default: CPU count)
        """
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = "models/embedding-001"  # Gemini embedding model
        self.max_workers = max_workers or min(cpu_count(), 10)  # Cap at 8 to avoid API rate limits
        
        # Store API key for worker processes        
        logger.info(f"DocumentSearchService initialized with {self.max_workers} workers")
        
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection."""
        if self.client is None:
            self.client = chromadb.PersistentClient(path=self.chroma_db_path)
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
            # Avoid infinite loop
            if start >= len(text):
                break
                
        return chunks
    
    def _generate_document_embedding_parallel(self, text: str) -> List[float]:
        """
        Generate document embedding using parallel processing and mean averaging.
        
        Args:
            text: Document text
            
        Returns:
            Mean embedding vector
        """
        # Split document into chunks
        chunks = self._chunk_text(text)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        if not chunks:
            logger.error("No chunks created from document")
            return [0.0] * 768
        
        # Prepare arguments for worker processes
        worker_args = [
            (chunk, self.embedding_model) 
            for chunk in chunks
        ]
        
        # Generate embeddings in parallel
        logger.info(f"Generating embeddings using {self.max_workers} parallel workers...")
        
        try:
            with Pool(processes=self.max_workers) as pool:
                chunk_embeddings = pool.map(_generate_embedding_worker, worker_args)
            
            # Filter out None results (failed embeddings)
            valid_embeddings = [emb for emb in chunk_embeddings if emb is not None]
            
            if not valid_embeddings:
                logger.error("No valid embeddings generated")
                return [0.0] * 768
            
            logger.info(f"Successfully generated {len(valid_embeddings)}/{len(chunks)} embeddings")
            
            # Compute mean of all chunk embeddings
            mean_embedding = np.mean(valid_embeddings, axis=0).tolist()
            logger.info(f"Computed mean embedding from {len(valid_embeddings)} chunks")
            
            return mean_embedding
            
        except Exception as e:
            logger.error(f"Error in parallel embedding generation: {e}")
            return [0.0] * 768
    
    def _generate_document_embedding(self, text: str) -> List[float]:
        """
        Generate document embedding using mean averaging of chunk embeddings.
        This is the fallback sequential method.
        
        Args:
            text: Document text
            
        Returns:
            Mean embedding vector
        """
        # Split document into chunks
        chunks = self._chunk_text(text)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Generate embeddings for each chunk
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = self.generate_embedding(chunk, self.embedding_model)
                chunk_embeddings.append(embedding)
                logger.info(f"Generated embedding for chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {i}: {e}")
                continue
        
        if not chunk_embeddings:
            logger.error("No embeddings generated")
            return [0.0] * 768  # Fallback zero vector
        
        # Compute mean of all chunk embeddings
        mean_embedding = np.mean(chunk_embeddings, axis=0).tolist()
        logger.info(f"Computed mean embedding from {len(chunk_embeddings)} chunks")
        
        return mean_embedding
    
    def generate_embedding(self, text: str, model_name: str) -> List[float]:
        """Generate embedding for text using Gemini embedding model."""
        try:
            # Use Gemini's embedding API
            result = genai.embed_content(
                model=model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback (768 dimensions for gemini-embedding-001)
            return [0.0] * 768

    def search_similar_documents(
        self, 
        document_path: str, 
        top_k: int = 5,
        chunk_size: int = 1000,
        use_parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to the input document.
        
        Args:
            document_path: Path to the input document
            top_k: Number of similar documents to return
            chunk_size: Size of chunks for embedding generation
            use_parallel: Whether to use parallel processing for embeddings
            
        Returns:
            List of similar documents with metadata and similarity scores
        """
        # Initialize ChromaDB
        self._initialize_chroma()
        
        # Extract text from input document
        logger.info(f"Extracting text from: {document_path}")
        try:
            text = extract_text_from_file(document_path)
            if not text.strip():
                logger.error("No text extracted from document")
                return []
        except Exception as e:
            logger.error(f"Error extracting text from document: {e}")
            return []
        
        # Generate document embedding using mean averaging
        logger.info("Generating document embedding...")
        if use_parallel:
            query_embedding = self._generate_document_embedding_parallel(text)
        else:
            query_embedding = self._generate_document_embedding(text)
        
        # Search for similar documents
        logger.info(f"Searching for top {top_k} similar documents...")
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            similar_docs = []
            for i in range(len(results['ids'][0])):
                doc_info = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                similar_docs.append(doc_info)
            
            logger.info(f"Found {len(similar_docs)} similar documents")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []


def search_documents(document_path: str, top_k: int = 5, use_parallel: bool = True) -> List[Dict[str, Any]]:
    """
    Simple function to search for similar documents.
    
    Args:
        document_path: Path to the input document
        top_k: Number of similar documents to return
        use_parallel: Whether to use parallel processing for embeddings
        
    Returns:
        List of similar documents
    """
    service = DocumentSearchService()
    return service.search_similar_documents(document_path, top_k, use_parallel=use_parallel)


# Example usage
if __name__ == "__main__":
    document_path = "./backend/data/documents/1/232602.json"
    # Example: search for documents similar to a specific file
    
    if os.path.exists(document_path):
        results = search_documents(document_path, top_k=3, use_parallel=True)
        
        print(f"\nFound {len(results)} similar documents:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Document ID: {doc['id']}")
            print(f"   Similarity: {doc['similarity']:.3f}")
            print(f"   Metadata: {doc['metadata']}")
            print(f"   Preview: {doc['document'][:200]}...")
    else:
        print(f"Document not found: {document_path}")
