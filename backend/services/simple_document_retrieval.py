"""
Simple document retrieval service for query-based search.
This service takes a text query and returns the most relevant documents from ChromaDB.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CHROMA_DB_PATH = str(PROJECT_ROOT / "chroma_db")


class SimpleDocumentRetrieval:
    """
    Simple document retrieval service that finds relevant documents for a given query.
    
    The approach:
    1. Take a text query from user
    2. Generate embedding for the query
    3. Search ChromaDB for most relevant document chunks
    4. Return top results with metadata
    """
    
    def __init__(self, chroma_db_path: str = DEFAULT_CHROMA_DB_PATH, collection_name: str = "legal_documents"):
        """
        Initialize the simple document retrieval service.
        
        Args:
            chroma_db_path: Path to ChromaDB directory
            collection_name: Name of the collection to search
        """
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = "models/embedding-001"  # Gemini embedding model
        
        logger.info(f"SimpleDocumentRetrieval initialized")
        
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection."""
        if self.client is None:
            self.client = chromadb.PersistentClient(path=self.chroma_db_path)
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for the search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"  # Use query task type for better results
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            # Return zero vector as fallback (768 dimensions for gemini-embedding-001)
            return [0.0] * 768

    def search_documents(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find documents relevant to the given query.
        
        Args:
            query: Search query text
            top_k: Number of relevant documents to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of relevant documents with metadata and similarity scores
        """
        # Initialize ChromaDB
        self._initialize_chroma()
        
        if not query.strip():
            logger.error("Empty query provided")
            return []
        
        # Generate query embedding
        logger.info(f"Generating embedding for query: '{query[:100]}...'")
        query_embedding = self._generate_query_embedding(query)
        
        # Search for relevant documents
        logger.info(f"Searching for top {top_k} relevant documents...")
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            relevant_docs = []
            for i in range(len(results['ids'][0])):
                similarity = 1 - results['distances'][0][i]  # Convert distance to similarity
                
                # Filter by minimum similarity if specified
                if similarity < min_similarity:
                    continue
                
                doc_info = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': similarity
                }
                relevant_docs.append(doc_info)
            
            logger.info(f"Found {len(relevant_docs)} relevant documents (min_similarity: {min_similarity})")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []

    def search_with_filters(
        self, 
        query: str, 
        top_k: int = 5,
        court_filter: Optional[str] = None,
        date_filter: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find documents relevant to the query with optional filters.
        
        Args:
            query: Search query text
            top_k: Number of relevant documents to return
            court_filter: Optional court name to filter by
            date_filter: Optional date to filter by
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of relevant documents with metadata and similarity scores
        """
        # Initialize ChromaDB
        self._initialize_chroma()
        
        if not query.strip():
            logger.error("Empty query provided")
            return []
        
        # Generate query embedding
        logger.info(f"Generating embedding for query: '{query[:100]}...'")
        query_embedding = self._generate_query_embedding(query)
        
        # Build where clause for filtering
        where_clause = {}
        if court_filter:
            where_clause["court"] = {"$eq": court_filter}
        if date_filter:
            where_clause["date"] = {"$eq": date_filter}
        
        # Search for relevant documents with filters
        logger.info(f"Searching for top {top_k} relevant documents with filters...")
        try:
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ['documents', 'metadatas', 'distances']
            }
            
            # Add where clause if filters are provided
            if where_clause:
                query_kwargs["where"] = where_clause
                logger.info(f"Applied filters: {where_clause}")
            
            results = self.collection.query(**query_kwargs)
            
            # Format results
            relevant_docs = []
            for i in range(len(results['ids'][0])):
                similarity = 1 - results['distances'][0][i]  # Convert distance to similarity
                
                # Filter by minimum similarity if specified
                if similarity < min_similarity:
                    continue
                
                doc_info = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': similarity
                }
                relevant_docs.append(doc_info)
            
            logger.info(f"Found {len(relevant_docs)} relevant documents with filters")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB with filters: {e}")
            return []


def search_documents(query: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
    """
    Simple function to search for relevant documents.
    
    Args:
        query: Search query text
        top_k: Number of relevant documents to return
        min_similarity: Minimum similarity threshold (0.0 to 1.0)
        
    Returns:
        List of relevant documents
    """
    service = SimpleDocumentRetrieval()
    return service.search_documents(query, top_k, min_similarity)


def search_documents_with_filters(
    query: str, 
    top_k: int = 5,
    court_filter: Optional[str] = None,
    date_filter: Optional[str] = None,
    min_similarity: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Simple function to search for relevant documents with filters.
    
    Args:
        query: Search query text
        top_k: Number of relevant documents to return
        court_filter: Optional court name to filter by
        date_filter: Optional date to filter by
        min_similarity: Minimum similarity threshold (0.0 to 1.0)
        
    Returns:
        List of relevant documents
    """
    service = SimpleDocumentRetrieval()
    return service.search_with_filters(query, top_k, court_filter, date_filter, min_similarity)


# Example usage
if __name__ == "__main__":
    # Example: search for documents related to privacy rights
    query = "right to privacy constitutional law"
    
    print("🔍 Simple Document Retrieval Example")
    print("=" * 50)
    print(f"Query: '{query}'")
    print()
    
    try:
        # Basic search
        results = search_documents(query, top_k=3, min_similarity=0.3)
        
        if not results:
            print("❌ No relevant documents found")
        else:
            print(f"✅ Found {len(results)} relevant documents:")
            print()
            
            for i, doc in enumerate(results, 1):
                print(f"{i}. Document ID: {doc['id']}")
                print(f"   Similarity: {doc['similarity']:.3f}")
                print(f"   Metadata: {doc['metadata']}")
                print(f"   Preview: {doc['document'][:200]}...")
                print()
        
        # Search with court filter
        print("\n" + "=" * 50)
        print("Search with court filter (Supreme Court of India):")
        print()
        
        filtered_results = search_documents_with_filters(
            query, 
            top_k=2, 
            court_filter="Supreme Court of India",
            min_similarity=0.3
        )
        
        if not filtered_results:
            print("❌ No relevant documents found with court filter")
        else:
            print(f"✅ Found {len(filtered_results)} relevant documents from Supreme Court:")
            print()
            
            for i, doc in enumerate(filtered_results, 1):
                print(f"{i}. Document ID: {doc['id']}")
                print(f"   Similarity: {doc['similarity']:.3f}")
                print(f"   Court: {doc['metadata'].get('court', 'Unknown')}")
                print(f"   Preview: {doc['document'][:200]}...")
                print()
                
    except Exception as e:
        print(f"❌ Error during search: {e}")
        print("\nMake sure:")
        print("1. ChromaDB is set up with documents")
        print("2. The collection name matches your data")
        print("3. You have the required dependencies installed")
        print("4. GEMINI_API_KEY is set in your .env file")
