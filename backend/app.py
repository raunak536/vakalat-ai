"""
FastAPI Chatbot Application for Legal Document Search

This application provides endpoints for:
1. Text query search - Search for relevant cases using natural language queries
2. Document upload search - Upload a document and find similar cases

Key FastAPI concepts used:
- @app.post() decorator: Defines HTTP POST endpoints
- File uploads: Using UploadFile for document uploads
- Response models: Simple dict responses (no Pydantic models as requested)
- Dependency injection: Services are initialized once and reused
"""

import os
from pydoc import doc
import tempfile
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import our existing services
from services.document_search import DocumentSearchService
from services.document_processor import extract_text_from_file, generate_embedding
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vakalat AI Chatbot",
    description="Legal document search chatbot using vector similarity",
    version="1.0.0"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for services (initialized once)
search_service = None
chroma_client = None
collection = None

def generate_match_reason(query: str, document_content: str, case_title: str) -> str:
    """
    Generate a meaningful reason for why a document matches a query using LLM.
    
    Args:
        query: The search query or uploaded document content
        document_content: The content of the matching document
        case_title: The title of the case
        
    Returns:
        A 2-3 line explanation of why the document matches the query
    """
    try:
        # Truncate content to avoid token limits (keep first 2000 chars)
        # truncated_doc = document_content[:10000] + "..." if len(document_content) > 10000 else document_content
        truncated_doc = document_content[:10000]
        prompt = f"""
You are a legal research assistant. Given a search query and a legal document, explain in 2-3 lines why this document is relevant to the query.

Query: "{query[:500]}"

Document Title: "{case_title}"

Document Content (excerpt): "{truncated_doc}"

Please provide a concise explanation (2-3 lines) of why this document matches the query. Focus on:
1. Key legal concepts that connect the query to the document
2. Specific legal issues or precedents mentioned
3. How the document addresses the query's legal concerns

Explanation:
"""

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-flash')
        print(f"Prompt length: {len(prompt)}")
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "Document matches based on semantic similarity analysis."
            
    except Exception as e:
        logger.error(f"Error generating match reason: {e}")
        return "Document matches based on semantic similarity analysis."

def load_document_content(document_id: str) -> str:
    """
    Load the full document content from the JSON file.
    
    Args:
        document_id: The document ID to load
        
    Returns:
        The document content as a string
    """
    try:
        # Look for the document in the data folder
        data_path = Path(__file__).parent / "data" / "ikanoon_data" / "article 21 right to privacy"
        
        # Search for the document in subdirectories
        for subdir in data_path.iterdir():
            if subdir.is_dir():
                json_file = subdir / f"{document_id}.json"
                if json_file.exists():
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        return data.get('doc', '')
        
        logger.warning(f"Document {document_id} not found in data folder")
        return ""
        
    except Exception as e:
        logger.error(f"Error loading document {document_id}: {e}")
        return ""

def initialize_services():
    """Initialize ChromaDB and search services once at startup."""
    global search_service, chroma_client, collection
    
    try:
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        
        # Initialize ChromaDB
        chroma_db_path = str(Path(__file__).parent / "chroma_db")
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        collection = chroma_client.get_collection(name="legal_documents")
        
        # Initialize search service
        search_service = DocumentSearchService(
            chroma_db_path=chroma_db_path,
            collection_name="legal_documents"
        )
        
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize services when the app starts up."""
    initialize_services()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Vakalat AI Chatbot is running", "status": "healthy"}

@app.post("/search/query")
async def search_by_query(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search for relevant legal cases using a text query.
    
    FastAPI automatically parses the query parameter from the request body.
    The function signature defines the expected input parameters.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return (default: 5)
        
    Returns:
        Dictionary with search results including case details and relevance scores
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Generate embedding for the query text
        # Using the same embedding model as the documents
        query_embedding = generate_embedding(query, "models/embedding-001")
        
        # Search ChromaDB for similar documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format the response with required fields and deduplicate by title keeping max relevance
        best_by_title = {}
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            similarity_score = 1 - distance  # Convert distance to similarity

            title = metadata.get('title', 'Unknown Title')
            preview_text = results['documents'][0][i]
            preview = preview_text[:200] + "..." if len(preview_text) > 200 else preview_text

            # Construct Indian Kanoon URL from document ID
            document_id = metadata.get('document_id', 'Unknown')
            pdf_url = f"https://indiankanoon.org/doc/{document_id}/" if document_id != 'Unknown' else None
            
            # Generate meaningful match reason using LLM
            document_content = load_document_content(document_id) if document_id != 'Unknown' else ""

            if document_content:
                match_reason = generate_match_reason(query, document_content, title)
            else:
                match_reason = f"Semantic similarity based on query: '{query[:100]}...'"
            
            current = {
                "case_title": title,
                "case_date": metadata.get('date', 'Unknown Date'),
                "relevance_score": round(similarity_score, 3),
                "reason_for_match": match_reason,
                "court": metadata.get('court', 'Unknown Court'),
                "document_id": document_id,
                "document_preview": preview,
                "pdf_url": pdf_url
            }

            if title not in best_by_title or current["relevance_score"] > best_by_title[title]["relevance_score"]:
                best_by_title[title] = current

        # Sort by relevance descending and trim to top_k
        search_results = sorted(best_by_title.values(), key=lambda r: r["relevance_score"], reverse=True)[:top_k]
        
        return {
            "query": query,
            "total_results": len(search_results),
            "results": search_results
        }
        
    except Exception as e:
        logger.error(f"Error in query search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/document")
async def search_by_document(
    file: UploadFile = File(...),
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Upload a document and find similar legal cases.
    
    FastAPI's UploadFile handles file uploads automatically.
    The File(...) parameter makes it a required file upload.
    
    Args:
        file: Uploaded document file (PDF, DOCX, or TXT)
        top_k: Number of results to return (default: 5)
        
    Returns:
        Dictionary with search results for similar cases
    """
    # Validate file type
    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt','.json']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create temporary file to store uploaded content
    temp_file_path = None
    try:
        # Read file content first
        content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
            # Extract text from the uploaded document
        document_text = extract_text_from_file(temp_file_path)
            
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
        
            # Use the existing document search service
        similar_docs = search_service.search_similar_documents(
            document_path=temp_file_path,
            top_k=top_k,
            use_parallel=True
        )
            
        # Format the response
        best_by_title = {}
        for doc in similar_docs:
            metadata = doc['metadata']
            similarity_score = doc['similarity']
            title = metadata.get('title', 'Unknown Title')

            # Construct Indian Kanoon URL from document ID
            document_id = metadata.get('document_id', 'Unknown')
            pdf_url = f"https://indiankanoon.org/doc/{document_id}/" if document_id != 'Unknown' else None
            
            # Generate meaningful match reason using LLM
            document_content = load_document_content(document_id) if document_id != 'Unknown' else ""
            case_title = metadata.get('title', 'Unknown Title')
            if document_content:
                # Use the uploaded document content as the "query" for LLM reasoning
                match_reason = generate_match_reason(document_text, document_content, case_title)
            else:
                match_reason = "Document similarity based on content analysis"
            
            result = {
                "case_title": case_title,
                "case_date": metadata.get('date', 'Unknown Date'),
                "relevance_score": round(similarity_score, 3),
                "reason_for_match": match_reason,
                "court": metadata.get('court', 'Unknown Court'),
                "document_id": document_id,
                "document_preview": doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document'],
                "pdf_url": pdf_url
            }
            if title not in best_by_title or result["relevance_score"] > best_by_title[title]["relevance_score"]:
                best_by_title[title] = result

        search_results = sorted(best_by_title.values(), key=lambda r: r["relevance_score"], reverse=True)[:top_k]
        
        
        return {
            "uploaded_file": file.filename,
            "total_results": len(search_results),
            "results": search_results
            }
            
    except Exception as e:
        logger.error(f"Error in document search: {e}")
        raise HTTPException(status_code=500, detail=f"Document search failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    try:
        # Check if services are properly initialized
        if not collection:
            return {"status": "unhealthy", "error": "ChromaDB collection not initialized"}
        
        # Test a simple query to verify everything works
        test_results = collection.query(
            query_embeddings=[[0.0] * 768],  # Dummy embedding
            n_results=1
        )
        
        return {
            "status": "healthy",
            "services": {
                "chromadb": "connected",
                "gemini_api": "configured",
                "collection": "accessible"
            },
            "total_documents": collection.count()
        }
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    # Run the FastAPI app using uvicorn
    # uvicorn is an ASGI server that can run FastAPI applications
    uvicorn.run(
        "app:app",  # app:app means the 'app' variable in this file
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,  # Default port
        reload=True  # Auto-reload on code changes (development only)
    )
