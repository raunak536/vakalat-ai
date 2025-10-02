import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
from docx import Document
import google.generativeai as genai
import numpy as np
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_documents(
    documents_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "gemini",
    api_key: Optional[str] = None,
    collection_name: str = "documents"
) -> Dict[str, Any]:
    """
    Process documents from a directory, chunk them, embed them, and store in ChromaDB.
    
    Args:
        documents_path: Path to directory containing PDF and Word documents
        chunk_size: Size of each text chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        embedding_model: Type of embedding model to use (currently only "gemini")
        api_key: API key for the embedding model
        collection_name: Name of the ChromaDB collection to store documents
    
    Returns:
        Dictionary with processing results and metadata
    """
    
    # Initialize ChromaDB
    logger.info("Initializing ChromaDB vector database...")
    import os
    os.makedirs("./chroma_db", exist_ok=True)  # ← Ensure directory exists
    client = chromadb.PersistentClient(path="./chroma_db")  # ← Use PersistentClient instead
    
    # Get or create collection
    try:
        collection = client.get_collection(collection_name)
        logger.info(f"Using existing collection: {collection_name}")
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Document chunks with embeddings"}
        )
        logger.info(f"Created new collection: {collection_name}")
    
    # Initialize embedding model
    logger.info("Configuring Gemini embedding model...")
    if embedding_model == "gemini":
        if not api_key:
            raise ValueError("API key is required for Gemini embedding model")
        genai.configure(api_key=api_key)
        # Use the embedding model directly
        embedding_model_name = "models/embedding-001"
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")
    
    # Extract text from documents
    logger.info(f"Scanning documents in: {documents_path}")
    documents = []
    for file_path in Path(documents_path).glob("*"):
        if file_path.suffix.lower() in ['.pdf', '.docx', '.doc']:
            text = extract_text_from_file(str(file_path))
            if text.strip():
                documents.append({
                    'file_path': str(file_path),
                    'text': text,
                    'file_type': file_path.suffix.lower()
                })
    
    if not documents:
        logger.warning("No valid documents found")
        return {"message": "No valid documents found", "processed_docs": 0}
    
    logger.info(f"Found {len(documents)} documents to process")
    
    # Chunk the documents
    logger.info("Chunking documents...")
    chunks = []
    for doc in documents:
        doc_chunks = chunk_text(doc['text'], chunk_size, chunk_overlap)
        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                'id': f"{doc['file_path']}_{i}",
                'text': chunk,
                'source_file': doc['file_path'],
                'chunk_index': i,
                'metadata': {
                    'file_type': doc['file_type'],
                    'chunk_size': len(chunk)
                }
            })
    
    logger.info(f"Created {len(chunks)} text chunks")
    
    # Generate embeddings and store in ChromaDB
    logger.info("Generating embeddings and storing in vector database...")
    texts = []
    ids = []
    metadatas = []
    embeddings = []  # ← Add this missing variable
    
    for i, chunk in enumerate(chunks):
        try:
            embedding = generate_embedding(chunk['text'], embedding_model_name)
            texts.append(chunk['text'])
            ids.append(chunk['id'])
            metadatas.append(chunk['metadata'])
            embeddings.append(embedding)  # ← Add this missing line
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error generating embedding for chunk {i}: {e}")
            continue
    
    # Add to ChromaDB collection
    if texts:
        collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings  # ← Now this will work
        )
        logger.info(f"Successfully stored {len(texts)} embeddings in vector database")
    
    logger.info("✅ Embeddings stored in vector database successfully!")
    
    return {
        "message": "Documents processed successfully",
        "processed_docs": len(documents),
        "total_chunks": len(chunks),
        "collection_name": collection_name,
        "chroma_db_path": "./chroma_db"
    }


def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF or Word document."""
    try:
        if file_path.lower().endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.docx', '.doc')):
            return extract_text_from_word(file_path)
        else:
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_word(file_path: str) -> str:
    """Extract text from Word document."""
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            search_start = max(start + chunk_size - 100, start)
            for i in range(end - 1, search_start, -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks


def generate_embedding(text: str, model_name: str) -> List[float]:
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


# Example usage
if __name__ == "__main__":
    # Example usage
    result = process_documents(
        documents_path="./documents",
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="gemini",
        api_key="AIzaSyCNWghUU8om5Na_jltJ3cQKZZ0W4Jmgc4A",
        collection_name="documents"
    )
