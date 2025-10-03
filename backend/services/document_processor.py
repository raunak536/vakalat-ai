import json
import os
import re
import logging
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
from docx import Document
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to store CSV mapping
_csv_mapping = None
_csv_cache_key = None

def load_csv_mapping(csv_path: str, court_filter: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Load CSV file and create a mapping of docid to metadata.
    
    Args:
        csv_path: Path to the CSV file
        court_filter: Optional court name to filter by (e.g., "Supreme Court of India")
        
    Returns:
        Dictionary mapping docid to metadata (title, date, court)
    """
    global _csv_mapping, _csv_cache_key
    
    # Create cache key that includes court filter
    cache_key = f"{csv_path}_{court_filter or 'all'}"
    if _csv_mapping is not None and _csv_cache_key == cache_key:
        return _csv_mapping
    
    mapping = {}
    filtered_count = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                docid = row['docid']
                court = row['court']
                
                # Apply court filter if specified
                if court_filter and court_filter.lower() not in court.lower():
                    filtered_count += 1
                    continue
                
                mapping[docid] = {
                    'title': row['title'],
                    'date': row['date'],
                    'court': court,
                    'position': row.get('position', '')
                }
        
        if court_filter:
            logger.info(f"Loaded {len(mapping)} document mappings from CSV (filtered by court: '{court_filter}', {filtered_count} documents filtered out)")
        else:
            logger.info(f"Loaded {len(mapping)} document mappings from CSV")
        
        # Store with cache key
        _csv_mapping = mapping
        _csv_cache_key = cache_key
        return mapping
    except Exception as e:
        logger.error(f"Error loading CSV mapping: {e}")
        return {}

def get_document_metadata(file_path: str, csv_mapping: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Get metadata for a document from CSV mapping.
    
    Args:
        file_path: Path to the JSON file
        csv_mapping: CSV mapping dictionary
        
    Returns:
        Dictionary with document metadata
    """
    # Extract document ID from filename (e.g., "232602.json" -> "232602")
    filename = Path(file_path).stem
    docid = filename
    
    if docid in csv_mapping:
        return csv_mapping[docid]
    else:
        logger.warning(f"No metadata found for document ID: {docid}")
        return {
            'title': 'Unknown Title',
            'date': 'Unknown Date',
            'court': 'Unknown Court',
            'position': 'Unknown Position'
        }


def process_documents(
    documents_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "gemini",
    api_key: Optional[str] = None,
    collection_name: str = "documents",
    csv_mapping_path: Optional[str] = None,
    court_filter: Optional[str] = None
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
        csv_mapping_path: Path to CSV file with document metadata mapping
        court_filter: Optional court name to filter by (e.g., "Supreme Court of India")
    
    Returns:
        Dictionary with processing results and metadata
    """
    
    # Load CSV mapping if provided
    csv_mapping = {}
    if csv_mapping_path and Path(csv_mapping_path).exists():
        if court_filter:
            logger.info(f"Loading CSV mapping from: {csv_mapping_path} (filtering by court: '{court_filter}')")
        else:
            logger.info(f"Loading CSV mapping from: {csv_mapping_path}")
        csv_mapping = load_csv_mapping(csv_mapping_path, court_filter)
    else:
        logger.warning("No CSV mapping provided or file not found. Using JSON metadata only.")
    
    # Initialize ChromaDB
    logger.info("Initializing ChromaDB vector database...")
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
        # Get API key from parameter or environment variable
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("API key is required for Gemini embedding model. Provide it as parameter or set GEMINI_API_KEY in .env file")
        genai.configure(api_key=api_key)
        # Use the embedding model directly
        embedding_model_name = "models/embedding-001"
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")
    
    # Extract text from documents
    logger.info(f"Scanning documents in: {documents_path}")
    documents = []
    
    # Handle both old structure (direct files) and new structure (nested directories)
    for file_path in Path(documents_path).rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.json']:
            # Get document ID for filtering
            doc_id = file_path.stem
            
            # If court filter is applied and CSV mapping exists, check if document should be processed
            if court_filter and csv_mapping and doc_id not in csv_mapping:
                logger.debug(f"Skipping document {file_path} - not found in filtered CSV mapping")
                continue
            
            text = extract_text_from_file(str(file_path))
            if text.strip():
                # Get additional metadata from CSV if available
                csv_metadata = get_document_metadata(str(file_path), csv_mapping)
                
                documents.append({
                    'file_path': str(file_path),
                    'text': text,
                    'file_type': file_path.suffix.lower(),
                    'csv_metadata': csv_metadata
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
            # Combine JSON metadata with CSV metadata
            chunk_metadata = {
                'file_type': doc['file_type'],
                'chunk_size': len(chunk),
                'chunk_index': i,
                'source_file': doc['file_path']
            }
            
            # Add CSV metadata if available
            if 'csv_metadata' in doc and doc['csv_metadata']:
                csv_meta = doc['csv_metadata']
                chunk_metadata.update({
                    'title': csv_meta.get('title', 'Unknown Title'),
                    'date': csv_meta.get('date', 'Unknown Date'),
                    'court': csv_meta.get('court', 'Unknown Court'),
                    'position': csv_meta.get('position', 'Unknown Position'),
                    'document_id': Path(doc['file_path']).stem
                })
            
            chunks.append({
                'id': f"{doc['file_path']}_{i}",
                'text': chunk,
                'source_file': doc['file_path'],
                'chunk_index': i,
                'metadata': chunk_metadata
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
    """Extract text from PDF, Word document, or JSON document."""
    try:
        if file_path.lower().endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.docx', '.doc')):
            return extract_text_from_word(file_path)
        elif file_path.lower().endswith('.json'):
            return extract_text_from_json(file_path)
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


def extract_text_from_json(file_path: str) -> str:
    """Extract text content from JSON document with HTML 'doc' key."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Extract the 'doc' field which contains HTML content
        if 'doc' in data:
            html_content = data['doc']
            # Extract content from specific HTML tags we care about
            # Remove HTML tags but preserve the text content
            clean_text = html_content
            
            # Replace specific HTML tags with meaningful text structure
            clean_text = re.sub(r'<h2[^>]*>', '\n## ', clean_text)  # Convert h2 to markdown-style
            clean_text = re.sub(r'<h3[^>]*>', '\n### ', clean_text)  # Convert h3 to markdown-style
            clean_text = re.sub(r'<h4[^>]*>', '\n#### ', clean_text)  # Convert h4 to markdown-style
            clean_text = re.sub(r'<h5[^>]*>', '\n##### ', clean_text)  # Convert h5 to markdown-style
            clean_text = re.sub(r'<h6[^>]*>', '\n###### ', clean_text)  # Convert h6 to markdown-style
            clean_text = re.sub(r'<p[^>]*>', '\n', clean_text)  # Convert p to newline
            clean_text = re.sub(r'<pre[^>]*>', '\n```\n', clean_text)  # Convert pre to code block
            clean_text = re.sub(r'</pre>', '\n```\n', clean_text)  # Close code block
            clean_text = re.sub(r'<br[^>]*>', '\n', clean_text)  # Convert br to newline
            clean_text = re.sub(r'<div[^>]*>', '\n', clean_text)  # Convert div to newline
            
            # Handle lists
            clean_text = re.sub(r'<ul[^>]*>', '\n', clean_text)  # Convert ul to newline
            clean_text = re.sub(r'<ol[^>]*>', '\n', clean_text)  # Convert ol to newline
            clean_text = re.sub(r'<li[^>]*>', '\n• ', clean_text)  # Convert li to bullet point
            clean_text = re.sub(r'</li>', '\n', clean_text)  # Close li
            
            # Handle emphasis (keep the text, remove tags)
            clean_text = re.sub(r'<(strong|b)[^>]*>', '**', clean_text)  # Bold start
            clean_text = re.sub(r'</(strong|b)>', '**', clean_text)  # Bold end
            clean_text = re.sub(r'<(em|i)[^>]*>', '*', clean_text)  # Italic start
            clean_text = re.sub(r'</(em|i)>', '*', clean_text)  # Italic end
            
            # Handle links (keep the text, remove the link)
            clean_text = re.sub(r'<a[^>]*>', '', clean_text)  # Remove opening a tag
            clean_text = re.sub(r'</a>', '', clean_text)  # Remove closing a tag
            
            # Handle tables
            clean_text = re.sub(r'<table[^>]*>', '\n', clean_text)  # Convert table to newline
            clean_text = re.sub(r'<tr[^>]*>', '\n', clean_text)  # Convert tr to newline
            clean_text = re.sub(r'<td[^>]*>', ' | ', clean_text)  # Convert td to separator
            clean_text = re.sub(r'<th[^>]*>', ' | ', clean_text)  # Convert th to separator
            clean_text = re.sub(r'</(table|tr|td|th)>', '', clean_text)  # Remove closing tags
            
            # Handle other common tags
            clean_text = re.sub(r'<span[^>]*>', '', clean_text)  # Remove span tags
            clean_text = re.sub(r'</span>', '', clean_text)  # Remove closing span
            clean_text = re.sub(r'<blockquote[^>]*>', '\n> ', clean_text)  # Convert blockquote
            clean_text = re.sub(r'</blockquote>', '\n', clean_text)  # Close blockquote
            
            # Remove all remaining HTML tags
            clean_text = re.sub(r'<[^>]+>', '', clean_text)
            
            # Clean up extra whitespace and newlines
            clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)  # Replace multiple newlines with double newline
            clean_text = re.sub(r'[ \t]+', ' ', clean_text)  # Replace multiple spaces with single space
            clean_text = clean_text.strip()
            
            # Add metadata from JSON for context
            title = data.get('title', '')
            publishdate = data.get('publishdate', '')
            tid = data.get('tid', '')
            
            # Combine metadata with document content
            full_text = f"Title: {title}\n"
            if publishdate:
                full_text += f"Date: {publishdate}\n"
            if tid:
                full_text += f"Document ID: {tid}\n"
            full_text += f"\n{clean_text}"
            
            return full_text
        else:
            logger.warning(f"No 'doc' key found in JSON file: {file_path}")
            return ""
            
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {file_path}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks, respecting structure for HTML-derived content."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at natural boundaries for structured content
        if end < len(text):
            # Look for paragraph breaks (double newlines) first
            search_start = max(start + chunk_size - 200, start)
            for i in range(end - 1, search_start, -1):
                if text[i:i+2] == '\n\n':
                    end = i + 2
                    break
            
            # If no paragraph break found, look for sentence endings
            if end == start + chunk_size:
                search_start = max(start + chunk_size - 100, start)
                for i in range(end - 1, search_start, -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            # If still no good break point, look for heading markers
            if end == start + chunk_size:
                search_start = max(start + chunk_size - 50, start)
                for i in range(end - 1, search_start, -1):
                    if text[i:i+3] in ['\n##', '\n###', '\n####']:
                        end = i
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
    # Example usage - API key will be loaded from .env file
    result = process_documents(
        documents_path="./backend/data/documents",
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="gemini",
        collection_name="legal_documents",
        csv_mapping_path="./backend/data/ikanoon_data/article 21 right to privacy/toc.csv",
        court_filter="Supreme Court of India"
    )
