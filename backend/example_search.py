#!/usr/bin/env python3
"""
Simple example demonstrating document-to-document search.
Run this script to test the document search functionality.
"""

import os
import sys
from pathlib import Path

# Add services directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services'))

from document_search import search_documents


def main():
    """Run a simple document search example."""
    
    # Example document path - you can change this to any document in your data
    document_path = "./data/documents/1/232602.json"
    
    print("🔍 Document-to-Document Search Example")
    print("=" * 50)
    
    # Check if document exists
    if not os.path.exists(document_path):
        print(f"❌ Document not found: {document_path}")
        print("\nAvailable documents:")
        data_dir = Path("./data/documents")
        if data_dir.exists():
            for doc_dir in data_dir.iterdir():
                if doc_dir.is_dir():
                    for file in doc_dir.glob("*.json"):
                        print(f"  - {file}")
        return
    
    print(f"📄 Searching for documents similar to: {document_path}")
    print()
    
    try:
        # Search for similar documents
        results = search_documents(document_path, top_k=3)
        
        if not results:
            print("❌ No similar documents found or error occurred")
            return
        
        print(f"✅ Found {len(results)} similar documents:")
        print()
        
        for i, doc in enumerate(results, 1):
            print(f"{i}. Document ID: {doc['id']}")
            print(f"   Similarity Score: {doc['similarity']:.3f}")
            print(f"   Distance: {doc['distance']:.3f}")
            
            # Show metadata if available
            if doc['metadata']:
                print(f"   Metadata: {doc['metadata']}")
            
            # Show document preview
            doc_preview = doc['document'][:150] + "..." if len(doc['document']) > 150 else doc['document']
            print(f"   Preview: {doc_preview}")
            print()
            
    except Exception as e:
        print(f"❌ Error during search: {e}")
        print("\nMake sure:")
        print("1. ChromaDB is set up with documents")
        print("2. The collection name matches your data")
        print("3. You have the required dependencies installed")


if __name__ == "__main__":
    main()
