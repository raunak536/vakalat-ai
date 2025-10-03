#!/usr/bin/env python3
"""
Test script to demonstrate CSV metadata mapping functionality
"""

from document_processor import load_csv_mapping, get_document_metadata, process_documents
from pathlib import Path

def test_csv_mapping():
    """Test CSV mapping functionality"""
    
    print("🔍 Testing CSV Metadata Mapping")
    print("=" * 50)
    
    # Test CSV loading
    csv_path = "backend/data/ikanoon_data/article 21 right to privacy/toc.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"📄 Loading CSV from: {csv_path}")
    csv_mapping = load_csv_mapping(csv_path)
    
    print(f"✅ Loaded {len(csv_mapping)} document mappings")
    
    # Show sample mappings
    print(f"\n📋 Sample Mappings:")
    for i, (docid, metadata) in enumerate(list(csv_mapping.items())[:3]):
        print(f"  {i+1}. Document ID: {docid}")
        print(f"     Title: {metadata['title'][:60]}...")
        print(f"     Date: {metadata['date']}")
        print(f"     Court: {metadata['court']}")
        print()
    
    # Test court filtering
    print(f"🏛️ Testing Court Filtering:")
    print(f"   Total documents: {len(csv_mapping)}")
    
    # Filter for Supreme Court
    supreme_court_mapping = load_csv_mapping(csv_path, "Supreme Court of India")
    print(f"   Supreme Court documents: {len(supreme_court_mapping)}")
    
    # Show court distribution
    court_counts = {}
    for metadata in csv_mapping.values():
        court = metadata['court']
        court_counts[court] = court_counts.get(court, 0) + 1
    
    print(f"\n📊 Court Distribution:")
    for court, count in sorted(court_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {court}: {count} documents")
    
    # Test document metadata lookup
    print(f"🔍 Testing Document Metadata Lookup:")
    test_files = [
        "backend/data/documents/1/232602.json",
        "backend/data/documents/2/107953018.json"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            metadata = get_document_metadata(file_path, csv_mapping)
            print(f"\n📄 File: {file_path}")
            print(f"   Title: {metadata['title']}")
            print(f"   Date: {metadata['date']}")
            print(f"   Court: {metadata['court']}")
            print(f"   Position: {metadata['position']}")
        else:
            print(f"❌ File not found: {file_path}")

def test_enhanced_processing():
    """Test the enhanced document processing with metadata"""
    
    print(f"\n🚀 Testing Enhanced Document Processing")
    print("=" * 50)
    
    # Test with Supreme Court filter
    print("🏛️ Processing ONLY Supreme Court documents...")
    result = process_documents(
        documents_path="./backend/data/documents",
        chunk_size=500,  # Smaller chunks for testing
        chunk_overlap=100,
        embedding_model="gemini",
        collection_name="test_supreme_court",
        csv_mapping_path="./backend/data/ikanoon_data/article 21 right to privacy/toc.csv",
        court_filter="Supreme Court of India"
    )
    
    print(f"✅ Processing completed!")
    print(f"   Processed documents: {result['processed_docs']}")
    print(f"   Total chunks: {result['total_chunks']}")
    print(f"   Collection: {result['collection_name']}")
    
    # Test without filter for comparison
    print(f"\n🔄 Processing ALL documents (no filter)...")
    result_all = process_documents(
        documents_path="./backend/data/documents",
        chunk_size=500,
        chunk_overlap=100,
        embedding_model="gemini",
        collection_name="test_all_courts",
        csv_mapping_path="./backend/data/ikanoon_data/article 21 right to privacy/toc.csv"
    )
    
    print(f"✅ All documents processing completed!")
    print(f"   Processed documents: {result_all['processed_docs']}")
    print(f"   Total chunks: {result_all['total_chunks']}")
    print(f"   Collection: {result_all['collection_name']}")
    
    print(f"\n📊 Filtering Results:")
    print(f"   Supreme Court only: {result['processed_docs']} documents")
    print(f"   All courts: {result_all['processed_docs']} documents")
    print(f"   Filtered out: {result_all['processed_docs'] - result['processed_docs']} documents")

def test_metadata_in_chunks():
    """Test that metadata is properly stored in chunks"""
    
    print(f"\n🔍 Testing Metadata in Vector Database")
    print("=" * 50)
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Test Supreme Court collection
        print("🏛️ Supreme Court Collection Metadata:")
        try:
            collection = client.get_collection("test_supreme_court")
            sample = collection.get(limit=3, include=['metadatas'])
            
            print(f"📦 Sample Chunk Metadata:")
            for i, metadata in enumerate(sample['metadatas']):
                print(f"\n  Chunk {i+1}:")
                print(f"    Title: {metadata.get('title', 'N/A')}")
                print(f"    Date: {metadata.get('date', 'N/A')}")
                print(f"    Court: {metadata.get('court', 'N/A')}")
                print(f"    Document ID: {metadata.get('document_id', 'N/A')}")
                print(f"    Chunk Size: {metadata.get('chunk_size', 'N/A')}")
                print(f"    Source File: {metadata.get('source_file', 'N/A')}")
        except Exception as e:
            print(f"❌ Error accessing Supreme Court collection: {e}")
        
        # Test All Courts collection
        print(f"\n🌐 All Courts Collection Metadata:")
        try:
            collection = client.get_collection("test_all_courts")
            sample = collection.get(limit=3, include=['metadatas'])
            
            print(f"📦 Sample Chunk Metadata:")
            for i, metadata in enumerate(sample['metadatas']):
                print(f"\n  Chunk {i+1}:")
                print(f"    Title: {metadata.get('title', 'N/A')}")
                print(f"    Date: {metadata.get('date', 'N/A')}")
                print(f"    Court: {metadata.get('court', 'N/A')}")
                print(f"    Document ID: {metadata.get('document_id', 'N/A')}")
                print(f"    Chunk Size: {metadata.get('chunk_size', 'N/A')}")
                print(f"    Source File: {metadata.get('source_file', 'N/A')}")
        except Exception as e:
            print(f"❌ Error accessing All Courts collection: {e}")
            
    except Exception as e:
        print(f"❌ Error testing metadata: {e}")

if __name__ == "__main__":
    test_csv_mapping()
    test_enhanced_processing()
    test_metadata_in_chunks()
    
    print(f"\n🎯 SUMMARY:")
    print("The enhanced document processor now includes:")
    print("✅ CSV metadata mapping (title, date, court, position)")
    print("✅ Document ID extraction from filename")
    print("✅ Court filtering (Supreme Court only)")
    print("✅ Rich metadata in each chunk")
    print("✅ Better search and retrieval capabilities")
    print("✅ Efficient filtering to reduce processing time and costs")

