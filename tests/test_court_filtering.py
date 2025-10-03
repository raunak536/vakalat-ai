#!/usr/bin/env python3
"""
Quick test to verify court filtering functionality
"""

from document_processor import load_csv_mapping, process_documents
from pathlib import Path

def test_court_filtering():
    """Test court filtering functionality"""
    
    print("🏛️ Testing Court Filtering")
    print("=" * 50)
    
    csv_path = "backend/data/ikanoon_data/article 21 right to privacy/toc.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Test different court filters
    print("📊 Testing different court filters:")
    
    # All documents
    all_docs = load_csv_mapping(csv_path)
    print(f"   All documents: {len(all_docs)}")
    
    # Supreme Court only
    supreme_docs = load_csv_mapping(csv_path, "Supreme Court of India")
    print(f"   Supreme Court: {len(supreme_docs)}")
    
    # High Court documents
    high_court_docs = load_csv_mapping(csv_path, "High Court")
    print(f"   High Courts: {len(high_court_docs)}")
    
    # Show court distribution
    court_counts = {}
    for metadata in all_docs.values():
        court = metadata['court']
        court_counts[court] = court_counts.get(court, 0) + 1
    
    print(f"\n📋 Court Distribution:")
    for court, count in sorted(court_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {court}: {count} documents")
    
    # Test processing with filter
    print(f"\n🚀 Testing Document Processing with Supreme Court Filter:")
    
    result = process_documents(
        documents_path="./backend/data/documents",
        chunk_size=500,
        chunk_overlap=100,
        embedding_model="gemini",
        collection_name="test_supreme_filter",
        csv_mapping_path=csv_path,
        court_filter="Supreme Court of India"
    )
    
    print(f"✅ Processing completed!")
    print(f"   Processed documents: {result['processed_docs']}")
    print(f"   Total chunks: {result['total_chunks']}")
    print(f"   Collection: {result['collection_name']}")
    
    # Verify all processed documents are from Supreme Court
    if result['processed_docs'] > 0:
        print(f"\n🔍 Verifying Court Filtering:")
        try:
            import chromadb
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection("test_supreme_filter")
            
            # Get all metadata
            all_data = collection.get(include=['metadatas'])
            courts = set()
            for metadata in all_data['metadatas']:
                courts.add(metadata.get('court', 'Unknown'))
            
            print(f"   Courts found in processed documents: {courts}")
            
            if len(courts) == 1 and 'Supreme Court of India' in courts:
                print("   ✅ All documents are from Supreme Court!")
            else:
                print("   ❌ Some documents are not from Supreme Court")
                
        except Exception as e:
            print(f"   ❌ Error verifying: {e}")
    else:
        print("   ⚠️ No documents were processed")

if __name__ == "__main__":
    test_court_filtering()
