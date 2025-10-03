#!/usr/bin/env python3
"""
Test script to demonstrate JSON document extraction and processing
"""

from document_processor import extract_text_from_json, chunk_text

def test_json_extraction():
    """Test the JSON extraction with a sample file"""
    
    # Test with one of your JSON files
    json_file = "backend/data/documents/1/232602.json"
    
    print("🔍 Testing JSON Document Extraction")
    print("=" * 50)
    
    # Extract text from JSON
    extracted_text = extract_text_from_json(json_file)
    
    if extracted_text:
        print(f"✅ Successfully extracted text")
        print(f"📏 Text length: {len(extracted_text)} characters")
        print(f"📄 First 800 characters:")
        print("-" * 30)
        print(extracted_text[:800])
        print("-" * 30)
        
        # Test chunking
        print(f"\n🔪 Testing chunking (chunk_size=1000, overlap=200)")
        chunks = chunk_text(extracted_text, chunk_size=1000, chunk_overlap=200)
        print(f"📦 Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    else:
        print("❌ Failed to extract text from JSON file")

if __name__ == "__main__":
    test_json_extraction()

