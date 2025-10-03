#!/usr/bin/env python3
"""
Test script to compare HTML input vs extracted text output
Shows the efficacy of HTML text extraction
"""

import json
import re
from pathlib import Path

def extract_text_from_json_test(file_path: str) -> tuple[str, str]:
    """
    Extract text from JSON document with HTML 'doc' key.
    Returns both the original HTML and cleaned text for comparison.
    """
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
            
            return html_content, full_text
        else:
            return "", "No 'doc' key found in JSON file"
            
    except json.JSONDecodeError as e:
        return "", f"Error parsing JSON file: {e}"
    except Exception as e:
        return "", f"Error reading JSON file: {e}"

def test_html_extraction():
    """Test HTML extraction and show comparison"""
    
    print("🔍 HTML Extraction Test - Input vs Output Comparison")
    print("=" * 80)
    
    # Test files
    test_files = [
        "backend/data/documents/1/232602.json",
        "backend/data/documents/2/107953018.json"
    ]
    
    for i, json_file in enumerate(test_files, 1):
        print(f"\n📄 TEST {i}: {json_file}")
        print("=" * 60)
        
        if not Path(json_file).exists():
            print(f"❌ File not found: {json_file}")
            continue
            
        # Extract both HTML and cleaned text
        html_content, cleaned_text = extract_text_from_json_test(json_file)
        
        if not html_content:
            print(f"❌ Failed to extract content from {json_file}")
            continue
            
        # Show statistics
        print(f"📊 STATISTICS:")
        print(f"   HTML Length: {len(html_content):,} characters")
        print(f"   Clean Length: {len(cleaned_text):,} characters")
        print(f"   Reduction: {((len(html_content) - len(cleaned_text)) / len(html_content) * 100):.1f}%")
        
        # Show HTML input (first 500 chars)
        print(f"\n🔤 ORIGINAL HTML (first 500 chars):")
        print("-" * 40)
        print(html_content[:500])
        if len(html_content) > 500:
            print("... [truncated]")
        print("-" * 40)
        
        # Show cleaned output (first 500 chars)
        print(f"\n✨ EXTRACTED TEXT (first 500 chars):")
        print("-" * 40)
        print(cleaned_text[:500])
        if len(cleaned_text) > 500:
            print("... [truncated]")
        print("-" * 40)
        
        # Show HTML tags found
        html_tags = re.findall(r'<([^>]+)>', html_content)
        unique_tags = list(set(html_tags))
        print(f"\n🏷️  HTML TAGS FOUND: {', '.join(unique_tags[:10])}")
        if len(unique_tags) > 10:
            print(f"   ... and {len(unique_tags) - 10} more")
        
        print("\n" + "="*60)

def test_specific_html_patterns():
    """Test specific HTML patterns to show extraction quality"""
    
    print("\n🧪 SPECIFIC HTML PATTERN TESTS")
    print("=" * 50)
    
    # Test HTML patterns
    test_htmls = [
        '<h2 class="doc_title">Case Title</h2>',
        '<h3 class="doc_citations">Citations: 2000(1)ALD(CRI)117</h3>',
        '<p data-structure="PetArg">The petitioner is a <strong>truck driver</strong>.</p>',
        '<pre id="pre_1">ORDER\n\nV.V.S. Rao, J.</pre>',
        '<a href="/search/?formInput=authorid:r-more">Ranjit More</a>',
        '<ul><li>First point</li><li>Second point</li></ul>',
        '<table><tr><th>Header</th></tr><tr><td>Data</td></tr></table>'
    ]
    
    for i, html in enumerate(test_htmls, 1):
        print(f"\nTest {i}: {html}")
        print("-" * 30)
        
        # Apply the same cleaning logic
        clean_text = html
        clean_text = re.sub(r'<h2[^>]*>', '\n## ', clean_text)
        clean_text = re.sub(r'<h3[^>]*>', '\n### ', clean_text)
        clean_text = re.sub(r'<p[^>]*>', '\n', clean_text)
        clean_text = re.sub(r'<pre[^>]*>', '\n```\n', clean_text)
        clean_text = re.sub(r'</pre>', '\n```\n', clean_text)
        clean_text = re.sub(r'<(strong|b)[^>]*>', '**', clean_text)
        clean_text = re.sub(r'</(strong|b)>', '**', clean_text)
        clean_text = re.sub(r'<a[^>]*>', '', clean_text)
        clean_text = re.sub(r'</a>', '', clean_text)
        clean_text = re.sub(r'<ul[^>]*>', '\n', clean_text)
        clean_text = re.sub(r'<li[^>]*>', '\n• ', clean_text)
        clean_text = re.sub(r'</li>', '\n', clean_text)
        clean_text = re.sub(r'<table[^>]*>', '\n', clean_text)
        clean_text = re.sub(r'<tr[^>]*>', '\n', clean_text)
        clean_text = re.sub(r'<td[^>]*>', ' | ', clean_text)
        clean_text = re.sub(r'<th[^>]*>', ' | ', clean_text)
        clean_text = re.sub(r'</(table|tr|td|th)>', '', clean_text)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)
        clean_text = clean_text.strip()
        
        print(f"Result: {clean_text}")

if __name__ == "__main__":
    test_html_extraction()
    test_specific_html_patterns()
    
    print("\n🎯 SUMMARY:")
    print("This test shows how HTML content is converted to clean, searchable text.")
    print("Compare the 'ORIGINAL HTML' vs 'EXTRACTED TEXT' sections above.")
    print("The extraction should preserve all important content while removing HTML syntax.")

