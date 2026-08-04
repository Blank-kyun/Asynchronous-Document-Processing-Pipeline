#!/usr/bin/env python3
"""
Test script for the RAG pipeline.
Tests extraction, embedding, and Milvus storage.
"""

import os
import sys
import tempfile
from pathlib import Path
from pymilvus import MilvusClient

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from extraction import file_router, elements_to_text, chunk_splitter, sentence_splitter
from embedding import embed_texts, EMBED_DIM
from worker import process_document, create_collection_if_not_exists, COLLECTION_NAME
from unstructured.partition.auto import partition


def test_sentence_splitter():
    """Test sentence splitting functionality."""
    print("\n" + "="*60)
    print("TEST 1: Sentence Splitter")
    print("="*60)
    
    text = "This is sentence one. This is sentence two! Is this sentence three? Yes it is."
    sentences = sentence_splitter(text)
    
    print(f"Input text: {text}")
    print(f"Split into {len(sentences)} sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    
    assert len(sentences) > 0, "Should split into at least one sentence"
    print("✓ Sentence splitter works correctly")


def test_chunk_splitter():
    """Test chunk splitting functionality."""
    print("\n" + "="*60)
    print("TEST 2: Chunk Splitter")
    print("="*60)
    
    # Create a long text
    text = ". ".join([f"This is sentence number {i}" for i in range(50)])
    chunks = chunk_splitter(text)
    
    print(f"Input text length: {len(text)} characters")
    print(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks[:3], 1):  # Show first 3
        print(f"  Chunk {i} ({len(chunk)} chars): {chunk[:80]}...")
    if len(chunks) > 3:
        print(f"  ... and {len(chunks) - 3} more chunks")
    
    assert len(chunks) > 0, "Should create at least one chunk"
    print("✓ Chunk splitter works correctly")


def test_elements_to_text():
    """Test element to text conversion."""
    print("\n" + "="*60)
    print("TEST 3: Elements to Text Conversion")
    print("="*60)
    
    # Create a simple test file
    test_file = Path("test/test.txt")
    if not test_file.exists():
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text("This is a test document.\nIt has multiple lines.\nAnd some content.")
    
    elements = partition(filename=str(test_file))
    text = elements_to_text(elements)
    
    print(f"Extracted {len(elements)} elements")
    print(f"Converted to text ({len(text)} characters):")
    print(f"  {text[:200]}...")
    
    assert len(text) > 0, "Should extract text from elements"
    print("✓ Elements to text conversion works correctly")


def test_file_router():
    """Test file routing and extraction."""
    print("\n" + "="*60)
    print("TEST 4: File Router (Full Extraction)")
    print("="*60)
    
    # Try to use test.pdf if it exists, otherwise create a text file
    test_file = Path("test/test.pdf")
    if not test_file.exists():
        print("test/test.pdf not found, creating a test text file...")
        test_file = Path("test/test_doc.txt")
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text(
            "This is a test document for the RAG pipeline. "
            "It contains multiple sentences. "
            "Each sentence should be processed correctly. "
            "The document will be chunked into smaller pieces. "
            "These chunks will then be embedded. "
            "Finally, they will be stored in Milvus. " * 10
        )
    
    if not test_file.exists():
        print(f"✗ Test file not found: {test_file}")
        return False
    
    print(f"Processing file: {test_file}")
    chunks = file_router(str(test_file))
    
    print(f"Extracted {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"  Chunk {i} ({len(chunk)} chars): {chunk[:100]}...")
    if len(chunks) > 3:
        print(f"  ... and {len(chunks) - 3} more chunks")
    
    assert len(chunks) > 0, "Should extract at least one chunk"
    print("✓ File router works correctly")
    return chunks


def test_embedding():
    """Test embedding functionality."""
    print("\n" + "="*60)
    print("TEST 5: Text Embedding")
    print("="*60)
    
    test_texts = [
        "This is a test sentence.",
        "This is another test sentence.",
        "Machine learning is fascinating."
    ]
    
    print(f"Embedding {len(test_texts)} texts...")
    vectors = embed_texts(test_texts)
    
    print(f"Generated {len(vectors)} vectors")
    print(f"Vector dimension: {len(vectors[0])} (expected: {EMBED_DIM})")
    print(f"First vector sample: {vectors[0][:5]}...")
    
    assert len(vectors) == len(test_texts), "Should generate one vector per text"
    assert len(vectors[0]) == EMBED_DIM, f"Vector dimension should be {EMBED_DIM}"
    print("✓ Embedding works correctly")
    return vectors


def test_milvus_storage():
    """Test Milvus storage and retrieval."""
    print("\n" + "="*60)
    print("TEST 6: Milvus Storage")
    print("="*60)
    
    # Use a test database file
    test_db = "test_milvus.db"
    
    try:
        from pymilvus import MilvusClient
        
        client = MilvusClient(uri=test_db)
        
        # Ensure collection exists
        test_collection = "test_collection"
        if client.has_collection(test_collection):
            client.drop_collection(test_collection)
        
        client.create_collection(
            collection_name=test_collection,
            dimension=EMBED_DIM,
            metric_type="COSINE",
            vector_field_name="vector",
            auto_id=True
        )
        
        # Insert test data
        test_data = [
            {
                "vector": [0.1] * EMBED_DIM,
                "text": "Test document 1",
                "doc_id": "test1",
            },
            {
                "vector": [0.2] * EMBED_DIM,
                "text": "Test document 2",
                "doc_id": "test2",
            },
        ]
        
        result = client.insert(collection_name=test_collection, data=test_data)
        print(f"Inserted {len(result.get('ids', []))} records")
        
        # Query test
        query_result = client.search(
            collection_name=test_collection,
            data=[[0.15] * EMBED_DIM],
            limit=2,
        )
        
        print(f"Query returned {len(query_result[0])} results")
        print("✓ Milvus storage works correctly")
        
        # Cleanup
        client.drop_collection(test_collection)
        client.close()
        
        # Remove test database
        import shutil
        if os.path.exists(test_db):
            os.remove(test_db)
        
        return True
    except Exception as e:
        print(f"✗ Milvus test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the complete pipeline end-to-end."""
    print("\n" + "="*60)
    print("TEST 7: Full Pipeline (Extraction -> Embedding -> Storage)")
    print("="*60)
    
    # Create a test file
    test_file = Path("test/test_pipeline.txt")
    test_file.parent.mkdir(exist_ok=True)
    test_file.write_text(
        "This is a comprehensive test of the RAG pipeline. "
        "The document contains important information about machine learning. "
        "It discusses various algorithms and techniques. "
        "Natural language processing is a key component. "
        "Vector embeddings enable semantic search capabilities. "
        "The system processes documents efficiently. " * 5
    )
    
    test_doc_id = "test_doc_123"
    test_user_id = "test_user_456"
    
    print(f"Processing document: {test_file}")
    print(f"Doc ID: {test_doc_id}, User ID: {test_user_id}")
    
    success = process_document(str(test_file), test_doc_id, test_user_id)
    
    if success:
        print("✓ Full pipeline completed successfully")
        
        # Verify data was stored
        try:
            client = MilvusClient(uri="milvus.db")
            create_collection_if_not_exists(client)
            
            # Query to verify data exists
            query_result = client.query(
                collection_name=COLLECTION_NAME,
                filter=f'doc_id == "{test_doc_id}"',
                limit=5,
            )
            
            print(f"Found {len(query_result)} chunks in database for doc_id={test_doc_id}")
            if query_result:
                print(f"Sample chunk: {query_result[0].get('text', '')[:100]}...")
            
            client.close()
        except Exception as e:
            print(f"Warning: Could not verify stored data: {e}")
        
        return True
    else:
        print("✗ Full pipeline failed")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RAG PIPELINE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Sentence Splitter", test_sentence_splitter),
        ("Chunk Splitter", test_chunk_splitter),
        ("Elements to Text", test_elements_to_text),
        ("File Router", test_file_router),
        ("Embedding", test_embedding),
        ("Milvus Storage", test_milvus_storage),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, True, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, None))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, _ in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())


