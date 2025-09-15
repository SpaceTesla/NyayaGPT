#!/usr/bin/env python3
"""Migration script to move from ChromaDB to Pinecone."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docling_core.types.doc import DoclingDocument
from utils import embed_document, chunk_document
from config.config import config
import pinecone
from pinecone import Pinecone, ServerlessSpec
import time


def setup_pinecone():
    """Set up Pinecone connection."""
    # You'll need to get these from Pinecone console
    api_key = os.getenv('PINECONE_API_KEY', '')
    environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    
    if not api_key:
        print("❌ Please set PINECONE_API_KEY in your .env file")
        print("Get your API key from: https://app.pinecone.io/")
        return None
    
    pc = Pinecone(api_key=api_key)
    return pc


def create_pinecone_index(pc, index_name="nyayagpt", dimension=1024):
    """Create Pinecone index."""
    try:
        # Check if index exists
        if index_name in pc.list_indexes().names():
            print(f"Index {index_name} already exists")
            return pc.Index(index_name)
        
        # Create new index
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=environment
            )
        )
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        
        print("✅ Index created successfully!")
        return pc.Index(index_name)
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return None


def upload_to_pinecone(index, embedded_chunks, document_name="indian_constitution"):
    """Upload chunks to Pinecone."""
    print(f"Uploading {len(embedded_chunks)} chunks to Pinecone...")
    
    # Prepare data for Pinecone
    vectors = []
    for i, chunk in enumerate(embedded_chunks):
        vector = {
            "id": f"{document_name}_{i}",
            "values": chunk.embedding.tolist(),
            "metadata": {
                "text": chunk.text,
                "document_name": document_name,
                "chunk_id": i,
                "text_length": len(chunk.text)
            }
        }
        vectors.append(vector)
    
    # Upload in batches
    batch_size = 100  # Pinecone handles larger batches well
    total_uploaded = 0
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        
        print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} vectors)")
        
        try:
            index.upsert(vectors=batch)
            total_uploaded += len(batch)
            print(f"✅ Uploaded {len(batch)} vectors")
        except Exception as e:
            print(f"❌ Error uploading batch {batch_num}: {e}")
            break
    
    print(f"🎉 Upload complete! {total_uploaded}/{len(vectors)} vectors uploaded")
    return total_uploaded


def test_pinecone_search(index, query="fundamental rights", top_k=3):
    """Test search functionality."""
    print(f"\n--- Testing Pinecone Search ---")
    print(f"Query: '{query}'")
    
    try:
        # Generate query embedding
        from utils.chunker import embedding_model
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        # Search
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"Found {len(results['matches'])} results:")
        for i, match in enumerate(results['matches']):
            print(f"\n{i+1}. Score: {match['score']:.4f}")
            print(f"   ID: {match['id']}")
            print(f"   Text: {match['metadata']['text'][:150]}...")
            
    except Exception as e:
        print(f"❌ Search error: {e}")


def main():
    """Main migration function."""
    print("🚀 Migrating to Pinecone...")
    
    # Setup Pinecone
    pc = setup_pinecone()
    if not pc:
        return
    
    # Create index
    index = create_pinecone_index(pc)
    if not index:
        return
    
    # Process document
    print("\n--- Processing Document ---")
    source = os.path.join("data", "indian_constitution.docling.json")
    docling_document = DoclingDocument.load_from_json(source)
    
    chunks = chunk_document(docling_document)
    print(f"Chunks: {len(chunks)}")
    
    embedded_chunks = embed_document(docling_document)
    print(f"Embedded chunks: {len(embedded_chunks)}")
    
    # Upload to Pinecone
    print("\n--- Uploading to Pinecone ---")
    uploaded = upload_to_pinecone(index, embedded_chunks)
    
    if uploaded > 0:
        # Test search
        test_pinecone_search(index)
        
        # Show index stats
        stats = index.describe_index_stats()
        print(f"\nIndex stats: {stats}")


if __name__ == "__main__":
    main()
