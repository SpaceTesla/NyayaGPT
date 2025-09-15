#!/usr/bin/env python3
"""Batch upload script to handle large documents within free tier limits."""

from docling_core.types.doc import DoclingDocument
from utils import embed_document, chunk_document
from utils.cloud_storage import CloudChromaStorage
from config.config import config
import os


def batch_upload_constitution(batch_size=250):
    """Upload Indian Constitution in batches to stay within free tier limits."""
    
    # Load document
    source = os.path.join("data", "indian_constitution.docling.json")
    docling_document = DoclingDocument.load_from_json(source)
    
    print("Processing Indian Constitution...")
    
    # Chunk and embed
    chunks = chunk_document(docling_document)
    print(f"Total chunks: {len(chunks)}")
    
    embedded_chunks = embed_document(docling_document)
    print(f"Total embedded chunks: {len(embedded_chunks)}")
    
    # Connect to cloud storage
    storage = CloudChromaStorage(
        collection_name=config.storage.collection_name
    )
    
    # Process in batches
    total_chunks = len(embedded_chunks)
    uploaded = 0
    
    for i in range(0, total_chunks, batch_size):
        batch = embedded_chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f"\n--- Uploading Batch {batch_num}/{total_batches} ---")
        print(f"Chunks {i+1}-{min(i+batch_size, total_chunks)} of {total_chunks}")
        
        try:
            storage.save_chunks(batch, document_name="indian_constitution")
            uploaded += len(batch)
            print(f"✅ Successfully uploaded {len(batch)} chunks")
            print(f"Total uploaded: {uploaded}/{total_chunks}")
            
        except Exception as e:
            print(f"❌ Error uploading batch {batch_num}: {e}")
            break
    
    print(f"\n🎉 Upload complete! {uploaded}/{total_chunks} chunks uploaded")
    
    # Test search
    print("\n--- Testing Search ---")
    search_results = storage.search("fundamental rights", n_results=3)
    print(f"Found {len(search_results)} results for 'fundamental rights':")
    for i, result in enumerate(search_results):
        print(f"\n{i+1}. Distance: {result['distance']:.4f}")
        print(f"   Text: {result['text'][:150]}...")


if __name__ == "__main__":
    batch_upload_constitution(batch_size=250)
