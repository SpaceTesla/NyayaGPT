#!/usr/bin/env python3
"""Clear existing data and upload fresh constitution data."""

from docling_core.types.doc import DoclingDocument
from utils import embed_document, chunk_document
from utils.cloud_storage import CloudChromaStorage
from config.config import config
import os


def clear_and_upload():
    """Clear existing data and upload constitution."""
    
    # Connect to cloud storage
    storage = CloudChromaStorage(
        collection_name=config.storage.collection_name
    )
    
    # Clear existing collection
    print("Clearing existing collection...")
    try:
        # Get all data first
        all_data = storage.get_all_data()
        if all_data['ids']:
            print(f"Found {len(all_data['ids'])} existing chunks")
            # Delete collection and recreate
            storage.client.delete_collection(storage.collection.name)
            print("✅ Collection cleared")
        else:
            print("Collection is already empty")
    except Exception as e:
        print(f"Note: {e}")
    
    # Load and process document
    source = os.path.join("data", "indian_constitution.docling.json")
    docling_document = DoclingDocument.load_from_json(source)
    
    print("\nProcessing Indian Constitution...")
    chunks = chunk_document(docling_document)
    print(f"Total chunks: {len(chunks)}")
    
    embedded_chunks = embed_document(docling_document)
    print(f"Total embedded chunks: {len(embedded_chunks)}")
    
    # Upload in smaller batches
    batch_size = 250
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
            if "quota" in str(e).lower():
                print("💡 Consider upgrading to paid tier or reducing batch size")
            break
    
    print(f"\n🎉 Upload complete! {uploaded}/{total_chunks} chunks uploaded")


if __name__ == "__main__":
    clear_and_upload()
