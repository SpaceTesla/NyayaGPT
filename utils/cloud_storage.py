"""Cloud storage configuration for ChromaDB."""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
from .embedder import ChunkWithEmbedding
from .chunker import embedding_model

# Load environment variables
load_dotenv()


class CloudChromaStorage:
    """ChromaDB cloud storage for document chunks and embeddings."""
    
    def __init__(self, collection_name: str = "documents", api_key: str = None, tenant: str = None, database: str = None):
        """Initialize ChromaDB cloud client and collection."""
        # Get credentials from environment or parameters
        api_key = api_key or os.getenv('CHROMA_API_KEY')
        tenant = tenant or os.getenv('CHROMA_TENANT')
        database = database or os.getenv('CHROMA_DATABASE')
        
        if not all([api_key, tenant, database]):
            raise ValueError("Missing ChromaDB Cloud credentials. Please set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE in your .env file")
        
        self.client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def save_chunks(self, chunks: List[ChunkWithEmbedding], document_name: str = "document") -> None:
        """Save embedded chunks to ChromaDB cloud."""
        if not chunks:
            print("No chunks to save!")
            return
        
        # Prepare data for ChromaDB
        ids = [f"{document_name}_{i}" for i in range(len(chunks))]
        texts = [chunk.text for chunk in chunks]
        embeddings = [chunk.embedding.tolist() for chunk in chunks]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            metadata = {
                "document_name": document_name,
                "chunk_id": i,
                "text_length": len(chunk.text),
                "created_at": datetime.now().isoformat(),
                "embedding_model": "bge-large-en-v1.5",
                "embedding_dim": 1024
            }
            if chunk.metadata:
                for key, value in chunk.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metadata[key] = value
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"Saved {len(chunks)} chunks to ChromaDB cloud collection '{self.collection.name}'")
    
    def search(self, query: str, n_results: int = 5, document_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using text query."""
        where_filter = {"document_name": document_name} if document_name else None
        
        # Generate embedding for the query using the same model
        query_embedding = embedding_model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            "name": self.collection.name,
            "count": count,
            "metadata": self.collection.metadata
        }
    
    def explore_database(self, limit: int = 10) -> None:
        """Display database contents in a readable format."""
        # Get all data from collection
        results = self.collection.get(limit=limit)
        
        print(f"\n=== Database Explorer - Collection: {self.collection.name} ===")
        print(f"Total chunks: {self.collection.count()}")
        print(f"Showing first {min(limit, len(results['ids']))} chunks:\n")
        
        for i, (chunk_id, text, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
            print(f"--- Chunk {i+1} ---")
            print(f"ID: {chunk_id}")
            print(f"Document: {metadata.get('document_name', 'Unknown')}")
            print(f"Text Length: {metadata.get('text_length', 'Unknown')}")
            print(f"Text Preview: {text[:200]}{'...' if len(text) > 200 else ''}")
            print()
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all data from the collection as a dictionary."""
        return self.collection.get()
