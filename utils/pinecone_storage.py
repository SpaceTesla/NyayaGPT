"""Pinecone storage for document chunks and embeddings."""

from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
from .embedder import ChunkWithEmbedding
from .chunker import embedding_model

# Load environment variables
load_dotenv()


class PineconeStorage:
    """Pinecone storage for document chunks and embeddings."""
    
    def __init__(self, index_name: str = "nyayagpt", dimension: int = 1024):
        """Initialize Pinecone client and index."""
        # Get credentials from environment
        api_key = os.getenv('PINECONE_API_KEY')
        environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        
        if not api_key:
            raise ValueError("Missing PINECONE_API_KEY. Please set it in your .env file")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.environment = environment
        
        # Get or create index
        self.index = self._get_or_create_index()
    
    def _get_or_create_index(self):
        """Get existing index or create new one."""
        try:
            # Check if index exists
            if self.index_name in self.pc.list_indexes().names():
                print(f"Using existing Pinecone index: {self.index_name}")
                return self.pc.Index(self.index_name)
            
            # Create new index
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.environment
                )
            )
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            import time
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            
            print("✅ Index created successfully!")
            return self.pc.Index(self.index_name)
            
        except Exception as e:
            print(f"❌ Error with Pinecone index: {e}")
            raise
    
    def save_chunks(self, chunks: List[ChunkWithEmbedding], document_name: str = "document") -> None:
        """Save embedded chunks to Pinecone."""
        if not chunks:
            print("No chunks to save!")
            return
        
        print(f"Preparing {len(chunks)} chunks for Pinecone upload...")
        
        # Prepare data for Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            vector = {
                "id": f"{document_name}_{i}",
                "values": chunk.embedding.tolist(),
                "metadata": {
                    "text": chunk.text,
                    "document_name": document_name,
                    "chunk_id": i,
                    "text_length": len(chunk.text),
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": "bge-large-en-v1.5",
                    "embedding_dim": self.dimension
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
                self.index.upsert(vectors=batch)
                total_uploaded += len(batch)
                print(f"✅ Uploaded {len(batch)} vectors")
            except Exception as e:
                print(f"❌ Error uploading batch {batch_num}: {e}")
                break
        
        print(f"🎉 Upload complete! {total_uploaded}/{len(vectors)} vectors uploaded to Pinecone")
    
    def search(self, query: str, n_results: int = 5, document_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using text query."""
        # Generate embedding for the query
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        # Prepare filter
        filter_dict = {"document_name": document_name} if document_name else None
        
        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'id': match['id'],
                'text': match['metadata']['text'],
                'distance': 1 - match['score'],  # Convert similarity to distance
                'metadata': match['metadata']
            })
        
        return formatted_results
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the index."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "name": self.index_name,
                "dimension": self.dimension,
                "total_vector_count": stats.total_vector_count,
                "namespaces": list(stats.namespaces.keys()) if stats.namespaces else []
            }
        except Exception as e:
            return {"error": str(e)}
    
    def explore_database(self, limit: int = 10) -> None:
        """Display database contents in a readable format."""
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            total_vectors = stats.total_vector_count
            
            print(f"\n=== Pinecone Index Explorer - {self.index_name} ===")
            print(f"Total vectors: {total_vectors}")
            print(f"Dimension: {self.dimension}")
            print(f"Namespaces: {list(stats.namespaces.keys()) if stats.namespaces else 'default'}")
            
            # Note: Pinecone doesn't have a direct "get all" method like ChromaDB
            # We can only search or get stats
            print("\nNote: Pinecone doesn't support browsing all vectors directly.")
            print("Use search functionality to find specific content.")
            
        except Exception as e:
            print(f"Error exploring database: {e}")
    
    def clear_index(self) -> None:
        """Clear all data from the index."""
        try:
            # Delete all vectors
            self.index.delete(delete_all=True)
            print(f"✅ Cleared all vectors from index '{self.index_name}'")
        except Exception as e:
            print(f"❌ Error clearing index: {e}")
