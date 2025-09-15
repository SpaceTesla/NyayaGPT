import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
from .embedder import ChunkWithEmbedding
from .chunker import embedding_model


class ChromaStorage:
    """ChromaDB storage for document chunks and embeddings."""
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def save_chunks(self, chunks: List[ChunkWithEmbedding], document_name: str = "document") -> None:
        """Save embedded chunks to ChromaDB."""
        if not chunks:
            print("No chunks to save!")
            return
        
        # Prepare data for ChromaDB
        ids = [f"{document_name}_{i}" for i in range(len(chunks))]
        texts = [chunk.text for chunk in chunks]
        embeddings = [chunk.embedding.tolist() for chunk in chunks]  # Convert numpy arrays to lists
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
            # Only add simple metadata values that ChromaDB can handle
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
        
        print(f"Saved {len(chunks)} chunks to ChromaDB collection '{self.collection.name}'")
    
    def search(self, query: str, n_results: int = 5, document_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using text query."""
        # Create query filter if document_name is specified
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
    
    def search_by_embedding(self, query_embedding: np.ndarray, n_results: int = 5, document_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using embedding vector."""
        where_filter = {"document_name": document_name} if document_name else None
        
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
    
    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Cleared collection '{self.collection.name}'")
    
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
    
    def search_with_filters(self, query: str, n_results: int = 5, 
                          document_name: Optional[str] = None,
                          min_text_length: Optional[int] = None,
                          max_text_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """Advanced search with multiple filters."""
        where_filter = {}
        
        if document_name:
            where_filter["document_name"] = document_name
        if min_text_length is not None:
            where_filter["text_length"] = {"$gte": min_text_length}
        if max_text_length is not None:
            where_filter["text_length"] = {"$lte": max_text_length}
        
        # Generate embedding for the query
        query_embedding = embedding_model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter if where_filter else None
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
    
    def get_document_chunks(self, document_name: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        results = self.collection.get(
            where={"document_name": document_name}
        )
        
        formatted_results = []
        for i in range(len(results['ids'])):
            formatted_results.append({
                'id': results['ids'][i],
                'text': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        return formatted_results
