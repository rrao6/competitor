"""
ChromaDB vector store tools for semantic memory.

Provides embedding and similarity search for intel items.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from radar.config import get_config, get_settings


# Lazy-loaded ChromaDB client and collection
_chroma_client = None
_collection = None


def get_chroma_client():
    """Get or create the ChromaDB client."""
    global _chroma_client
    
    if _chroma_client is None:
        import chromadb
        from chromadb.config import Settings
        
        # Determine persistence directory
        persist_dir = Path(__file__).parent.parent.parent / "data" / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        _chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
    
    return _chroma_client


def get_collection():
    """Get or create the intel embeddings collection."""
    global _collection
    
    if _collection is None:
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        
        config = get_config()
        settings = get_settings()
        
        # Create embedding function using OpenAI
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=config.global_config.models.embedding,
        )
        
        client = get_chroma_client()
        collection_name = config.global_config.chroma.collection_name
        
        _collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    
    return _collection


@tool
def embed_and_index_intel(intel_id: int, text: str, metadata: Optional[dict] = None) -> bool:
    """
    Embed and index an intel item in the vector store.
    
    Args:
        intel_id: The database ID of the intel item
        text: The text to embed (typically the summary)
        metadata: Optional metadata to store with the embedding
    
    Returns:
        True if successful
    """
    try:
        collection = get_collection()
        
        # Prepare metadata
        doc_metadata = metadata or {}
        
        # Add or update the document
        collection.upsert(
            ids=[str(intel_id)],
            documents=[text],
            metadatas=[doc_metadata],
        )
        
        return True
    except Exception as e:
        print(f"Error indexing intel {intel_id}: {e}")
        return False


def embed_intel_batch(items: list[dict]) -> int:
    """
    Embed and index multiple intel items.
    
    Args:
        items: List of dicts with intel_id, text, and optional metadata
    
    Returns:
        Number of items successfully indexed
    """
    if not items:
        return 0
    
    try:
        collection = get_collection()
        
        ids = [str(item["intel_id"]) for item in items]
        documents = [item["text"] for item in items]
        metadatas = [item.get("metadata", {}) for item in items]
        
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        
        return len(items)
    except Exception as e:
        print(f"Error batch indexing intel: {e}")
        return 0


@tool
def search_similar_intel(
    text: str,
    top_k: int = 5,
    category_filter: Optional[str] = None,
) -> list[dict]:
    """
    Search for similar intel items in the vector store.
    
    Args:
        text: The query text
        top_k: Number of results to return
        category_filter: Optional category to filter by
    
    Returns:
        List of similar intel items with scores
    """
    try:
        collection = get_collection()
        
        # Build where clause if filtering
        where = None
        if category_filter:
            where = {"category": category_filter}
        
        results = collection.query(
            query_texts=[text],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        # Transform results
        similar_items = []
        if results["ids"] and results["ids"][0]:
            for i, intel_id in enumerate(results["ids"][0]):
                # Convert distance to similarity (cosine distance to similarity)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance  # Cosine similarity
                
                similar_items.append({
                    "intel_id": int(intel_id),
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": similarity,
                })
        
        return similar_items
    except Exception as e:
        print(f"Error searching similar intel: {e}")
        return []


def find_duplicates(
    text: str,
    threshold: float = 0.85,
    exclude_ids: Optional[list[int]] = None,
) -> list[dict]:
    """
    Find potential duplicates for a given text.
    
    Args:
        text: The text to check for duplicates
        threshold: Similarity threshold (0-1)
        exclude_ids: Intel IDs to exclude from results
    
    Returns:
        List of potential duplicate intel items above threshold
    """
    results = search_similar_intel.invoke({
        "text": text,
        "top_k": 10,
    })
    
    exclude_set = set(exclude_ids or [])
    
    duplicates = []
    for item in results:
        if item["intel_id"] in exclude_set:
            continue
        if item["similarity"] >= threshold:
            duplicates.append(item)
    
    return duplicates


def reset_vector_store() -> bool:
    """
    Reset the vector store (delete all embeddings). USE WITH CAUTION.
    
    Returns:
        True if successful
    """
    global _collection
    
    try:
        client = get_chroma_client()
        config = get_config()
        collection_name = config.global_config.chroma.collection_name
        
        # Delete and recreate collection
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # Collection might not exist
        
        _collection = None  # Force recreation
        get_collection()  # Recreate
        
        return True
    except Exception as e:
        print(f"Error resetting vector store: {e}")
        return False

