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


# =============================================================================
# ENHANCED MEMORY CAPABILITIES
# =============================================================================

def get_competitor_collection():
    """Get or create the competitor profiles collection."""
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    
    settings = get_settings()
    client = get_chroma_client()
    
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name="text-embedding-3-small",
    )
    
    return client.get_or_create_collection(
        name="competitor_profiles",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def get_trends_collection():
    """Get or create the trends history collection."""
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    
    settings = get_settings()
    client = get_chroma_client()
    
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name="text-embedding-3-small",
    )
    
    return client.get_or_create_collection(
        name="trends_history",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def store_competitor_profile(
    competitor_id: str,
    profile_text: str,
    metadata: dict
) -> bool:
    """
    Store a competitor profile in the vector store.
    
    Args:
        competitor_id: Unique competitor ID
        profile_text: Full profile text for embedding
        metadata: Profile metadata (strengths, weaknesses, etc.)
        
    Returns:
        True if successful
    """
    try:
        collection = get_competitor_collection()
        collection.upsert(
            ids=[competitor_id],
            documents=[profile_text],
            metadatas=[metadata],
        )
        return True
    except Exception as e:
        print(f"Error storing competitor profile: {e}")
        return False


def get_competitor_context(competitor_id: str) -> Optional[dict]:
    """
    Retrieve competitor profile from memory.
    
    Returns:
        Profile dict or None
    """
    try:
        collection = get_competitor_collection()
        result = collection.get(
            ids=[competitor_id],
            include=["documents", "metadatas"],
        )
        
        if result["ids"]:
            return {
                "profile_text": result["documents"][0] if result["documents"] else "",
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
            }
        return None
    except Exception:
        return None


def find_similar_historical(
    query: str,
    top_k: int = 5,
    competitor_filter: Optional[str] = None,
) -> list[dict]:
    """
    Find similar historical intel for context.
    
    Useful for understanding how current news relates to past events.
    
    Args:
        query: The query text
        top_k: Number of results
        competitor_filter: Optional competitor to filter by
        
    Returns:
        List of similar historical items
    """
    try:
        collection = get_collection()
        
        where = None
        if competitor_filter:
            where = {"competitor_id": competitor_filter}
        
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        items = []
        if results["ids"] and results["ids"][0]:
            for i, intel_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                items.append({
                    "intel_id": int(intel_id),
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": 1 - distance,
                })
        
        return items
    except Exception as e:
        print(f"Error finding similar historical: {e}")
        return []


def store_trend(
    trend_id: str,
    trend_text: str,
    metadata: dict
) -> bool:
    """
    Store a trend in the history collection.
    
    Args:
        trend_id: Unique trend ID (include date)
        trend_text: Full trend description
        metadata: Trend metadata
        
    Returns:
        True if successful
    """
    try:
        collection = get_trends_collection()
        collection.upsert(
            ids=[trend_id],
            documents=[trend_text],
            metadatas=[metadata],
        )
        return True
    except Exception as e:
        print(f"Error storing trend: {e}")
        return False


def get_trend_evolution(trend_name: str, top_k: int = 10) -> list[dict]:
    """
    Get historical evolution of a trend.
    
    Args:
        trend_name: Name of the trend to track
        top_k: Number of historical records
        
    Returns:
        List of historical trend data
    """
    try:
        collection = get_trends_collection()
        
        results = collection.query(
            query_texts=[trend_name],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        items = []
        if results["ids"] and results["ids"][0]:
            for i, trend_id in enumerate(results["ids"][0]):
                items.append({
                    "trend_id": trend_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": 1 - (results["distances"][0][i] if results["distances"] else 0),
                })
        
        return items
    except Exception as e:
        print(f"Error getting trend evolution: {e}")
        return []


def build_context_for_analysis(
    intel_items: list[dict],
    max_historical: int = 10
) -> str:
    """
    Build a rich context string for analysis by finding related historical intel.
    
    Args:
        intel_items: Current intel items
        max_historical: Max historical items to include
        
    Returns:
        Context string
    """
    context_parts = []
    seen_ids = set()
    
    for item in intel_items[:5]:  # Use top 5 items as seeds
        summary = item.get("summary", "") or item.get("text", "")
        if not summary:
            continue
        
        historical = find_similar_historical(summary, top_k=3)
        for h in historical:
            if h["intel_id"] not in seen_ids and h["similarity"] > 0.6:
                seen_ids.add(h["intel_id"])
                context_parts.append(f"- [Historical] {h['text']}")
                
                if len(context_parts) >= max_historical:
                    break
        
        if len(context_parts) >= max_historical:
            break
    
    if context_parts:
        return "Historical context from vector memory:\n" + "\n".join(context_parts)
    
    return ""

