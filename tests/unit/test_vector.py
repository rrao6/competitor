"""Tests for vector store operations."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


class TestVectorStore:
    """Tests for ChromaDB vector store operations."""
    
    @patch("radar.tools.vector.get_collection")
    def test_embed_and_index_intel(self, mock_get_collection, mock_config):
        """Test embedding and indexing intel."""
        from radar.tools.vector import embed_and_index_intel
        
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        
        result = embed_and_index_intel.invoke({
            "intel_id": 1,
            "text": "Netflix announced new streaming features.",
            "metadata": {"category": "product"},
        })
        
        assert result is True
        mock_collection.upsert.assert_called_once()
    
    @patch("radar.tools.vector.get_collection")
    def test_embed_intel_batch(self, mock_get_collection, mock_config):
        """Test batch embedding of intel items."""
        from radar.tools.vector import embed_intel_batch
        
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        
        items = [
            {"intel_id": 1, "text": "Netflix news", "metadata": {}},
            {"intel_id": 2, "text": "Disney news", "metadata": {}},
            {"intel_id": 3, "text": "Roku news", "metadata": {}},
        ]
        
        count = embed_intel_batch(items)
        
        assert count == 3
        mock_collection.upsert.assert_called_once()
    
    @patch("radar.tools.vector.get_collection")
    def test_embed_intel_batch_empty(self, mock_get_collection, mock_config):
        """Test batch embedding with empty list."""
        from radar.tools.vector import embed_intel_batch
        
        count = embed_intel_batch([])
        
        assert count == 0
        mock_get_collection.assert_not_called()
    
    @patch("radar.tools.vector.get_collection")
    def test_search_similar_intel(self, mock_get_collection, mock_config):
        """Test searching for similar intel."""
        from radar.tools.vector import search_similar_intel
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["1", "2"]],
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"category": "product"}, {"category": "content"}]],
            "distances": [[0.1, 0.3]],
        }
        mock_get_collection.return_value = mock_collection
        
        results = search_similar_intel.invoke({
            "text": "Netflix streaming features",
            "top_k": 5,
        })
        
        assert len(results) == 2
        assert results[0]["intel_id"] == 1
        assert results[0]["similarity"] == 0.9  # 1 - 0.1
    
    @patch("radar.tools.vector.get_collection")
    def test_search_similar_intel_with_filter(self, mock_get_collection, mock_config):
        """Test searching with category filter."""
        from radar.tools.vector import search_similar_intel
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "documents": [["Doc 1"]],
            "metadatas": [[{"category": "product"}]],
            "distances": [[0.2]],
        }
        mock_get_collection.return_value = mock_collection
        
        results = search_similar_intel.invoke({
            "text": "Test",
            "top_k": 5,
            "category_filter": "product",
        })
        
        mock_collection.query.assert_called_once()
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["where"] == {"category": "product"}
    
    @patch("radar.tools.vector.get_collection")
    def test_search_similar_intel_empty_results(self, mock_get_collection, mock_config):
        """Test searching with no results."""
        from radar.tools.vector import search_similar_intel
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_get_collection.return_value = mock_collection
        
        results = search_similar_intel.invoke({
            "text": "Unknown topic",
            "top_k": 5,
        })
        
        assert results == []
    
    @patch("radar.tools.vector.search_similar_intel")
    def test_find_duplicates(self, mock_search, mock_config):
        """Test finding duplicates."""
        from radar.tools.vector import find_duplicates
        
        mock_search.invoke.return_value = [
            {"intel_id": 1, "similarity": 0.95, "document": "Doc", "metadata": {}},
            {"intel_id": 2, "similarity": 0.7, "document": "Doc", "metadata": {}},
        ]
        
        duplicates = find_duplicates(
            text="Test text",
            threshold=0.85,
            exclude_ids=[3],
        )
        
        assert len(duplicates) == 1
        assert duplicates[0]["intel_id"] == 1
    
    @patch("radar.tools.vector.search_similar_intel")
    def test_find_duplicates_with_exclusion(self, mock_search, mock_config):
        """Test finding duplicates with exclusion list."""
        from radar.tools.vector import find_duplicates
        
        mock_search.invoke.return_value = [
            {"intel_id": 1, "similarity": 0.95, "document": "Doc", "metadata": {}},
            {"intel_id": 2, "similarity": 0.90, "document": "Doc", "metadata": {}},
        ]
        
        duplicates = find_duplicates(
            text="Test text",
            threshold=0.85,
            exclude_ids=[1],  # Exclude the first result
        )
        
        assert len(duplicates) == 1
        assert duplicates[0]["intel_id"] == 2


class TestChromaClient:
    """Tests for ChromaDB client initialization."""
    
    def test_get_chroma_client_returns_client(self, mock_config):
        """Test that ChromaDB client is returned."""
        import radar.tools.vector as vector_module
        
        # Set a mock client directly
        mock_client = MagicMock()
        vector_module._chroma_client = mock_client
        
        client = vector_module.get_chroma_client()
        
        assert client == mock_client
    
    def test_chroma_client_is_cached(self, mock_config):
        """Test that client is cached on subsequent calls."""
        import radar.tools.vector as vector_module
        
        mock_client = MagicMock()
        vector_module._chroma_client = mock_client
        
        client1 = vector_module.get_chroma_client()
        client2 = vector_module.get_chroma_client()
        
        assert client1 is client2

