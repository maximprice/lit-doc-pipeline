"""
Semantic vector search indexer using Chroma and Ollama embeddings.

This module provides semantic search for litigation document chunks using
Chroma vector database with nomic-embed-text embeddings from Ollama.

NOTE: ChromaDB import is delayed to runtime to avoid Python 3.14 compatibility issues.
"""

import logging
import requests
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from citation_types import Chunk

logger = logging.getLogger(__name__)


class VectorIndexer:
    """
    Semantic vector search using Chroma and Ollama embeddings.

    This provides semantic similarity search for finding chunks
    with similar meaning, even if they use different words.
    """

    def __init__(
        self,
        persist_directory: str,
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434/api/embeddings",
        collection_name: str = "lit_docs"
    ):
        """
        Initialize vector indexer.

        Args:
            persist_directory: Directory to persist Chroma database
            embedding_model: Ollama embedding model name
            ollama_url: Ollama API endpoint for embeddings
            collection_name: Chroma collection name
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self.collection_name = collection_name

        self.client = None
        self.collection = None
        self._chromadb = None
        self._chromadb_error = None

        # Check Ollama availability
        self._ollama_available = self._check_ollama()

    def _load_chromadb(self) -> bool:
        """Lazy-load chromadb module to avoid import-time errors."""
        if self._chromadb is not None:
            return True

        if self._chromadb_error is not None:
            return False

        try:
            import chromadb
            self._chromadb = chromadb
            return True
        except Exception as e:
            self._chromadb_error = str(e)
            logger.error(f"Failed to import chromadb: {e}")
            logger.error("ChromaDB has compatibility issues with Python 3.14")
            logger.error("Vector search will not be available")
            return False

    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        if not self._load_chromadb():
            return False

        try:
            response = requests.get(
                self.ollama_url.replace("/api/embeddings", "/api/tags"),
                timeout=2
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                has_model = any(self.embedding_model in name for name in model_names)
                if has_model:
                    logger.info(f"Ollama available with {self.embedding_model}")
                    return True
                else:
                    logger.warning(f"Ollama available but {self.embedding_model} not found")
                    return False
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    def is_available(self) -> bool:
        """Check if vector search is available."""
        return self._ollama_available

    # nomic-embed-text context is 8192 tokens.  Token/char ratio varies by
    # vocabulary density; legal text can be as low as ~2.5 chars/token.
    # We try progressively shorter truncations on context-length errors.
    _TRUNCATION_LIMITS = [5000, 3000, 2000]

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding vector for text from Ollama.

        Automatically truncates and retries on context-length errors.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if failed
        """
        if not self._ollama_available:
            return None

        # First attempt with text as-is (or capped at first limit)
        limits = [len(text)] + self._TRUNCATION_LIMITS
        for limit in limits:
            prompt = text[:limit] if limit < len(text) else text
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.embedding_model,
                        "prompt": prompt
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json().get("embedding")
                # Context-length error — try shorter
                if response.status_code == 500 and limit == len(text):
                    continue
                if response.status_code == 500:
                    continue
                logger.error("Ollama embedding failed: %d", response.status_code)
                return None
            except Exception as e:
                logger.error("Error getting embedding: %s", e)
                return None

        logger.warning("Text too long for embedding even at %d chars", self._TRUNCATION_LIMITS[-1])
        return None

    def _init_client(self) -> None:
        """Initialize Chroma client."""
        if not self._load_chromadb():
            raise RuntimeError("ChromaDB not available")

        if self.client is None:
            self.client = self._chromadb.PersistentClient(
                path=str(self.persist_directory)
            )

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Build vector index from chunks.

        Args:
            chunks: List of Chunk objects to index
        """
        if not self._ollama_available:
            logger.warning("Ollama not available, skipping vector index build")
            return

        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        logger.info(f"Building vector index for {len(chunks)} chunks...")

        # Initialize Chroma client
        self._init_client()

        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        # Create collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Litigation document chunks"}
        )

        # Batch embed and add chunks
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._add_batch(batch)

            if (i + batch_size) % 50 == 0:
                logger.info(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

        logger.info(f"Vector index built: {len(chunks)} chunks")

    def _add_batch(self, chunks: List[Chunk]) -> None:
        """
        Add a batch of chunks to the collection.

        Args:
            chunks: Batch of chunks to add
        """
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            # Get embedding
            embedding = self._get_embedding(chunk.core_text)
            if embedding is None:
                logger.warning(f"Failed to embed chunk {chunk.chunk_id}, skipping")
                continue

            ids.append(chunk.chunk_id)
            embeddings.append(embedding)
            documents.append(chunk.core_text)

            # Store metadata
            metadata = {
                "citation_string": chunk.citation_string,
                "doc_type": chunk.doc_type.value if hasattr(chunk.doc_type, 'value') else str(chunk.doc_type),
                "pages": ",".join(map(str, chunk.pages)),
                "tokens": chunk.tokens
            }
            metadatas.append(metadata)

        # Add to collection
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

    def load_index(self) -> None:
        """Load existing index."""
        if not self._ollama_available:
            logger.warning("Ollama not available, cannot load vector index")
            return

        self._init_client()

        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            count = self.collection.count()
            logger.info(f"Vector index loaded: {count} chunks")
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            raise

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for chunks semantically similar to query.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if not self._ollama_available:
            logger.warning("Ollama not available, cannot perform semantic search")
            return []

        if self.collection is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first")

        if not query.strip():
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            logger.error("Failed to embed query")
            return []

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Extract results
        chunk_scores = []
        if results['ids'] and len(results['ids']) > 0:
            ids = results['ids'][0]
            distances = results['distances'][0]

            # Convert distances to similarity scores (1 - normalized_distance)
            # Chroma uses L2 distance, convert to similarity score [0, 1]
            for chunk_id, distance in zip(ids, distances):
                # Convert distance to similarity (inverse relationship)
                # Using exponential decay: similarity = exp(-distance)
                similarity = float(1.0 / (1.0 + distance))
                chunk_scores.append((chunk_id, similarity))

        # Normalize scores to [0, 1]
        if chunk_scores:
            chunk_scores = self._normalize_scores(chunk_scores)

        return chunk_scores

    def _normalize_scores(self, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Normalize scores to [0, 1] range.

        Args:
            results: List of (chunk_id, score) tuples

        Returns:
            List with normalized scores
        """
        if not results:
            return results

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score == min_score:
            return [(chunk_id, 1.0) for chunk_id, _ in results]

        normalized = [
            (chunk_id, (score - min_score) / (max_score - min_score))
            for chunk_id, score in results
        ]

        return normalized
