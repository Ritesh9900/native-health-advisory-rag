import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any
from src.config import Config

class MedicalRetriever:
    """
    Multilingual Semantic Retriever:
    Queries Qdrant for health guidelines using sentence embeddings.
    Supports cross-lingual matching (e.g. Hindi or Spanish queries matching English clinical guidelines).
    """

    def __init__(self, embedder=None, qdrant_client=None):
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient

        self.model_name = Config.EMBEDDING_MODEL_NAME
        self.embedder = embedder or SentenceTransformer(self.model_name)
        self.client = qdrant_client or QdrantClient(path=Config.QDRANT_STORAGE_PATH)
        self.collection_name = Config.QDRANT_COLLECTION_NAME

    def retrieve(self, query: str, top_k: int = Config.TOP_K, min_score: float = Config.SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
        # Encode query
        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()

        try:
            # Newer qdrant-client supports query_points, search is universal
            if hasattr(self.client, "query_points"):
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True
                ).points
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True
                )
        except Exception as e:
            print(f"[Retriever] Search error: {e}")
            return []

        retrieved_chunks = []
        for r in results:
            score = getattr(r, "score", 0.0)
            if score >= min_score:
                payload = getattr(r, "payload", {})
                retrieved_chunks.append({
                    "score": round(float(score), 4),
                    "doc_title": payload.get("doc_title", "Unknown Guideline"),
                    "section_title": payload.get("section_title", "General"),
                    "source": payload.get("source", "Official Guideline"),
                    "authority": payload.get("authority", "High"),
                    "content": payload.get("content", ""),
                    "chunk_id": payload.get("chunk_id", "")
                })

        return retrieved_chunks
