import os
import sys
import re
import uuid
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any
from src.config import Config

class MedicalDocumentIngestor:
    """
    Structure-Aware Medical Document Chunking & Indexing Pipeline:
    - Parses Markdown guidelines preserving clinical context
    - Contextual prefixing (document title + section path)
    - Multilingual local embedding generation
    - Persistent Qdrant vector storage
    """

    def __init__(self, storage_path: str = None, collection_name: str = None):
        self.storage_path = storage_path or Config.QDRANT_STORAGE_PATH
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        self.raw_dir = Config.DATA_RAW_DIR
        self.model_name = Config.EMBEDDING_MODEL_NAME

        print(f"[Ingestor] Loading embedding model: {self.model_name}...")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(self.model_name)

        # Initialize Qdrant Client (Local disk persistence)
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(path=self.storage_path)
        
        # Get embedding dimension dynamically
        test_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Create or recreate collection
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            print(f"[Ingestor] Creating Qdrant collection '{self.collection_name}' (dim: {test_dim})...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=test_dim, distance=Distance.COSINE),
            )
        else:
            print(f"[Ingestor] Found existing collection '{self.collection_name}'.")

    def parse_markdown_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Splits markdown guideline by headings (##), generating context-prefixed chunks."""
        text = file_path.read_text(encoding="utf-8")
        
        # Extract metadata headers if present
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else file_path.stem.replace("_", " ").title()

        source_match = re.search(r"Source:\s*(.+)$", text, re.MULTILINE)
        doc_source = source_match.group(1).strip() if source_match else "Official Health Guideline"

        authority_match = re.search(r"Authority:\s*(.+)$", text, re.MULTILINE)
        doc_authority = authority_match.group(1).strip() if authority_match else "High"

        # Split on section headings (## )
        sections = re.split(r"(?m)^##\s+", text)
        chunks = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            lines = section.split("\n", 1)
            section_title = lines[0].strip()
            section_body = lines[1].strip() if len(lines) > 1 else ""

            if not section_body:
                continue

            # Contextual prefix prepending: guarantees isolated chunks retain full context
            context_prefixed_content = f"[{doc_title} > {section_title}]\n{section_body}"

            chunk_id = str(uuid.uuid4())
            chunks.append({
                "chunk_id": chunk_id,
                "doc_title": doc_title,
                "section_title": section_title,
                "source": doc_source,
                "authority": doc_authority,
                "file_name": file_path.name,
                "content": context_prefixed_content,
                "raw_body": section_body
            })

        return chunks

    def run_ingestion(self) -> int:
        from qdrant_client.models import PointStruct

        md_files = list(self.raw_dir.glob("*.md"))
        if not md_files:
            print(f"[Ingestor] Warning: No markdown files found in {self.raw_dir}")
            return 0

        all_chunks = []
        for f in md_files:
            file_chunks = self.parse_markdown_file(f)
            all_chunks.extend(file_chunks)
            print(f"[Ingestor] Parsed {len(file_chunks)} chunks from {f.name}")

        print(f"[Ingestor] Generating multilingual embeddings for {len(all_chunks)} chunks...")
        texts_to_embed = [c["content"] for c in all_chunks]
        embeddings = self.embedder.encode(texts_to_embed, show_progress_bar=True, normalize_embeddings=True)

        points = []
        for idx, (chunk, emb) in enumerate(zip(all_chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx + 1,
                    vector=emb.tolist(),
                    payload=chunk
                )
            )

        print(f"[Ingestor] Uploading {len(points)} points to collection '{self.collection_name}'...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"[Ingestor] Ingestion complete! Total chunks indexed: {len(points)}")
        return len(points)

if __name__ == "__main__":
    ingestor = MedicalDocumentIngestor()
    ingestor.run_ingestion()
