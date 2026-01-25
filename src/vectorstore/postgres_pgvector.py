"""
Postgres + pgvector vector store for chunk embeddings.

Stores chunks with required provenance fields:
- doc_id
- page_number
- chunk_id
- text

Also supports deduplication via:
- unique chunk_id
- document hash
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Sequence

import psycopg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    doc_id: str
    doc_name: Optional[str]
    page_number: Optional[int]
    text: str
    score: float
    source_path: Optional[str]


class PostgresVectorStore:
    def __init__(self, dsn: str, embedding_dim: int):
        self.dsn = dsn
        self.embedding_dim = embedding_dim

    def _connect(self):
        return psycopg.connect(self.dsn, autocommit=True)

    def ensure_schema(self) -> None:
        """Create pgvector extension + tables if missing."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                # Execute each statement separately to avoid "multiple commands" error
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id TEXT PRIMARY KEY,
                        doc_hash TEXT,
                        source_path TEXT,
                        created_at TIMESTAMPTZ DEFAULT now()
                    );
                """)
                
                # Use f-string or manual substitution for the dimension since it's part of the type definition
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                        page_number INT,
                        chunk_index INT,
                        text TEXT NOT NULL,
                        embedding vector({self.embedding_dim}) NOT NULL
                    );
                """)
                
                cur.execute("CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);")
                
                # Note: Index creation for > 2000 dimensions (like 3072) is disabled due to Postgres limits.
                # Exact search (without index) will be used, which is accurate and fast for mid-sized datasets.
                # cur.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);")

    def upsert_document(self, doc_id: str, doc_hash: Optional[str], source_path: Optional[str]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (doc_id, doc_hash, source_path)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE
                    SET doc_hash = EXCLUDED.doc_hash,
                        source_path = EXCLUDED.source_path
                    """,
                    (doc_id, doc_hash, source_path),
                )

    def document_exists(self, doc_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM documents WHERE doc_id = %s LIMIT 1", (doc_id,))
                return cur.fetchone() is not None

    def document_exists_by_hash(self, doc_hash: str) -> Optional[str]:
        """Return doc_id if a document with this hash exists."""
        if not doc_hash:
            return None
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT doc_id FROM documents WHERE doc_hash = %s LIMIT 1", (doc_hash,))
                row = cur.fetchone()
                return row[0] if row else None

    def upsert_chunk(
        self,
        *,
        chunk_id: str,
        doc_id: str,
        page_number: Optional[int],
        chunk_index: int,
        text: str,
        embedding: Sequence[float],
    ) -> None:
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.embedding_dim}, got {len(embedding)}")

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chunks (chunk_id, doc_id, page_number, chunk_index, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET doc_id = EXCLUDED.doc_id,
                        page_number = EXCLUDED.page_number,
                        chunk_index = EXCLUDED.chunk_index,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding
                    """,
                    (chunk_id, doc_id, page_number, chunk_index, text, list(embedding)),
                )

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> List[VectorHit]:
        """
        Cosine distance in pgvector:
          embedding <=> query_embedding  (cosine distance)
        We'll convert to a score: score = 1 - distance.
        """
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(
                f"Query embedding dim mismatch: expected {self.embedding_dim}, got {len(query_embedding)}"
            )

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.chunk_id, c.doc_id, c.page_number, c.text,
                           1 - (c.embedding <=> %s::vector) AS score,
                           d.source_path
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.doc_id
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (list(query_embedding), list(query_embedding), top_k),
                )
                rows = cur.fetchall() or []

        return [
            VectorHit(
                chunk_id=r[0],
                doc_id=r[1],
                doc_name=os.path.basename(r[5]) if r[5] else None,
                page_number=r[2],
                text=r[3],
                score=float(r[4]),
                source_path=r[5],
            )
            for r in rows
        ]

