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
import re
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
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    image_b64: Optional[str] = None  # Base64-encoded image if present
    content_type: str = "text"  # "text" or "image" or "mixed"


class PostgresVectorStore:
    def __init__(self, dsn: str, embedding_dim: int):
        self.dsn = dsn
        self.embedding_dim = embedding_dim
        self.keyword_weight = 0.3
        self.semantic_weight = 0.7

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
                
                # If chunks exists with wrong embedding dimension (e.g. switched from Azure 1536 to Titan 1024), recreate it
                cur.execute("""
                    SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'chunks'
                """)
                if cur.fetchone():
                    cur.execute("""
                        SELECT format_type(a.atttypid, a.atttypmod)
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid
                        WHERE c.relname = 'chunks' AND a.attname = 'embedding'
                          AND a.attnum > 0 AND NOT a.attisdropped
                    """)
                    row = cur.fetchone()
                    if row:
                        # format_type returns e.g. 'vector(1536)' or 'vector(1024)'
                        match = re.search(r"vector\((\d+)\)", row[0] or "")
                        if match:
                            existing_dim = int(match.group(1))
                            if existing_dim != self.embedding_dim:
                                logger.warning(
                                    "chunks table has embedding dimension %s but current model uses %s; recreating chunks table",
                                    existing_dim, self.embedding_dim,
                                )
                                cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
                
                # Use f-string or manual substitution for the dimension since it's part of the type definition
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                        page_number INT,
                        chunk_index INT,
                        text TEXT NOT NULL,
                        embedding vector({self.embedding_dim}) NOT NULL,
                        image_b64 TEXT,
                        content_type TEXT DEFAULT 'text'
                    );
                """)
                
                # Add columns if they don't exist (for existing databases)
                cur.execute("""
                    DO $$ 
                    BEGIN
                        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                       WHERE table_name='chunks' AND column_name='image_b64') THEN
                            ALTER TABLE chunks ADD COLUMN image_b64 TEXT;
                        END IF;
                        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                       WHERE table_name='chunks' AND column_name='content_type') THEN
                            ALTER TABLE chunks ADD COLUMN content_type TEXT DEFAULT 'text';
                        END IF;
                    END $$;
                """)
                
                cur.execute("CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);")
                
                # Full-text index for keyword search
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS chunks_text_fts_idx "
                    "ON chunks USING GIN (to_tsvector('english', text));"
                )

                # HNSW index is supported only for <= 2000 dimensions
                if self.embedding_dim <= 2000:
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx "
                        "ON chunks USING hnsw (embedding vector_cosine_ops);"
                    )
                else:
                    logger.info(
                        "Skipping HNSW index creation; embedding_dim=%s exceeds 2000",
                        self.embedding_dim,
                    )

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
        image_b64: Optional[str] = None,
        content_type: str = "text",
    ) -> None:
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.embedding_dim}, got {len(embedding)}")

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chunks (chunk_id, doc_id, page_number, chunk_index, text, embedding, image_b64, content_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET doc_id = EXCLUDED.doc_id,
                        page_number = EXCLUDED.page_number,
                        chunk_index = EXCLUDED.chunk_index,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        image_b64 = EXCLUDED.image_b64,
                        content_type = EXCLUDED.content_type
                    """,
                    (chunk_id, doc_id, page_number, chunk_index, text, list(embedding), image_b64, content_type),
                )

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        query_text: Optional[str] = None,
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

        semantic_rows = self._semantic_search(query_embedding, top_k=top_k * 2)
        if query_text:
            keyword_rows = self._keyword_search(query_text, top_k=top_k * 2)
            return self._merge_semantic_keyword(semantic_rows, keyword_rows, top_k=top_k)
        return self._rows_to_hits(semantic_rows)

    def _semantic_search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> List[tuple]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.chunk_id, c.doc_id, c.page_number, c.text,
                           1 - (c.embedding <=> %s::vector) AS score,
                           d.source_path, c.image_b64, c.content_type
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.doc_id
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (list(query_embedding), list(query_embedding), top_k),
                )
                return cur.fetchall() or []

    def _keyword_search(self, query_text: str, top_k: int) -> List[tuple]:
        if not query_text.strip():
            return []
        
        # Convert query to flexible OR-based tsquery (any word can match)
        # Split into words, filter out empty, join with OR operator
        words = [w.strip() for w in re.split(r'\s+', query_text.lower()) if w.strip()]
        if not words:
            return []
        
        # Build OR query: word1 | word2 | word3
        # Escape special characters and join with OR operator
        or_query = ' | '.join([re.sub(r'[^\w\s]', '', w) for w in words if re.sub(r'[^\w\s]', '', w)])
        
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.chunk_id, c.doc_id, c.page_number, c.text,
                           ts_rank_cd(
                             to_tsvector('english', c.text),
                             to_tsquery('english', %s)
                           ) AS kw_score,
                           d.source_path, c.image_b64, c.content_type
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.doc_id
                    WHERE to_tsvector('english', c.text) @@ to_tsquery('english', %s)
                    ORDER BY kw_score DESC
                    LIMIT %s
                    """,
                    (or_query, or_query, top_k),
                )
                return cur.fetchall() or []

    def _merge_semantic_keyword(
        self,
        semantic_rows: List[tuple],
        keyword_rows: List[tuple],
        top_k: int,
    ) -> List[VectorHit]:
        semantic_map: Dict[str, tuple] = {r[0]: r for r in semantic_rows}
        keyword_map: Dict[str, tuple] = {r[0]: r for r in keyword_rows}
        max_kw = max([float(r[4]) for r in keyword_rows], default=0.0)
        if max_kw <= 0.0:
            max_kw = 1.0

        combined: List[tuple] = []
        for chunk_id in set(semantic_map.keys()) | set(keyword_map.keys()):
            sem_row = semantic_map.get(chunk_id)
            kw_row = keyword_map.get(chunk_id)
            base_row = sem_row or kw_row
            sem_score = float(sem_row[4]) if sem_row else 0.0
            kw_score = float(kw_row[4]) if kw_row else 0.0
            kw_norm = kw_score / max_kw
            combined_score = (sem_score * self.semantic_weight) + (kw_norm * self.keyword_weight)
            combined.append((base_row, combined_score, sem_score, kw_norm))

        combined.sort(key=lambda x: x[1], reverse=True)
        return [
            VectorHit(
                chunk_id=row[0],
                doc_id=row[1],
                doc_name=os.path.basename(row[5]) if row[5] else None,
                page_number=row[2],
                text=row[3],
                score=float(score),
                source_path=row[5],
                semantic_score=float(sem_score),
                keyword_score=float(kw_norm),
                image_b64=row[6] if len(row) > 6 else None,
                content_type=row[7] if len(row) > 7 else "text",
            )
            for row, score, sem_score, kw_norm in combined[:top_k]
        ]

    def _rows_to_hits(self, rows: List[tuple]) -> List[VectorHit]:
        return [
            VectorHit(
                chunk_id=r[0],
                doc_id=r[1],
                doc_name=os.path.basename(r[5]) if r[5] else None,
                page_number=r[2],
                text=r[3],
                score=float(r[4]),
                source_path=r[5],
                semantic_score=float(r[4]),
                keyword_score=None,
                image_b64=r[6] if len(r) > 6 else None,
                content_type=r[7] if len(r) > 7 else "text",
            )
            for r in rows
        ]

    def get_image_chunks(self, limit: int = 10) -> List[VectorHit]:
        """
        Retrieve image chunks from the database for visual queries.
        Returns image chunks ordered by page number.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.chunk_id, c.doc_id, c.page_number, c.text,
                           1.0 AS score, d.source_path, c.image_b64, c.content_type
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.doc_id
                    WHERE c.content_type = 'image' AND c.image_b64 IS NOT NULL
                    ORDER BY c.doc_id, c.page_number
                    LIMIT %s
                    """,
                    (limit,),
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
                        semantic_score=float(r[4]),
                        keyword_score=None,
                        image_b64=r[6],
                        content_type=r[7],
                    )
                    for r in rows
                ]

