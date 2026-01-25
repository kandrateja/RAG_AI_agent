"""
Text Chunking Utilities for GraphRAG
"""
from typing import List, Dict
import re


class TextChunker:
    """Chunk text into smaller pieces for embedding and storage"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end),
                    text.rfind('\n', start, end)
                )
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_data = {
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    **(metadata or {})
                }
                chunks.append(chunk_data)
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    
    def chunk_text_by_pages(self, pages: List[Dict], metadata: Dict = None) -> List[Dict]:
        """
        Chunk text that is already split by pages, preserving page numbers
        
        Args:
            pages: List of page dictionaries with 'page_number' and 'text'
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with page_number preserved
        """
        all_chunks = []
        chunk_index = 0
        
        for page in pages:
            page_number = page.get("page_number", 1)
            page_text = page.get("text", "")
            
            if not page_text:
                continue
            
            # Chunk this page's text
            page_chunks = self.chunk_text(
                page_text,
                metadata={**(metadata or {}), "page_number": page_number}
            )
            
            # Update chunk indices and add page numbers
            for chunk in page_chunks:
                chunk["chunk_index"] = chunk_index
                chunk["page_number"] = page_number
                chunk_index += 1
            
            all_chunks.extend(page_chunks)
        
        return all_chunks
    
    def chunk_by_sentences(self, text: str, sentences_per_chunk: int = 5, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text by sentences
        
        Args:
            text: Input text to chunk
            sentences_per_chunk: Number of sentences per chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = ' '.join(chunk_sentences)
            
            chunk_data = {
                "content": chunk_text,
                "chunk_index": i // sentences_per_chunk,
                "start_sentence": i,
                "end_sentence": min(i + sentences_per_chunk, len(sentences)),
                **(metadata or {})
            }
            chunks.append(chunk_data)
        
        return chunks
