"""
Text Chunking Utilities for GraphRAG
Supports Arabic-aware sentence boundary detection.
"""
from typing import List, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunk text into smaller pieces for embedding and storage"""
    
    # Arabic punctuation marks for sentence boundaries
    ARABIC_SENTENCE_ENDS = [
        '\u06D4',  # Arabic full stop (۔)
        '\u061F',  # Arabic question mark (؟)
        '\u061B',  # Arabic semicolon (؛) - often used as sentence end
        '.',        # Western full stop (commonly used in modern Arabic)
        '!',        # Exclamation
        '?',        # Western question mark
        '\n',       # Newline
    ]
    
    # Arabic-specific regex for sentence splitting
    # Matches: Arabic full stop, Arabic question mark, Arabic semicolon, or Western punctuation
    ARABIC_SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?\u06D4\u061F\u061B])\s+'
    )
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_arabic_chunking: bool = False
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            use_arabic_chunking: If True, use Arabic-aware sentence boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_arabic_chunking = use_arabic_chunking
    
    def _is_predominantly_arabic(self, text: str, threshold: float = 0.3) -> bool:
        """Check if text is predominantly Arabic."""
        if not text or len(text.strip()) < 20:
            return False
        clean = text.replace(" ", "").replace("\n", "")
        if not clean:
            return False
        arabic_count = sum(1 for c in clean if "\u0600" <= c <= "\u06FF")
        return (arabic_count / len(clean)) >= threshold
    
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
            
            # Try to break at sentence boundary (Latin + Arabic punctuation)
            if end < len(text):
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end),
                    text.rfind('\n', start, end),
                    text.rfind('\u06D4', start, end),   # Arabic full stop (۔)
                    text.rfind('\u061B', start, end),   # Arabic semicolon (؛)
                    text.rfind('\u061F', start, end),   # Arabic question mark (؟)
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
    
    def _split_arabic_sentences(self, text: str) -> List[str]:
        """
        Split Arabic text into sentences using Arabic-aware boundaries.
        
        Handles:
        - Arabic full stop (۔)
        - Arabic question mark (؟)
        - Arabic semicolon (؛)
        - Western punctuation (used in modern Arabic)
        - Paragraph breaks
        """
        if not text:
            return []
        
        # First, split by paragraph breaks (multiple newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        sentences = []
        for para in paragraphs:
            if not para.strip():
                continue
            
            # Split by Arabic and Western sentence-ending punctuation
            # Pattern: lookbehind for sentence-ending punctuation, followed by whitespace
            para_sentences = self.ARABIC_SENTENCE_PATTERN.split(para)
            
            for sent in para_sentences:
                sent = sent.strip()
                if sent:
                    sentences.append(sent)
        
        return sentences
    
    def chunk_arabic_text(
        self,
        text: str,
        sentences_per_chunk: int = 3,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Chunk Arabic text using sentence boundaries.
        
        This produces semantically coherent chunks by:
        1. Splitting text into proper Arabic sentences
        2. Grouping sentences up to max_chunk_size
        3. Ensuring chunks don't break mid-sentence
        
        Args:
            text: Arabic text to chunk
            sentences_per_chunk: Target number of sentences per chunk
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries
        """
        sentences = self._split_arabic_sentences(text)
        
        if not sentences:
            return []
        
        logger.debug(f"Arabic chunking: split into {len(sentences)} sentences")
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_size = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # Check if adding this sentence would exceed max size
            if current_chunk_size + sentence_len > max_chunk_size and current_chunk_sentences:
                # Flush current chunk
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append({
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                    "sentence_count": len(current_chunk_sentences),
                    "is_arabic": True,
                    **(metadata or {})
                })
                chunk_index += 1
                current_chunk_sentences = []
                current_chunk_size = 0
            
            current_chunk_sentences.append(sentence)
            current_chunk_size += sentence_len + 1  # +1 for space
            
            # Also check sentence count threshold
            if len(current_chunk_sentences) >= sentences_per_chunk:
                if current_chunk_size >= min_chunk_size:
                    chunk_text = ' '.join(current_chunk_sentences)
                    chunks.append({
                        "content": chunk_text,
                        "chunk_index": chunk_index,
                        "sentence_count": len(current_chunk_sentences),
                        "is_arabic": True,
                        **(metadata or {})
                    })
                    chunk_index += 1
                    current_chunk_sentences = []
                    current_chunk_size = 0
        
        # Flush remaining sentences
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append({
                "content": chunk_text,
                "chunk_index": chunk_index,
                "sentence_count": len(current_chunk_sentences),
                "is_arabic": True,
                **(metadata or {})
            })
        
        logger.info(f"Arabic chunking: produced {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks
    
    def chunk_text_smart(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Smart chunking that auto-detects language and uses appropriate strategy.
        
        - For Arabic text: Use Arabic sentence boundary chunking
        - For other text: Use standard character-based chunking
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries
        """
        if self.use_arabic_chunking and self._is_predominantly_arabic(text):
            logger.info("Using Arabic-specific sentence chunking")
            return self.chunk_arabic_text(
                text,
                sentences_per_chunk=4,  # ~4 Arabic sentences per chunk
                min_chunk_size=150,
                max_chunk_size=self.chunk_size,
                metadata=metadata
            )
        else:
            return self.chunk_text(text, metadata=metadata)
    
    def chunk_text_by_pages_smart(self, pages: List[Dict], metadata: Dict = None) -> List[Dict]:
        """
        Smart page-based chunking with automatic Arabic detection.
        
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
            
            page_metadata = {**(metadata or {}), "page_number": page_number}
            
            # Use smart chunking (auto-detects Arabic)
            page_chunks = self.chunk_text_smart(page_text, metadata=page_metadata)
            
            # Update chunk indices
            for chunk in page_chunks:
                chunk["chunk_index"] = chunk_index
                chunk["page_number"] = page_number
                chunk_index += 1
            
            all_chunks.extend(page_chunks)
        
        return all_chunks
