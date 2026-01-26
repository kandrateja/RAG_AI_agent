"""
Azure Document Intelligence OCR Processor
"""
import os
import fitz
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents using Azure Document Intelligence API"""
    
    def __init__(self, endpoint: str, key: str):
        """
        Initialize Document Intelligence client
        
        Args:
            endpoint: Azure Document Intelligence endpoint URL
            key: Azure Document Intelligence API key
        """
        # Ensure endpoint has scheme to avoid "No connection adapters" errors
        if endpoint and not endpoint.startswith(("http://", "https://")):
            endpoint = f"https://{endpoint}"
        self.endpoint = endpoint
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
    
    def analyze_document(self, document_path: str, model_id: str = "prebuilt-read") -> Dict:
        """
        Analyze a document using Azure Document Intelligence
        
        Args:
            document_path: Path to the document file
            model_id: Model ID to use (default: "prebuilt-read" for OCR)
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            with open(document_path, "rb") as f:
                poller = self.client.begin_analyze_document(
                    model_id=model_id,
                    body=f,
                    content_type="application/octet-stream"
                )
                result = poller.result()
            
            # Extract text content
            extracted_text = ""
            if result.content:
                extracted_text = result.content
            
            # Extract pages with page-level text
            pages = []
            if result.pages:
                for page in result.pages:
                    page_lines = []
                    if hasattr(page, "lines") and page.lines:
                        page_lines = [line.content for line in page.lines if getattr(line, "content", None)]
                    page_text = " ".join(page_lines).strip()
                    pages.append({
                        "page_number": page.page_number,
                        "width": page.width,
                        "height": page.height,
                        "text": page_text
                    })
            
            # Extract tables if present
            tables = []
            if result.tables:
                for table in result.tables:
                    tables.append({
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "cells": [
                            {
                                "row_index": cell.row_index,
                                "column_index": cell.column_index,
                                "content": cell.content
                            }
                            for cell in table.cells
                        ]
                    })
            
            return {
                "text": extracted_text,
                "pages": pages,
                "tables": tables,
                "model_id": result.model_id,
                "api_version": result.api_version
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            raise
    
    def extract_text(self, document_path: str) -> str:
        """
        Extract text content from a document
        
        Args:
            document_path: Path to the document file
            
        Returns:
            Extracted text content
        """
        result = self.analyze_document(document_path)
        return result.get("text", "")
    
    def extract_text_with_pages(self, document_path: str) -> Dict:
        """
        Extract text content with page-level information
        
        Args:
            document_path: Path to the document file
            
        Returns:
            Dictionary with 'text' (full text) and 'pages' (list of page dicts with page_number and text)
        """
        result = self.analyze_document(document_path)
        full_text = result.get("text", "")
        pages_info = result.get("pages", [])

        pages = []
        if pages_info:
            for i, page_info in enumerate(pages_info):
                page_text = (page_info.get("text") or "").strip()
                if not page_text and full_text:
                    # Fallback to full text if page text is missing
                    page_text = full_text
                pages.append({
                    "page_number": page_info.get("page_number", i + 1),
                    "text": page_text
                })
        else:
            pages = [{"page_number": 1, "text": full_text}]
        
        return {
            "text": full_text,
            "pages": pages
        }

    def extract_page_images(self, document_path: str, dpi: int = 150) -> List[Dict]:
        """
        Render each PDF page to an image for vision-based captioning.
        """
        images: List[Dict] = []
        try:
            doc = fitz.open(document_path)
            for idx, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi)
                images.append({
                    "page_number": idx + 1,
                    "image_bytes": pix.tobytes("png"),
                })
            return images
        except Exception as e:
            logger.error(f"Error extracting page images: {str(e)}")
            return images
    
    def process_batch(self, document_paths: List[str]) -> List[Dict]:
        """
        Process multiple documents
        
        Args:
            document_paths: List of document file paths
            
        Returns:
            List of extraction results
        """
        results = []
        for doc_path in document_paths:
            try:
                result = self.analyze_document(doc_path)
                result["file_path"] = doc_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {doc_path}: {str(e)}")
                results.append({"file_path": doc_path, "error": str(e)})
        
        return results
