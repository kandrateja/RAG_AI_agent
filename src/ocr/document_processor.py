"""
Azure Document Intelligence OCR Processor
"""
import os
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
            
            # Extract pages
            pages = []
            if result.pages:
                for page in result.pages:
                    pages.append({
                        "page_number": page.page_number,
                        "width": page.width,
                        "height": page.height
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
        
        # If we have page information, try to split text by pages
        # Note: Azure Document Intelligence may not provide page-level text directly
        # This is a simplified approach - you may need to enhance based on actual API response
        pages = []
        if pages_info and full_text:
            # Estimate page boundaries (simplified - adjust based on actual API)
            total_chars = len(full_text)
            num_pages = len(pages_info)
            chars_per_page = total_chars // num_pages if num_pages > 0 else total_chars
            
            for i, page_info in enumerate(pages_info):
                start_idx = i * chars_per_page
                end_idx = (i + 1) * chars_per_page if i < num_pages - 1 else total_chars
                page_text = full_text[start_idx:end_idx].strip()
                
                pages.append({
                    "page_number": page_info.get("page_number", i + 1),
                    "text": page_text
                })
        else:
            # Fallback: single page
            pages = [{"page_number": 1, "text": full_text}]
        
        return {
            "text": full_text,
            "pages": pages
        }
    
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
