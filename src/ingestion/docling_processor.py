"""
Docling-based document processor with pipeline selection.
Handles PDF/A, PDF/B, Arabic, handwritten, tables, and multilingual content.
Returns same shape as Azure extract_text_with_pages: {"text", "pages": [{"page_number", "text"}]}.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

DOCLING_AVAILABLE = False
DocumentConverter = None
PdfPipelineOptions = None
OcrMacOptions = None
TesseractCliOcrOptions = None

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
    # Try to import pipeline options for OCR configuration
    try:
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError:
        pass
    try:
        from docling.datamodel.pipeline_options import OcrMacOptions, TesseractCliOcrOptions
    except ImportError:
        pass
except ImportError:
    pass


class PipelineHint(str, Enum):
    """User-facing pipeline selection for ingestion."""
    AUTO = "auto"
    STANDARD = "standard"   # Typed/printed PDF-A, PDF-B, mixed
    ARABIC = "arabic"       # RTL / Arabic-dominant
    HANDWRITTEN = "handwritten"


def _page_no_from_item(item: Any) -> Optional[int]:
    """Get first page number from a Docling element's prov (provenance)."""
    prov = getattr(item, "prov", None)
    if not prov or not isinstance(prov, list):
        return None
    for p in prov:
        pno = getattr(p, "page_no", None)
        if pno is not None:
            return int(pno)
    return None


def _get_bbox_from_item(item: Any) -> Optional[Dict]:
    """Get bounding box from a Docling element's prov (provenance)."""
    prov = getattr(item, "prov", None)
    if not prov or not isinstance(prov, list):
        return None
    for p in prov:
        bbox = getattr(p, "bbox", None)
        if bbox is not None:
            # bbox is typically (x0, y0, x1, y1) or has l, t, r, b attributes
            if hasattr(bbox, 'l'):
                return {
                    "x0": bbox.l,
                    "y0": bbox.t, 
                    "x1": bbox.r,
                    "y1": bbox.b,
                    "page_no": getattr(p, "page_no", 1)
                }
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return {
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "page_no": getattr(p, "page_no", 1)
                }
    return None


def _text_from_item(item: Any) -> str:
    """Get text content from a text or table item."""
    text = getattr(item, "text", None)
    if text is not None and isinstance(text, str):
        return text.strip()
    
    # Tables: try multiple extraction methods
    data = getattr(item, "data", None)
    if data is not None:
        # Method 1: Try table_cells with proper row/column positioning
        cells = getattr(data, "table_cells", []) or []
        if cells:
            try:
                # Create a 2D grid: grid[row_idx][col_idx] = text
                grid: Dict[int, Dict[int, str]] = {}
                max_col = 0
                max_row = 0
                col_widths: Dict[int, int] = {}  # Track max width per column
                
                for c in cells:
                    r = getattr(c, "start_row_offset_idx", 0)
                    col = getattr(c, "start_col_offset_idx", 0)
                    txt = (getattr(c, "text", "") or "").strip()
                    # Clean up text - replace newlines with spaces for table cells
                    txt = " ".join(txt.split())
                    if r not in grid:
                        grid[r] = {}
                    grid[r][col] = txt
                    max_col = max(max_col, col)
                    max_row = max(max_row, r)
                    # Track column width
                    col_widths[col] = max(col_widths.get(col, 3), len(txt))
                
                # Debug: log table dimensions
                print(f"[DOCLING TABLE] Found {len(cells)} cells in {max_row+1} rows x {max_col+1} cols")
                
                # Build proper markdown table with header separator
                lines = []
                for row_idx in sorted(grid.keys()):
                    row_cells = []
                    for col_idx in range(max_col + 1):
                        cell_text = grid[row_idx].get(col_idx, "")
                        # Pad cell to column width for alignment
                        padded = cell_text.ljust(col_widths.get(col_idx, 3))
                        row_cells.append(padded)
                    lines.append("| " + " | ".join(row_cells) + " |")
                    
                    # Add separator after first row (header)
                    if row_idx == 0:
                        sep_cells = ["-" * col_widths.get(c, 3) for c in range(max_col + 1)]
                        lines.append("| " + " | ".join(sep_cells) + " |")
                
                result = "\n".join(lines)
                if max_col > 0:  # Multi-column table
                    print(f"[DOCLING TABLE] Extracted markdown table: {len(lines)} lines")
                return result
            except Exception as e:
                print(f"[DOCLING TABLE] Cell extraction failed: {e}")
        
        # Method 2: Try grid attribute (some Docling versions use this)
        grid_data = getattr(data, "grid", None)
        if grid_data:
            try:
                lines = []
                col_widths = {}
                # First pass: calculate column widths
                for row in grid_data:
                    for col_idx, cell in enumerate(row):
                        txt = str(cell).strip() if cell else ""
                        col_widths[col_idx] = max(col_widths.get(col_idx, 3), len(txt))
                
                # Second pass: build table
                for row_idx, row in enumerate(grid_data):
                    row_texts = []
                    for col_idx, cell in enumerate(row):
                        txt = str(cell).strip() if cell else ""
                        padded = txt.ljust(col_widths.get(col_idx, 3))
                        row_texts.append(padded)
                    lines.append("| " + " | ".join(row_texts) + " |")
                    
                    # Add separator after header
                    if row_idx == 0:
                        sep_cells = ["-" * col_widths.get(c, 3) for c in range(len(row))]
                        lines.append("| " + " | ".join(sep_cells) + " |")
                
                if lines:
                    print(f"[DOCLING TABLE] Extracted from grid: {len(lines)} lines")
                    return "\n".join(lines)
            except Exception as e:
                print(f"[DOCLING TABLE] Grid extraction failed: {e}")
        
        # Method 3: Try to_dataframe (pandas) if available
        try:
            df = getattr(data, "to_dataframe", None)
            if callable(df):
                table_df = df()
                result = table_df.to_markdown(index=False)
                print(f"[DOCLING TABLE] Extracted via DataFrame: {len(table_df)} rows")
                return result
        except Exception:
            pass
        
        # Method 4: Fallback to structured text representation
        if cells:
            try:
                # Group cells by row for better readability
                rows_dict: Dict[int, List[str]] = {}
                for c in cells:
                    r = getattr(c, "start_row_offset_idx", 0)
                    txt = (getattr(c, "text", "") or "").strip()
                    if r not in rows_dict:
                        rows_dict[r] = []
                    rows_dict[r].append(txt)
                
                lines = []
                for row_idx in sorted(rows_dict.keys()):
                    lines.append(" | ".join(rows_dict[row_idx]))
                return "\n".join(lines)
            except Exception:
                pass
    
    return ""


def _postprocess_table_text(text: str) -> str:
    """
    Post-process text that contains table-like patterns (| separators).
    Converts unstructured table text into proper markdown tables.
    """
    if "|" not in text:
        return text
    
    lines = text.split("\n")
    result_lines = []
    table_buffer = []
    in_table = False
    
    for line in lines:
        stripped = line.strip()
        # Check if this line looks like a table row (has | separators)
        pipe_count = stripped.count("|")
        
        # A table row should have at least one | and some text content
        if pipe_count >= 1 and len(stripped) > 3:
            # Check if it's NOT already a markdown table separator
            if stripped.replace("|", "").replace("-", "").replace(" ", "").replace(":", "") == "":
                # This is a separator line, keep it
                result_lines.append(line)
                continue
            
            in_table = True
            table_buffer.append(stripped)
        else:
            if in_table and table_buffer:
                # End of table section - process the buffer
                formatted_table = _format_table_buffer(table_buffer)
                result_lines.extend(formatted_table)
                table_buffer = []
            in_table = False
            result_lines.append(line)
    
    # Process any remaining table buffer
    if table_buffer:
        formatted_table = _format_table_buffer(table_buffer)
        result_lines.extend(formatted_table)
    
    return "\n".join(result_lines)


def _format_table_buffer(lines: List[str]) -> List[str]:
    """
    Format a list of table-like lines into proper markdown table format.
    """
    if not lines:
        return []
    
    # Parse each line into cells
    rows = []
    max_cols = 0
    
    for line in lines:
        # Split by | and clean up
        parts = line.split("|")
        cells = [p.strip() for p in parts if p.strip()]  # Remove empty cells
        if cells:
            rows.append(cells)
            max_cols = max(max_cols, len(cells))
    
    if not rows or max_cols == 0:
        return lines  # Return original if parsing failed
    
    # Normalize rows to have same number of columns
    for row in rows:
        while len(row) < max_cols:
            row.append("")
    
    # Calculate column widths
    col_widths = [3] * max_cols  # Minimum width of 3
    for row in rows:
        for i, cell in enumerate(row):
            if i < max_cols:
                col_widths[i] = max(col_widths[i], len(cell))
    
    # Build formatted table
    formatted = []
    for idx, row in enumerate(rows):
        # Pad each cell to column width
        padded_cells = [cell.ljust(col_widths[i]) for i, cell in enumerate(row)]
        formatted.append("| " + " | ".join(padded_cells) + " |")
        
        # Add separator after first row (header)
        if idx == 0:
            sep_cells = ["-" * col_widths[i] for i in range(max_cols)]
            formatted.append("| " + " | ".join(sep_cells) + " |")
    
    return formatted


def _build_pages_from_document(document: Any) -> List[Dict[str, Any]]:
    """
    Build page-level text from DoclingDocument: texts + tables (by prov page).
    Document has .texts (list of text items), .tables (list of table items).
    Items have .prov (list of ProvenanceItem with .page_no) and .text or .data.
    """
    page_texts: Dict[int, List[str]] = {}

    for item in getattr(document, "texts", []) or []:
        pno = _page_no_from_item(item)
        if pno is None:
            pno = 1
        text = _text_from_item(item)
        if text:
            if pno not in page_texts:
                page_texts[pno] = []
            page_texts[pno].append(text)

    tables_list = getattr(document, "tables", []) or []
    print(f"[DOCLING] Found {len(tables_list)} tables in document")
    for idx, item in enumerate(tables_list):
        pno = _page_no_from_item(item)
        # Debug: show table structure
        data = getattr(item, "data", None)
        if data:
            cells = getattr(data, "table_cells", []) or []
            print(f"[DOCLING] Table {idx+1} on page {pno}: {len(cells)} cells")
            # Show first few cells for debugging
            for i, c in enumerate(cells[:5]):
                r = getattr(c, "start_row_offset_idx", "?")
                col = getattr(c, "start_col_offset_idx", "?")
                txt = (getattr(c, "text", "") or "")[:30]
                print(f"  Cell {i}: row={r}, col={col}, text='{txt}'")
            if len(cells) > 5:
                print(f"  ... and {len(cells)-5} more cells")
        if pno is None:
            pno = 1
        text = _text_from_item(item)
        if text:
            if pno not in page_texts:
                page_texts[pno] = []
            page_texts[pno].append("\n[Table]\n" + text)

    if not page_texts:
        return []

    pages = []
    for pno in sorted(page_texts.keys()):
        # Combine all text for this page
        raw_text = "\n\n".join(page_texts[pno])
        # Post-process to format table-like text as proper markdown tables
        formatted_text = _postprocess_table_text(raw_text)
        pages.append({
            "page_number": pno,
            "text": formatted_text
        })
    return pages


class DoclingProcessor:
    """
    Document extraction using Docling (local).
    Supports pipeline hint: auto, standard, arabic, handwritten.
    """

    def __init__(self, pipeline_options: Optional[Dict] = None):
        if not DOCLING_AVAILABLE:
            raise RuntimeError("docling is not installed. Install with: pip install docling")
        self._pipeline_options = pipeline_options or {}
        self._converter = self._create_converter(enable_ocr=False)
        self._converter_ocr = self._create_converter(enable_ocr=True)

    def _create_converter(self, enable_ocr: bool = False) -> Any:
        """Create a DocumentConverter with optional OCR for scanned/handwritten docs."""
        if PdfPipelineOptions is None:
            # Fallback: basic converter without custom options
            return DocumentConverter()
        
        try:
            pipeline_opts = PdfPipelineOptions()
            pipeline_opts.do_ocr = enable_ocr
            pipeline_opts.do_table_structure = True
            
            # Configure OCR backend
            if enable_ocr:
                if OcrMacOptions is not None:
                    # macOS native OCR (best for Mac)
                    pipeline_opts.ocr_options = OcrMacOptions()
                elif TesseractCliOcrOptions is not None:
                    # Tesseract OCR with Arabic support
                    pipeline_opts.ocr_options = TesseractCliOcrOptions(
                        lang=["eng", "ara"]  # English + Arabic
                    )
            
            from docling.document_converter import PdfFormatOption
            return DocumentConverter(
                format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_opts)}
            )
        except Exception as e:
            logger.warning(f"Could not create custom converter: {e}. Using default.")
            return DocumentConverter()

    def extract_text_with_pages(
        self,
        document_path: str,
        pipeline_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract full text and page-level text from a document.
        Same contract:
        {"text": str, "pages": [{"page_number": int, "text": str}]}
        
        Pipeline hints:
        - auto: Try standard first, fall back to OCR if low text
        - standard: Fast extraction for typed/printed PDFs
        - arabic: Same as standard (Docling handles RTL automatically)
        - handwritten: Use OCR-enabled converter for scanned/handwritten docs
        """
        path = Path(document_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")

        hint = (pipeline_hint or "auto").lower().strip()
        
        # Select converter based on hint
        use_ocr = hint in ("handwritten",)
        converter = self._converter_ocr if use_ocr else self._converter
        
        logger.info(f"[Docling] Processing {path.name} with hint={hint}, ocr={use_ocr}")
        
        try:
            result = converter.convert(str(path))
        except Exception as e:
            logger.error(f"Docling conversion failed for {document_path}: {e}")
            raise

        doc = result.document
        if doc is None:
            return {"text": "", "pages": []}

        # Use export_to_markdown for better table formatting
        full_md = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""
        # Post-process to format table-like text as proper markdown tables
        full_text = _postprocess_table_text((full_md or "").strip())
        
        # Debug: log markdown output for tables
        if "|" in full_text:
            table_lines = [l for l in full_text.split("\n") if "|" in l]
            print(f"[DOCLING] Markdown export contains {len(table_lines)} table-formatted lines")
            # Show first few table lines
            for tl in table_lines[:5]:
                print(f"[DOCLING] Table line: {tl[:100]}...")

        pages = _build_pages_from_document(doc)
        
        def _looks_like_placeholder_only(text: str) -> bool:
            """True if text is mostly placeholder text like 'image 1', 'image one', 'figure 1'."""
            if not text or len(text.strip()) < 20:
                return True
            import re
            # Remove common placeholder patterns and see if much real content remains
            lowered = text.lower()
            placeholder_pattern = re.compile(
                r"\b(image\s*(one|two|three|\d+)|figure\s*\d+|page\s*\d+|صورة|شكل)\b",
                re.IGNORECASE
            )
            stripped = placeholder_pattern.sub("", lowered)
            # Count non-placeholder word chars
            real_chars = len([c for c in stripped if c.isalnum() or c.isspace()])
            total_word_chars = len([c for c in lowered if c.isalnum() or c.isspace()])
            if total_word_chars < 10:
                return True
            return (real_chars / total_word_chars) < 0.5
        
        combined = " ".join(p.get("text", "") or "" for p in pages)
        
        # Auto-fallback: if very little text OR placeholder-only text, retry with OCR (handwritten/scanned/Arabic image-PDFs)
        if hint == "auto" and not use_ocr and pages:
            total_chars = sum(len(p.get("text", "") or "") for p in pages)
            num_pages = len(pages)
            avg_per_page = total_chars / num_pages if num_pages else 0
            low_text = total_chars < 400
            low_per_page = num_pages > 0 and avg_per_page < 120
            placeholder_only = _looks_like_placeholder_only(combined)
            if low_text or low_per_page or placeholder_only:
                logger.info(
                    f"[Docling] Likely scanned/image-based or placeholder-only "
                    f"(total={total_chars}, avg/page={avg_per_page:.0f}, placeholder_only={placeholder_only}), retrying with OCR (Arabic+English)"
                )
                try:
                    result_ocr = self._converter_ocr.convert(str(path))
                    doc_ocr = result_ocr.document
                    if doc_ocr:
                        full_md_ocr = doc_ocr.export_to_markdown() if hasattr(doc_ocr, "export_to_markdown") else ""
                        pages_ocr = _build_pages_from_document(doc_ocr)
                        ocr_combined = " ".join(p.get("text", "") or "" for p in pages_ocr)
                        ocr_chars = len(ocr_combined)
                        ocr_better = ocr_chars > total_chars or (
                            not _looks_like_placeholder_only(ocr_combined) and _looks_like_placeholder_only(combined)
                        )
                        if ocr_better or ocr_chars > 100:
                            full_text = (full_md_ocr or "").strip()
                            pages = pages_ocr
                            hint = "auto-ocr"
                            logger.info(f"[Docling] Using OCR result ({ocr_chars} chars, Arabic+English)")
                except Exception as e:
                    logger.warning(f"[Docling] OCR fallback failed: {e}")
        
        if not pages and full_text:
            pages = [{"page_number": 1, "text": full_text}]

        # Fallback: if PDF has multiple pages but Docling returned only 1 (or few) pages of text,
        # extract text per page with PyMuPDF so every page gets its own content (important for Arabic/multipage).
        try:
            import fitz
            pdf_doc = fitz.open(str(path))
            actual_page_count = len(pdf_doc)
            pdf_doc.close()
        except Exception:
            actual_page_count = 0

        if actual_page_count > 1 and len(pages) < actual_page_count:
            logger.info(f"[Docling] PDF has {actual_page_count} pages but only {len(pages)} page(s) from Docling; extracting per-page with PyMuPDF")
            try:
                pages_pymupdf = self._extract_text_per_page_pymupdf(document_path)
                if pages_pymupdf:
                    total_pymupdf = sum(len(p.get("text", "") or "") for p in pages_pymupdf)
                    total_docling = sum(len(p.get("text", "") or "") for p in pages)
                    if total_pymupdf >= total_docling or len(pages_pymupdf) > len(pages):
                        pages = pages_pymupdf
                        full_text = "\n\n".join(p.get("text", "") or "" for p in pages)
                        hint = (hint or "auto") + "-pymupdf"
                        logger.info(f"[Docling] Using PyMuPDF per-page text ({len(pages)} pages, {total_pymupdf} chars)")
            except Exception as e:
                logger.warning(f"[Docling] PyMuPDF per-page fallback failed: {e}")

        # FINAL FALLBACK: If still no text (scanned PDF), use OCR on page images
        total_text_so_far = sum(len((p.get("text") or "").strip()) for p in pages)
        if total_text_so_far < 100 and actual_page_count > 0:
            logger.info(f"[Docling] Scanned PDF detected (only {total_text_so_far} chars). Using OCR to extract Arabic+English text from images...")
            try:
                ocr_pages = self._ocr_pdf_pages(document_path)
                if ocr_pages:
                    total_ocr = sum(len((p.get("text") or "").strip()) for p in ocr_pages)
                    if total_ocr > total_text_so_far:
                        pages = ocr_pages
                        full_text = "\n\n".join(p.get("text", "") or "" for p in pages)
                        hint = (hint or "auto") + "-ocr"
                        logger.info(f"[Docling] Using OCR text ({len(pages)} pages, {total_ocr} chars)")
            except Exception as e:
                logger.warning(f"[Docling] OCR fallback failed: {e}")

        return {
            "text": full_text,
            "pages": pages,
            "pipeline_used": hint,
        }

    def _extract_text_per_page_pymupdf(self, document_path: str) -> List[Dict[str, Any]]:
        """Extract text page-by-page using PyMuPDF. Use when Docling merges all pages into one."""
        try:
            import fitz
            doc = fitz.open(document_path)
            pages_out = []
            for idx in range(len(doc)):
                page = doc[idx]
                text = page.get_text()
                pages_out.append({"page_number": idx + 1, "text": (text or "").strip()})
            doc.close()
            return pages_out
        except Exception as e:
            logger.warning(f"PyMuPDF per-page extraction failed: {e}")
            return []

    def _ocr_pdf_pages(self, document_path: str, dpi: int = 200) -> List[Dict[str, Any]]:
        """
        OCR PDF pages using Tesseract with Arabic+English support.
        Used for scanned PDFs where text extraction returns 0 chars.
        """
        try:
            import fitz
            import pytesseract
            from PIL import Image
            import io
            
            doc = fitz.open(document_path)
            pages_out = []
            
            for idx in range(len(doc)):
                page = doc[idx]
                # Render page to image at specified DPI
                pix = page.get_pixmap(dpi=dpi)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                
                # OCR with Arabic + English
                # Try ara+eng first, fall back to eng if Arabic not available
                try:
                    text = pytesseract.image_to_string(img, lang='ara+eng')
                except pytesseract.TesseractError:
                    try:
                        text = pytesseract.image_to_string(img, lang='ara')
                    except pytesseract.TesseractError:
                        text = pytesseract.image_to_string(img, lang='eng')
                
                text = (text or "").strip()
                if text:
                    logger.info(f"[OCR] Page {idx+1}: {len(text)} chars extracted")
                pages_out.append({"page_number": idx + 1, "text": text})
            
            doc.close()
            return pages_out
            
        except ImportError as e:
            logger.warning(f"OCR dependencies not available: {e}. Install pytesseract and Pillow.")
            return []
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return []

    def extract_page_images(
        self, 
        document_path: str, 
        dpi: int = 100,
        max_size_bytes: int = 4_500_000,  # Stay under Bedrock's 5MB limit
        page_numbers: Optional[List[int]] = None,  # Specific pages to render (1-indexed)
    ) -> List[Dict]:
        """
        Render PDF pages to images (e.g. for vision fallback).
        Uses PyMuPDF. Images are compressed to stay under max_size_bytes.
        
        Args:
            page_numbers: If provided, only render these pages (1-indexed). Otherwise render all.
        """
        try:
            import fitz
            import io
            images: List[Dict] = []
            doc = fitz.open(document_path)
            
            for idx, page in enumerate(doc):
                page_num = idx + 1  # 1-indexed
                
                # If specific pages requested, skip others
                if page_numbers and page_num not in page_numbers:
                    continue
                # Start with requested DPI
                current_dpi = dpi
                image_bytes = None
                
                # Try progressively lower DPI until image fits
                for attempt_dpi in [current_dpi, 72, 50]:
                    pix = page.get_pixmap(dpi=attempt_dpi)
                    png_bytes = pix.tobytes("png")
                    
                    if len(png_bytes) <= max_size_bytes:
                        image_bytes = png_bytes
                        break
                    
                    # Try JPEG compression if PNG is too large
                    try:
                        from PIL import Image
                        img = Image.open(io.BytesIO(png_bytes))
                        if img.mode == "RGBA":
                            img = img.convert("RGB")
                        
                        # Try different JPEG qualities
                        for quality in [85, 70, 50, 30]:
                            jpeg_buffer = io.BytesIO()
                            img.save(jpeg_buffer, format="JPEG", quality=quality, optimize=True)
                            jpeg_bytes = jpeg_buffer.getvalue()
                            if len(jpeg_bytes) <= max_size_bytes:
                                image_bytes = jpeg_bytes
                                break
                        
                        if image_bytes and len(image_bytes) <= max_size_bytes:
                            break
                    except ImportError:
                        # PIL not available, skip JPEG compression
                        pass
                
                # If still too large, use smallest version we got
                if image_bytes is None:
                    pix = page.get_pixmap(dpi=50)
                    image_bytes = pix.tobytes("png")
                    if len(image_bytes) > max_size_bytes:
                        logger.warning(f"Page {idx+1} image still too large ({len(image_bytes)} bytes), skipping")
                        continue
                
                images.append({
                    "page_number": idx + 1,
                    "image_bytes": image_bytes,
                })
            
            doc.close()
            return images
        except Exception as e:
            logger.error(f"Error extracting page images: {e}")
            return []

    def extract_embedded_images(
        self,
        document_path: str,
        max_size_bytes: int = 4_500_000,
        min_content_size: int = 120,
    ) -> List[Dict]:
        """
        Extract embedded images from PDF (actual image objects, not page renders).
        Skips small images (logos, icons) so only content-sized figures/diagrams are kept.
        
        Returns:
            List of {"page_number", "image_bytes", "position_y", "width", "height"}
        """
        try:
            import fitz
            import io
            images: List[Dict] = []
            doc = fitz.open(document_path)
            
            for page_idx, page in enumerate(doc):
                page_images = page.get_images(full=True)
                
                for img_idx, img_info in enumerate(page_images):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        ext = base_image["ext"]
                        
                        # Get image position and size on page
                        img_rects = page.get_image_rects(xref)
                        position_y = img_rects[0].y0 if img_rects else img_idx * 100
                        width = img_rects[0].width if img_rects else 0
                        height = img_rects[0].height if img_rects else 0
                        
                        # Skip tiny images (icons, bullets)
                        if width < 50 or height < 50:
                            continue
                        # Skip logo-sized images: keep only content-sized figures (e.g. diagrams, charts)
                        if width < min_content_size or height < min_content_size:
                            logger.debug(f"Skipped small/logo image on page {page_idx+1}: {width:.0f}x{height:.0f} (min content size {min_content_size}px)")
                            continue
                        
                        logger.info(f"Found content-sized embedded image on page {page_idx+1}: {width:.0f}x{height:.0f}")
                        
                        # Compress if needed
                        if len(image_bytes) > max_size_bytes:
                            try:
                                from PIL import Image
                                img = Image.open(io.BytesIO(image_bytes))
                                if img.mode == "RGBA":
                                    img = img.convert("RGB")
                                
                                for quality in [85, 70, 50, 30]:
                                    buffer = io.BytesIO()
                                    img.save(buffer, format="JPEG", quality=quality, optimize=True)
                                    compressed = buffer.getvalue()
                                    if len(compressed) <= max_size_bytes:
                                        image_bytes = compressed
                                        ext = "jpeg"
                                        break
                            except ImportError:
                                logger.warning("PIL not available for image compression")
                                continue
                        
                        if len(image_bytes) > max_size_bytes:
                            logger.warning(f"Image on page {page_idx+1} too large, skipping")
                            continue
                        
                        images.append({
                            "page_number": page_idx + 1,
                            "image_bytes": image_bytes,
                            "position_y": position_y,
                            "width": width,
                            "height": height,
                            "format": ext,
                        })
                    except Exception as e:
                        logger.warning(f"Could not extract image {xref} from page {page_idx+1}: {e}")
                        continue
            
            doc.close()
            logger.info(f"Extracted {len(images)} embedded images from {document_path}")
            return images
        except Exception as e:
            logger.error(f"Error extracting embedded images: {e}")
            return []

    def get_pages_with_images(
        self,
        document_path: str,
        min_content_size: int = 120,
    ) -> List[int]:
        """
        Detect which pages in the PDF contain meaningful images (diagrams, figures, photos).
        Returns a list of 1-indexed page numbers that have content-sized images.
        
        This is used to determine which pages need multimodal (image) embedding
        in addition to text embedding for the hybrid approach.
        
        Args:
            document_path: Path to the PDF file
            min_content_size: Minimum width/height to consider an image as content (not icon/logo)
        
        Returns:
            List of page numbers (1-indexed) that contain meaningful images
        """
        pages_with_images: List[int] = []
        try:
            import fitz
            doc = fitz.open(document_path)
            
            for page_idx, page in enumerate(doc):
                page_images = page.get_images(full=True)
                has_content_image = False
                
                for img_info in page_images:
                    xref = img_info[0]
                    try:
                        # Get image position and size on page
                        img_rects = page.get_image_rects(xref)
                        if not img_rects:
                            continue
                        
                        width = img_rects[0].width
                        height = img_rects[0].height
                        
                        # Skip tiny images (icons, bullets, logos)
                        if width < 50 or height < 50:
                            continue
                        if width < min_content_size or height < min_content_size:
                            continue
                        
                        # Found a content-sized image on this page
                        has_content_image = True
                        break
                    except Exception:
                        continue
                
                if has_content_image:
                    pages_with_images.append(page_idx + 1)  # 1-indexed
            
            doc.close()
            logger.info(f"[Docling] Pages with content images: {pages_with_images or 'none'}")
            return pages_with_images
        except Exception as e:
            logger.error(f"Error detecting pages with images: {e}")
            return []

    def extract_figures_with_docling(
        self,
        document_path: str,
        pipeline_hint: Optional[str] = None,
        max_size_bytes: int = 2_000_000,
        dpi: int = 150,
        min_figure_size: int = 50,
    ) -> List[Dict]:
        """
        Extract figures/diagrams using Docling's figure detection with iterate_items().
        This detects actual figures (charts, diagrams, images) and renders just those regions,
        NOT the full page. Handles BOTTOMLEFT coordinate origin correctly.
        
        Returns figures sorted by page_number, then by figure_index (vertical position).
        Each figure has:
            - page_number: 1-indexed page
            - figure_index: 1-indexed figure number on that page (top to bottom)
            - image_bytes: PNG bytes of just the figure
            - width, height: dimensions in points
            - format: "png"
        """
        try:
            import fitz
            import io
            
            path = Path(document_path)
            if not path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            # Use Docling to detect figures via iterate_items (works with newer Docling)
            hint = (pipeline_hint or "auto").lower().strip()
            use_ocr = hint in ("handwritten",)
            converter = self._converter_ocr if use_ocr else self._converter
            
            logger.info(f"[Docling] Detecting figures in {path.name} using iterate_items()")
            result = converter.convert(str(path))
            doc_obj = result.document
            
            # Open PDF with PyMuPDF for rendering and coordinate conversion
            pdf_doc = fitz.open(document_path)
            
            # Get page heights for BOTTOMLEFT to TOPLEFT conversion
            page_heights = {}
            for page_idx in range(len(pdf_doc)):
                page_heights[page_idx + 1] = pdf_doc[page_idx].rect.height
            
            # Collect PictureItems from Docling using iterate_items
            picture_items = []
            for item, level in doc_obj.iterate_items():
                item_type = type(item).__name__
                if "Picture" in item_type or "Figure" in item_type:
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'bbox') and hasattr(prov, 'page_no'):
                                picture_items.append({
                                    "item": item,
                                    "bbox": prov.bbox,
                                    "page_no": prov.page_no
                                })
            
            if not picture_items:
                logger.info("[Docling] No figures detected via iterate_items, trying embedded images")
                pdf_doc.close()
                return self.extract_embedded_images(document_path, max_size_bytes)
            
            logger.info(f"[Docling] Found {len(picture_items)} PictureItems")
            
            # Extract each figure with proper coordinate conversion
            raw_figures = []
            
            for idx, pic_info in enumerate(picture_items):
                try:
                    bbox = pic_info["bbox"]
                    page_no = pic_info["page_no"]
                    
                    if page_no < 1 or page_no > len(pdf_doc):
                        logger.warning(f"Figure {idx} has invalid page {page_no}")
                        continue
                    
                    page = pdf_doc[page_no - 1]
                    page_height = page.rect.height
                    page_width = page.rect.width
                    
                    # Convert BOTTOMLEFT to TOPLEFT coordinates
                    # In BOTTOMLEFT: y=0 at bottom, y increases upward, bbox.t > bbox.b
                    # In TOPLEFT: y=0 at top, y increases downward
                    if hasattr(bbox, 'l'):  # Docling BoundingBox object
                        left = bbox.l
                        right = bbox.r
                        # Convert y: new_y = page_height - old_y
                        top = page_height - bbox.t
                        bottom = page_height - bbox.b
                    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        left, t, right, b = bbox[0], bbox[1], bbox[2], bbox[3]
                        top = page_height - t
                        bottom = page_height - b
                    else:
                        logger.warning(f"Figure {idx}: Unknown bbox format {bbox}")
                        continue
                    
                    # Ensure top < bottom (proper TOPLEFT orientation)
                    if top > bottom:
                        top, bottom = bottom, top
                    
                    # Add small padding
                    padding = 5
                    x0 = max(0, left - padding)
                    y0 = max(0, top - padding)
                    x1 = min(page_width, right + padding)
                    y1 = min(page_height, bottom + padding)
                    
                    # Validate dimensions
                    width = x1 - x0
                    height = y1 - y0
                    
                    if width < min_figure_size or height < min_figure_size:
                        logger.info(f"[Docling] Figure {idx} too small ({width:.0f}x{height:.0f}), skipping")
                        continue
                    
                    # Render figure region
                    clip_rect = fitz.Rect(x0, y0, x1, y1)
                    if clip_rect.is_empty or clip_rect.is_infinite:
                        logger.warning(f"Figure {idx} has invalid rect, skipping")
                        continue
                    
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                    
                    if pix.width < 20 or pix.height < 20:
                        logger.warning(f"Figure {idx} rendered too small, skipping")
                        continue
                    
                    image_bytes = pix.tobytes("png")
                    
                    # Compress if needed
                    if len(image_bytes) > max_size_bytes:
                        from PIL import Image
                        img = Image.open(io.BytesIO(image_bytes))
                        if img.mode == "RGBA":
                            img = img.convert("RGB")
                        for quality in [85, 70, 50]:
                            buffer = io.BytesIO()
                            img.save(buffer, format="JPEG", quality=quality)
                            if len(buffer.getvalue()) <= max_size_bytes:
                                image_bytes = buffer.getvalue()
                                break
                    
                    raw_figures.append({
                        "page_number": page_no,
                        "image_bytes": image_bytes,
                        "width": width,
                        "height": height,
                        "position_y": y0,  # For sorting (top to bottom)
                        "format": "png",
                        "pixel_width": pix.width,
                        "pixel_height": pix.height,
                    })
                    
                    logger.info(f"[Docling] Extracted figure from page {page_no}: {pix.width}x{pix.height}px")
                    
                except Exception as e:
                    logger.warning(f"Could not extract figure {idx}: {e}")
                    continue
            
            pdf_doc.close()
            
            if not raw_figures:
                logger.info("[Docling] No figures extracted, trying drawing detection")
                return self._extract_diagrams_from_drawings(document_path, max_size_bytes)
            
            # Sort figures by page_number, then by position_y (top to bottom)
            raw_figures.sort(key=lambda f: (f["page_number"], f["position_y"]))
            
            # Assign figure_index per page (1-indexed, in order from top to bottom)
            figures = []
            current_page = None
            page_figure_count = 0
            
            for fig in raw_figures:
                if fig["page_number"] != current_page:
                    current_page = fig["page_number"]
                    page_figure_count = 0
                
                page_figure_count += 1
                
                figures.append({
                    "page_number": fig["page_number"],
                    "figure_index": page_figure_count,  # 1-indexed figure on this page
                    "image_bytes": fig["image_bytes"],
                    "width": fig["width"],
                    "height": fig["height"],
                    "pixel_width": fig["pixel_width"],
                    "pixel_height": fig["pixel_height"],
                    "format": fig["format"],
                })
            
            # Log summary
            pages_with_figures = {}
            for f in figures:
                pg = f["page_number"]
                pages_with_figures[pg] = pages_with_figures.get(pg, 0) + 1
            
            logger.info(f"[Docling] Extracted {len(figures)} figures from {len(pages_with_figures)} pages")
            for pg, count in sorted(pages_with_figures.items()):
                logger.info(f"  Page {pg}: {count} figure(s)")
            
            return figures
            
        except Exception as e:
            logger.error(f"Error extracting figures with Docling: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to drawing detection
            return self._extract_diagrams_from_drawings(document_path, max_size_bytes)

    def _extract_diagrams_from_drawings(
        self,
        document_path: str,
        max_size_bytes: int = 2_000_000,
    ) -> List[Dict]:
        """
        Fallback: detect diagram regions by finding areas with drawings/graphics.
        This works for PDFs where diagrams are vector graphics, not embedded images.
        """
        try:
            import fitz
            import io
            
            pdf_doc = fitz.open(document_path)
            diagrams = []
            
            for page_idx, page in enumerate(pdf_doc):
                try:
                    # Get all drawings on the page
                    drawings = page.get_drawings()
                    
                    if not drawings:
                        continue
                    
                    # Find bounding box of all drawings (diagram region)
                    all_rects = [d.get("rect") for d in drawings if d.get("rect")]
                    if not all_rects:
                        continue
                    
                    # Cluster nearby drawings to find distinct diagram regions
                    # For simplicity, we'll find the overall bounding box of dense drawing areas
                    x0 = min(r.x0 for r in all_rects)
                    y0 = min(r.y0 for r in all_rects)
                    x1 = max(r.x1 for r in all_rects)
                    y1 = max(r.y1 for r in all_rects)
                    
                    # Check if this is a significant diagram (not just lines/borders)
                    width = x1 - x0
                    height = y1 - y0
                    page_width = page.rect.width
                    page_height = page.rect.height
                    
                    # Skip if too small or covers almost entire page
                    if width < 100 or height < 100:
                        continue
                    if width > page_width * 0.95 and height > page_height * 0.95:
                        continue
                    
                    # Add padding
                    padding = 15
                    x0 = max(0, x0 - padding)
                    y0 = max(0, y0 - padding)
                    x1 = min(page_width, x1 + padding)
                    y1 = min(page_height, y1 + padding)
                    
                    clip_rect = fitz.Rect(x0, y0, x1, y1)
                    
                    # Render the diagram region
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                    image_bytes = pix.tobytes("png")
                    
                    # Compress if needed
                    if len(image_bytes) > max_size_bytes:
                        from PIL import Image
                        img = Image.open(io.BytesIO(image_bytes))
                        if img.mode == "RGBA":
                            img = img.convert("RGB")
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=70)
                        image_bytes = buffer.getvalue()
                    
                    diagrams.append({
                        "page_number": page_idx + 1,
                        "image_bytes": image_bytes,
                        "caption": "",
                        "width": width,
                        "height": height,
                        "position_y": y0,
                        "format": "png",
                    })
                    logger.info(f"[Drawings] Extracted diagram from page {page_idx+1}: {width:.0f}x{height:.0f}")
                    
                except Exception as e:
                    logger.warning(f"Could not extract drawings from page {page_idx+1}: {e}")
                    continue
            
            pdf_doc.close()
            
            if diagrams:
                logger.info(f"[Drawings] Extracted {len(diagrams)} diagrams from drawings")
                return diagrams
            
            # Final fallback to embedded images
            logger.info("[Drawings] No diagrams from drawings, falling back to embedded images")
            return self.extract_embedded_images(document_path, max_size_bytes)
            
        except Exception as e:
            logger.error(f"Error extracting diagrams from drawings: {e}")
            return self.extract_embedded_images(document_path, max_size_bytes)
