#!/usr/bin/env python3
"""
Test script V2: Extract ONLY diagrams/figures from PDF pages
using Docling's layout analysis to identify figure vs text regions.

This handles PDFs where the entire page is a single image with text overlay.

Approach:
1. Use Docling to analyze document layout
2. Identify "picture" / "figure" regions from Docling's output
3. Crop only those regions from the rendered page
4. Skip text and table regions

Usage:
    python test_figure_extraction_v2.py /path/to/your.pdf
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

# Create output directory
OUTPUT_DIR = Path("output_figure_test_v2")
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_figures_with_docling(pdf_path: str, dpi: int = 150) -> List[Dict]:
    """
    Use Docling to identify figure/picture regions and extract them.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    import fitz
    from PIL import Image
    
    print(f"\n{'='*60}")
    print(f"Extracting figures using Docling layout analysis")
    print(f"PDF: {pdf_path}")
    print(f"{'='*60}\n")
    
    # Configure Docling for layout analysis
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # We just need layout, not OCR
    pipeline_options.do_table_structure = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # Convert document
    print("Running Docling layout analysis...")
    result = converter.convert(pdf_path)
    doc = result.document
    
    # Open PDF with fitz for rendering
    fitz_doc = fitz.open(pdf_path)
    
    figures_extracted = []
    
    # Iterate through document items to find pictures/figures
    print("\nAnalyzing document structure...")
    
    # Method 1: Look for PictureItem in Docling output
    picture_count = 0
    table_count = 0
    text_count = 0
    
    for item, level in doc.iterate_items():
        item_type = type(item).__name__
        
        if "Picture" in item_type or "Figure" in item_type:
            picture_count += 1
            print(f"  Found Picture/Figure: {item_type}")
            
            # Get bounding box if available
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    if hasattr(prov, 'bbox') and hasattr(prov, 'page_no'):
                        bbox = prov.bbox
                        page_no = prov.page_no
                        
                        print(f"    Page {page_no}, BBox: {bbox}")
                        
                        # Extract this region
                        if page_no <= len(fitz_doc):
                            page = fitz_doc[page_no - 1]
                            
                            # Render page
                            mat = fitz.Matrix(dpi / 72, dpi / 72)
                            pix = page.get_pixmap(matrix=mat)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            
                            # Scale bbox to pixel coordinates
                            scale = dpi / 72
                            # Docling bbox format varies, handle different cases
                            if hasattr(bbox, 'l'):  # left, top, right, bottom
                                crop_box = (
                                    int(bbox.l * scale),
                                    int(bbox.t * scale),
                                    int(bbox.r * scale),
                                    int(bbox.b * scale)
                                )
                            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                crop_box = (
                                    int(bbox[0] * scale),
                                    int(bbox[1] * scale),
                                    int(bbox[2] * scale),
                                    int(bbox[3] * scale)
                                )
                            else:
                                print(f"    Unknown bbox format: {bbox}")
                                continue
                            
                            # Ensure valid crop box
                            crop_box = (
                                max(0, crop_box[0]),
                                max(0, crop_box[1]),
                                min(img.width, crop_box[2]),
                                min(img.height, crop_box[3])
                            )
                            
                            if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                                fig_img = img.crop(crop_box)
                                
                                # Save
                                fig_path = OUTPUT_DIR / f"page_{page_no}_figure_{picture_count}.png"
                                fig_img.save(fig_path)
                                print(f"    Saved: {fig_path} ({fig_img.width}x{fig_img.height}px)")
                                
                                figures_extracted.append({
                                    "page": page_no,
                                    "type": item_type,
                                    "bbox": crop_box,
                                    "size": (fig_img.width, fig_img.height),
                                    "path": str(fig_path)
                                })
        
        elif "Table" in item_type:
            table_count += 1
        elif "Text" in item_type or "Paragraph" in item_type:
            text_count += 1
    
    print(f"\nDocling found: {picture_count} pictures, {table_count} tables, {text_count} text blocks")
    
    # Method 2: If no pictures found via Docling, try embedded image extraction
    if not figures_extracted:
        print("\nNo pictures detected by Docling. Trying embedded image extraction...")
        
        for page_idx in range(len(fitz_doc)):
            page = fitz_doc[page_idx]
            page_num = page_idx + 1
            
            images = page.get_images(full=True)
            
            # Filter to find actual content images (not full-page background)
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height
            
            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                
                try:
                    img_rects = page.get_image_rects(xref)
                    if not img_rects:
                        continue
                    
                    rect = img_rects[0]
                    img_area = rect.width * rect.height
                    
                    # Skip if image covers >90% of page (likely background)
                    if img_area / page_area > 0.9:
                        print(f"  Page {page_num}: Skipping full-page background image")
                        continue
                    
                    # Skip tiny images
                    if rect.width < 100 or rect.height < 100:
                        continue
                    
                    print(f"  Page {page_num}: Found embedded image at {rect}")
                    
                    # Extract this specific image
                    base_image = fitz_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    from io import BytesIO
                    fig_img = Image.open(BytesIO(image_bytes))
                    
                    fig_path = OUTPUT_DIR / f"page_{page_num}_embedded_{img_idx + 1}.png"
                    fig_img.save(fig_path)
                    print(f"    Saved: {fig_path} ({fig_img.width}x{fig_img.height}px)")
                    
                    figures_extracted.append({
                        "page": page_num,
                        "type": "EmbeddedImage",
                        "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                        "size": (fig_img.width, fig_img.height),
                        "path": str(fig_path)
                    })
                    
                except Exception as e:
                    print(f"  Page {page_num}: Error extracting image {xref}: {e}")
    
    fitz_doc.close()
    
    return figures_extracted


def render_page_without_text(pdf_path: str, page_num: int, dpi: int = 150):
    """
    Alternative approach: Render page and use CV to mask text regions.
    """
    import fitz
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    
    # Render page
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Get text dict with character-level bboxes
    text_dict = page.get_text("dict", flags=11)
    
    draw = ImageDraw.Draw(img)
    scale = dpi / 72
    
    # Mask each text span
    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:  # Text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    bbox = span.get("bbox")
                    if bbox:
                        rect = (
                            int(bbox[0] * scale),
                            int(bbox[1] * scale),
                            int(bbox[2] * scale),
                            int(bbox[3] * scale)
                        )
                        # Fill with white (or could blur)
                        draw.rectangle(rect, fill=(255, 255, 255))
    
    doc.close()
    return img


def analyze_pdf_structure(pdf_path: str):
    """
    Analyze the structure of the PDF to determine best extraction strategy.
    """
    import fitz
    
    print(f"\n{'='*60}")
    print("PDF Structure Analysis")
    print(f"{'='*60}\n")
    
    doc = fitz.open(pdf_path)
    
    for page_idx in range(min(3, len(doc))):
        page = doc[page_idx]
        page_num = page_idx + 1
        
        print(f"Page {page_num}:")
        
        # Check text layer
        text = page.get_text()
        text_blocks = page.get_text("dict", flags=11).get("blocks", [])
        text_block_count = sum(1 for b in text_blocks if b.get("type") == 0)
        
        print(f"  Text layer: {len(text)} chars, {text_block_count} blocks")
        
        # Check images
        images = page.get_images(full=True)
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        
        full_page_images = 0
        content_images = 0
        
        for img_info in images:
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                if rects:
                    img_area = rects[0].width * rects[0].height
                    if img_area / page_area > 0.9:
                        full_page_images += 1
                    elif rects[0].width > 100 and rects[0].height > 100:
                        content_images += 1
            except:
                pass
        
        print(f"  Images: {len(images)} total ({full_page_images} full-page, {content_images} content)")
        
        # Determine PDF type
        if full_page_images > 0 and text_block_count == 0:
            print(f"  Type: SCANNED (full-page image, no text layer)")
        elif full_page_images > 0 and text_block_count > 0:
            print(f"  Type: IMAGE+TEXT OVERLAY (hybrid)")
        elif text_block_count > 0:
            print(f"  Type: NATIVE TEXT (standard PDF)")
        
        print()
    
    doc.close()


if __name__ == "__main__":
    # Check dependencies
    try:
        import fitz
        from PIL import Image
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pymupdf pillow docling")
        sys.exit(1)
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Try sample paths
        sample_paths = [
            "/Users/tejakandra/Downloads/ADEO AI Assessment 2/pdf_a.pdf",
            "/Users/tejakandra/Downloads/ADEO AI Assessment 2/pdf_b.pdf",
        ]
        pdf_path = next((p for p in sample_paths if os.path.exists(p)), None)
        
        if not pdf_path:
            print("Usage: python test_figure_extraction_v2.py /path/to/your.pdf")
            sys.exit(1)
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Analyze structure first
    analyze_pdf_structure(pdf_path)
    
    # Extract figures
    figures = extract_figures_with_docling(pdf_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total figures extracted: {len(figures)}")
    
    for fig in figures:
        print(f"  - Page {fig['page']}: {fig['type']} ({fig['size'][0]}x{fig['size'][1]}px)")
    
    print(f"\nOutput saved to: {OUTPUT_DIR.absolute()}")
    
    if not figures:
        print("\n⚠️  No separate figures found!")
        print("This PDF likely has full-page images with text overlay.")
        print("\nRecommended approach for this PDF type:")
        print("1. Use Docling to extract text/tables (already working)")
        print("2. For image embedding, consider:")
        print("   a) Embedding the full page (current approach)")
        print("   b) Using vision model to identify diagram regions")
        print("   c) Skipping image embedding if text extraction is sufficient")
