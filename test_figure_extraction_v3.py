#!/usr/bin/env python3
"""
Test script V3: Extract ONLY diagrams/figures from PDF pages
using Docling's layout analysis with correct coordinate handling.

This handles PDFs with BOTTOMLEFT coordinate origin.

Usage:
    python test_figure_extraction_v3.py /path/to/your.pdf
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO

# Create output directory
OUTPUT_DIR = Path("output_figure_test_v3")
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_figures_with_docling(pdf_path: str, dpi: int = 150) -> List[Dict]:
    """
    Use Docling to identify figure/picture regions and extract them.
    Handles BOTTOMLEFT coordinate origin correctly.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    import fitz
    from PIL import Image
    
    print(f"\n{'='*60}")
    print(f"Extracting figures using Docling")
    print(f"PDF: {pdf_path}")
    print(f"{'='*60}\n")
    
    # Configure Docling
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
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
    
    # Get page heights for coordinate conversion
    page_heights = {}
    for page_idx in range(len(fitz_doc)):
        page = fitz_doc[page_idx]
        page_heights[page_idx + 1] = page.rect.height
    
    figures_extracted = []
    picture_items = []
    table_count = 0
    text_count = 0
    
    # Collect all items
    print("\nAnalyzing document structure...")
    for item, level in doc.iterate_items():
        item_type = type(item).__name__
        
        if "Picture" in item_type or "Figure" in item_type:
            picture_items.append(item)
            print(f"  Found: {item_type}")
            
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    if hasattr(prov, 'bbox') and hasattr(prov, 'page_no'):
                        bbox = prov.bbox
                        page_no = prov.page_no
                        print(f"    Page {page_no}, Raw BBox: {bbox}")
        
        elif "Table" in item_type:
            table_count += 1
        elif "Text" in item_type or "Paragraph" in item_type:
            text_count += 1
    
    print(f"\nDocling found: {len(picture_items)} pictures, {table_count} tables, {text_count} text blocks")
    
    # Extract each picture
    print("\n" + "-"*40)
    print("Extracting figures...")
    print("-"*40)
    
    for idx, item in enumerate(picture_items):
        if not hasattr(item, 'prov') or not item.prov:
            continue
        
        for prov in item.prov:
            if not hasattr(prov, 'bbox') or not hasattr(prov, 'page_no'):
                continue
            
            bbox = prov.bbox
            page_no = prov.page_no
            
            if page_no > len(fitz_doc):
                continue
            
            page = fitz_doc[page_no - 1]
            page_height = page.rect.height
            
            # Convert BOTTOMLEFT to TOPLEFT coordinates
            # In BOTTOMLEFT: y=0 is at bottom, y increases upward
            # In TOPLEFT: y=0 is at top, y increases downward
            # Also: bbox.t > bbox.b in BOTTOMLEFT because t is higher up
            
            if hasattr(bbox, 'l'):  # Docling BoundingBox object
                left = bbox.l
                right = bbox.r
                # Convert y coordinates: new_y = page_height - old_y
                top = page_height - bbox.t  # bbox.t is higher, so smaller after conversion
                bottom = page_height - bbox.b  # bbox.b is lower, so larger after conversion
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                left, t, right, b = bbox
                top = page_height - t
                bottom = page_height - b
            else:
                print(f"  Unknown bbox format: {bbox}")
                continue
            
            # Ensure top < bottom
            if top > bottom:
                top, bottom = bottom, top
            
            print(f"\n  Figure {idx + 1} (Page {page_no}):")
            print(f"    Raw: l={getattr(bbox, 'l', 'N/A')}, t={getattr(bbox, 't', 'N/A')}, r={getattr(bbox, 'r', 'N/A')}, b={getattr(bbox, 'b', 'N/A')}")
            print(f"    Converted: left={left:.1f}, top={top:.1f}, right={right:.1f}, bottom={bottom:.1f}")
            
            # Render page at DPI
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Scale coordinates to pixels
            scale = dpi / 72
            crop_box = (
                int(left * scale),
                int(top * scale),
                int(right * scale),
                int(bottom * scale)
            )
            
            # Ensure valid bounds
            crop_box = (
                max(0, crop_box[0]),
                max(0, crop_box[1]),
                min(img.width, crop_box[2]),
                min(img.height, crop_box[3])
            )
            
            print(f"    Crop box (pixels): {crop_box}")
            
            # Check if valid
            width = crop_box[2] - crop_box[0]
            height = crop_box[3] - crop_box[1]
            
            if width < 50 or height < 50:
                print(f"    ⚠️ Too small ({width}x{height}), skipping")
                continue
            
            # Crop and save
            fig_img = img.crop(crop_box)
            fig_path = OUTPUT_DIR / f"page_{page_no}_figure_{idx + 1}.png"
            fig_img.save(fig_path)
            print(f"    ✓ Saved: {fig_path} ({fig_img.width}x{fig_img.height}px)")
            
            figures_extracted.append({
                "page": page_no,
                "figure_index": idx + 1,
                "bbox_original": str(bbox),
                "bbox_converted": (left, top, right, bottom),
                "size": (fig_img.width, fig_img.height),
                "path": str(fig_path)
            })
    
    # Also save full pages for comparison
    print("\n" + "-"*40)
    print("Saving full pages for comparison...")
    print("-"*40)
    
    pages_with_figures = set(f["page"] for f in figures_extracted)
    
    for page_no in pages_with_figures:
        page = fitz_doc[page_no - 1]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        full_path = OUTPUT_DIR / f"page_{page_no}_FULL.png"
        img.save(full_path)
        print(f"  Page {page_no}: {full_path}")
    
    fitz_doc.close()
    
    return figures_extracted


def visualize_regions(pdf_path: str, dpi: int = 150):
    """
    Create a visualization showing detected regions on each page.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    import fitz
    from PIL import Image, ImageDraw
    
    print(f"\n{'='*60}")
    print("Creating visualization with bounding boxes")
    print(f"{'='*60}\n")
    
    # Configure Docling
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    result = converter.convert(pdf_path)
    doc = result.document
    
    fitz_doc = fitz.open(pdf_path)
    
    # Collect all items with their bboxes
    items_by_page = {}
    
    for item, level in doc.iterate_items():
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'bbox') and hasattr(prov, 'page_no'):
                    page_no = prov.page_no
                    if page_no not in items_by_page:
                        items_by_page[page_no] = []
                    
                    item_type = type(item).__name__
                    items_by_page[page_no].append({
                        "type": item_type,
                        "bbox": prov.bbox
                    })
    
    # Draw on each page
    for page_no, items in items_by_page.items():
        if page_no > len(fitz_doc):
            continue
        
        page = fitz_doc[page_no - 1]
        page_height = page.rect.height
        
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img)
        
        scale = dpi / 72
        
        for item in items:
            bbox = item["bbox"]
            item_type = item["type"]
            
            # Convert coordinates
            if hasattr(bbox, 'l'):
                left = bbox.l * scale
                right = bbox.r * scale
                top = (page_height - bbox.t) * scale
                bottom = (page_height - bbox.b) * scale
            else:
                continue
            
            if top > bottom:
                top, bottom = bottom, top
            
            # Color by type
            if "Picture" in item_type or "Figure" in item_type:
                color = (0, 255, 0)  # Green for pictures
                label = "PICTURE"
            elif "Table" in item_type:
                color = (0, 0, 255)  # Blue for tables
                label = "TABLE"
            elif "Text" in item_type:
                color = (255, 0, 0)  # Red for text
                label = "TEXT"
            else:
                color = (128, 128, 128)
                label = item_type[:10]
            
            # Draw rectangle
            draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)
            
            # Draw label
            draw.text((left + 5, top + 5), label, fill=color)
        
        viz_path = OUTPUT_DIR / f"page_{page_no}_VISUALIZATION.png"
        img.save(viz_path)
        print(f"  Page {page_no}: {viz_path}")
    
    fitz_doc.close()


if __name__ == "__main__":
    try:
        import fitz
        from PIL import Image
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        sample_paths = [
            "/Users/tejakandra/Downloads/ADEO AI Assessment 2/pdf_a.pdf",
            "/Users/tejakandra/Downloads/ADEO AI Assessment 2/pdf_b.pdf",
        ]
        pdf_path = next((p for p in sample_paths if os.path.exists(p)), None)
        
        if not pdf_path:
            print("Usage: python test_figure_extraction_v3.py /path/to/your.pdf")
            sys.exit(1)
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Extract figures
    figures = extract_figures_with_docling(pdf_path)
    
    # Create visualization
    visualize_regions(pdf_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total figures extracted: {len(figures)}")
    
    for fig in figures:
        print(f"  - Page {fig['page']}, Figure {fig['figure_index']}: {fig['size'][0]}x{fig['size'][1]}px")
    
    print(f"\nOutput saved to: {OUTPUT_DIR.absolute()}")
    print("\nFiles created:")
    print("  - page_X_figure_Y.png: Extracted figures ONLY (no text/tables)")
    print("  - page_X_FULL.png: Full page for comparison")
    print("  - page_X_VISUALIZATION.png: Page with colored bounding boxes")
    print("    GREEN = Pictures, BLUE = Tables, RED = Text")
