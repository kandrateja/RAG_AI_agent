#!/usr/bin/env python3
"""
Test script to extract ONLY images/figures from PDF pages,
masking out text and table regions to avoid duplication.

This is for testing the concept before integrating into the RAG pipeline.

Usage:
    python test_image_extraction.py /path/to/your.pdf

Output:
    - Saves original page images to: output/page_X_original.png
    - Saves masked page images to: output/page_X_masked.png
    - Saves extracted figures only to: output/page_X_figure_Y.png
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import io

# Create output directory
OUTPUT_DIR = Path("output_image_test")
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_text_regions(page) -> List[Tuple[float, float, float, float]]:
    """
    Extract bounding boxes of text blocks on the page.
    Returns list of (x0, y0, x1, y1) rectangles.
    """
    text_regions = []
    
    # Get text blocks (each block is a paragraph/text area)
    blocks = page.get_text("dict", flags=11)["blocks"]
    
    for block in blocks:
        if block["type"] == 0:  # Type 0 = text block
            bbox = block["bbox"]  # (x0, y0, x1, y1)
            # Add some padding around text
            padding = 5
            text_regions.append((
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                bbox[2] + padding,
                bbox[3] + padding
            ))
    
    return text_regions


def extract_table_regions(page) -> List[Tuple[float, float, float, float]]:
    """
    Detect table regions using line detection heuristics.
    Tables typically have many horizontal/vertical lines close together.
    """
    table_regions = []
    
    try:
        # Use PyMuPDF's table detection if available (v1.23+)
        tables = page.find_tables()
        for table in tables:
            bbox = table.bbox
            padding = 10
            table_regions.append((
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                bbox[2] + padding,
                bbox[3] + padding
            ))
    except AttributeError:
        # Fallback: detect tables by looking for grid-like line patterns
        # This is a simplified heuristic
        drawings = page.get_drawings()
        
        # Group lines by proximity to detect table boundaries
        h_lines = []
        v_lines = []
        
        for d in drawings:
            for item in d.get("items", []):
                if item[0] == "l":  # line
                    p1, p2 = item[1], item[2]
                    # Horizontal line
                    if abs(p1.y - p2.y) < 2:
                        h_lines.append((min(p1.x, p2.x), p1.y, max(p1.x, p2.x), p2.y))
                    # Vertical line
                    elif abs(p1.x - p2.x) < 2:
                        v_lines.append((p1.x, min(p1.y, p2.y), p2.x, max(p1.y, p2.y)))
        
        # If we have many lines, likely a table region
        if len(h_lines) > 3 and len(v_lines) > 3:
            all_lines = h_lines + v_lines
            if all_lines:
                x0 = min(l[0] for l in all_lines)
                y0 = min(l[1] for l in all_lines)
                x1 = max(l[2] for l in all_lines)
                y1 = max(l[3] for l in all_lines)
                table_regions.append((x0 - 10, y0 - 10, x1 + 10, y1 + 10))
    
    return table_regions


def extract_image_regions(page) -> List[Dict[str, Any]]:
    """
    Extract embedded image regions from the page.
    Returns list of dicts with bbox and image data.
    """
    image_regions = []
    
    images = page.get_images(full=True)
    
    for img_index, img_info in enumerate(images):
        xref = img_info[0]
        
        try:
            # Get image position on page
            img_rects = page.get_image_rects(xref)
            if not img_rects:
                continue
            
            rect = img_rects[0]
            
            # Skip tiny images (icons, bullets, logos)
            if rect.width < 50 or rect.height < 50:
                continue
            
            image_regions.append({
                "index": img_index,
                "xref": xref,
                "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                "width": rect.width,
                "height": rect.height
            })
        except Exception as e:
            print(f"  Warning: Could not get rect for image {xref}: {e}")
            continue
    
    return image_regions


def mask_regions_on_image(page, regions_to_mask: List[Tuple], dpi: int = 150):
    """
    Render page as image and mask (white-out) specified regions.
    Returns the masked image as PIL Image.
    """
    from PIL import Image, ImageDraw
    
    # Render page at specified DPI
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Scale factor for coordinates (page coords to pixel coords)
    scale = dpi / 72
    
    # Draw white rectangles over regions to mask
    draw = ImageDraw.Draw(img)
    
    for region in regions_to_mask:
        x0, y0, x1, y1 = region
        # Scale coordinates
        pixel_rect = (
            int(x0 * scale),
            int(y0 * scale),
            int(x1 * scale),
            int(y1 * scale)
        )
        # Fill with white
        draw.rectangle(pixel_rect, fill=(255, 255, 255))
    
    return img


def extract_figure_only(page, figure_bbox: Tuple, dpi: int = 150):
    """
    Extract just the figure region from a page.
    Returns PIL Image of just that region.
    """
    from PIL import Image
    
    # Render page
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Scale and crop
    scale = dpi / 72
    x0, y0, x1, y1 = figure_bbox
    crop_box = (
        int(x0 * scale),
        int(y0 * scale),
        int(x1 * scale),
        int(y1 * scale)
    )
    
    return img.crop(crop_box)


def is_region_overlapping(r1: Tuple, r2: Tuple, threshold: float = 0.5) -> bool:
    """Check if two regions overlap significantly."""
    x0_1, y0_1, x1_1, y1_1 = r1
    x0_2, y0_2, x1_2, y1_2 = r2
    
    # Calculate intersection
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)
    
    if x1_i <= x0_i or y1_i <= y0_i:
        return False
    
    intersection = (x1_i - x0_i) * (y1_i - y0_i)
    area_r2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    
    if area_r2 == 0:
        return False
    
    return (intersection / area_r2) >= threshold


def process_pdf(pdf_path: str, dpi: int = 150):
    """
    Process a PDF and extract images with text/tables masked out.
    """
    import fitz
    from PIL import Image
    
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"{'='*60}\n")
    
    doc = fitz.open(pdf_path)
    
    results = []
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        
        print(f"\n--- Page {page_num} ---")
        
        # Extract regions
        text_regions = extract_text_regions(page)
        table_regions = extract_table_regions(page)
        image_regions = extract_image_regions(page)
        
        print(f"  Text blocks found: {len(text_regions)}")
        print(f"  Table regions found: {len(table_regions)}")
        print(f"  Image/figure regions found: {len(image_regions)}")
        
        if not image_regions:
            print(f"  → No significant images on this page, skipping...")
            continue
        
        # Combine text and table regions to mask
        regions_to_mask = text_regions + table_regions
        
        # Save original page
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        original_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        original_path = OUTPUT_DIR / f"page_{page_num}_original.png"
        original_img.save(original_path)
        print(f"  Saved original: {original_path}")
        
        # Create masked version (text/tables removed)
        masked_img = mask_regions_on_image(page, regions_to_mask, dpi)
        masked_path = OUTPUT_DIR / f"page_{page_num}_masked.png"
        masked_img.save(masked_path)
        print(f"  Saved masked (text/tables removed): {masked_path}")
        
        # Extract individual figures
        for fig_idx, fig_info in enumerate(image_regions):
            fig_bbox = fig_info["bbox"]
            
            # Check if this figure overlaps significantly with text
            overlaps_text = any(
                is_region_overlapping(fig_bbox, text_region, 0.3)
                for text_region in text_regions
            )
            
            if overlaps_text:
                print(f"  Figure {fig_idx + 1}: Overlaps with text, may be inline image")
            
            # Extract just the figure
            try:
                fig_img = extract_figure_only(page, fig_bbox, dpi)
                
                # Skip if too small after extraction
                if fig_img.width < 50 or fig_img.height < 50:
                    print(f"  Figure {fig_idx + 1}: Too small ({fig_img.width}x{fig_img.height}), skipping")
                    continue
                
                fig_path = OUTPUT_DIR / f"page_{page_num}_figure_{fig_idx + 1}.png"
                fig_img.save(fig_path)
                print(f"  Saved figure {fig_idx + 1}: {fig_path} ({fig_img.width}x{fig_img.height}px)")
                
                results.append({
                    "page": page_num,
                    "figure_index": fig_idx + 1,
                    "bbox": fig_bbox,
                    "size": (fig_img.width, fig_img.height),
                    "path": str(fig_path)
                })
            except Exception as e:
                print(f"  Figure {fig_idx + 1}: Error extracting - {e}")
    
    doc.close()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total figures extracted: {len(results)}")
    for r in results:
        print(f"  - Page {r['page']}, Figure {r['figure_index']}: {r['size'][0]}x{r['size'][1]}px")
    print(f"\nOutput saved to: {OUTPUT_DIR.absolute()}")
    
    return results


def compare_approaches(pdf_path: str):
    """
    Compare the two approaches visually:
    1. Full page as image (current approach)
    2. Page with text/tables masked (new approach)
    """
    import fitz
    from PIL import Image
    
    print(f"\n{'='*60}")
    print("APPROACH COMPARISON")
    print(f"{'='*60}")
    print("\nCurrent approach: Full page → includes text + images (DUPLICATION)")
    print("New approach: Masked page → only images/figures (NO DUPLICATION)\n")
    
    doc = fitz.open(pdf_path)
    
    for page_idx in range(min(3, len(doc))):  # First 3 pages
        page = doc[page_idx]
        page_num = page_idx + 1
        
        image_regions = extract_image_regions(page)
        if not image_regions:
            continue
        
        text_regions = extract_text_regions(page)
        table_regions = extract_table_regions(page)
        
        # Calculate stats
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        
        text_area = sum((r[2]-r[0]) * (r[3]-r[1]) for r in text_regions)
        table_area = sum((r[2]-r[0]) * (r[3]-r[1]) for r in table_regions)
        image_area = sum(img["width"] * img["height"] for img in image_regions)
        
        print(f"Page {page_num}:")
        print(f"  - Text coverage: {text_area/page_area*100:.1f}%")
        print(f"  - Table coverage: {table_area/page_area*100:.1f}%")
        print(f"  - Image coverage: {image_area/page_area*100:.1f}%")
        print(f"  - Duplication avoided: {(text_area+table_area)/page_area*100:.1f}%")
        print()
    
    doc.close()


if __name__ == "__main__":
    # Check dependencies
    try:
        import fitz
        from PIL import Image, ImageDraw
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pymupdf pillow")
        sys.exit(1)
    
    # Get PDF path from command line or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        print("Usage: python test_image_extraction.py /path/to/your.pdf")
        print("\nNo PDF provided. Looking for sample PDFs...")
        
        # Try to find a sample PDF
        sample_paths = [
            "/Users/tejakandra/Downloads/ADEO AI Assessment 2/pdf_a.pdf",
            "/Users/tejakandra/Downloads/ADEO AI Assessment 2/pdf_b.pdf",
        ]
        
        pdf_path = None
        for p in sample_paths:
            if os.path.exists(p):
                pdf_path = p
                break
        
        if not pdf_path:
            print("No sample PDF found. Please provide a PDF path.")
            sys.exit(1)
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Run the test
    results = process_pdf(pdf_path)
    
    # Compare approaches
    compare_approaches(pdf_path)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nCheck the 'output_image_test' folder to compare:")
    print("  - *_original.png: Full page (current approach)")
    print("  - *_masked.png: Text/tables removed (new approach)")
    print("  - *_figure_*.png: Individual figures extracted")
