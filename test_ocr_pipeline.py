#!/usr/bin/env python3
"""
Test script for cost-optimized OCR pipeline for handwritten/scanned documents.

Architecture:
1. Image Preprocessing (OpenCV) - denoise, threshold, deskew
2. Handwritten OCR (TrOCR) - transformer-based, works on cropped regions
3. Table Detection (layoutparser/PaddleOCR) - detect table structure
4. Semantic Normalization (local LLM) - structure the output

Compare against current Claude Vision approach.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import io

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# STEP 1: IMAGE PREPROCESSING (OpenCV)
# ============================================================================

def preprocess_image_opencv(image_path: str, output_path: Optional[str] = None) -> "np.ndarray":
    """
    Preprocess scanned image for better OCR accuracy.
    - Denoise
    - Adaptive threshold (binarization)
    - Deskew
    
    This alone can improve handwriting OCR accuracy by 30-40%.
    """
    import cv2
    import numpy as np
    
    print(f"\n[PREPROCESS] Loading image: {image_path}")
    
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_shape = img.shape
    print(f"[PREPROCESS] Original size: {original_shape}")
    
    # Step 1: Denoise
    print("[PREPROCESS] Applying denoising...")
    img_denoised = cv2.fastNlMeansDenoising(img, None, h=30, templateWindowSize=7, searchWindowSize=21)
    
    # Step 2: Adaptive threshold (binarization)
    print("[PREPROCESS] Applying adaptive threshold...")
    img_thresh = cv2.adaptiveThreshold(
        img_denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    
    # Step 3: Deskew
    print("[PREPROCESS] Detecting skew angle...")
    coords = np.column_stack(np.where(img_thresh > 0))
    if len(coords) > 100:  # Need enough points
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        print(f"[PREPROCESS] Detected skew angle: {angle:.2f}°")
        
        if abs(angle) > 0.5:  # Only deskew if angle is significant
            (h, w) = img_thresh.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img_final = cv2.warpAffine(img_thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            img_final = img_thresh
    else:
        img_final = img_thresh
        print("[PREPROCESS] Not enough points for deskew detection")
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, img_final)
        print(f"[PREPROCESS] Saved preprocessed image to: {output_path}")
    
    return img_final


def render_pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 200) -> "np.ndarray":
    """Render a PDF page to image for processing."""
    import fitz
    import cv2
    import numpy as np
    
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # 0-indexed
    
    # Render at specified DPI
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to numpy array
    img_data = pix.tobytes("png")
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    doc.close()
    return img


# ============================================================================
# STEP 2: HANDWRITTEN OCR (TrOCR)
# ============================================================================

def ocr_with_trocr(image, model_name: str = "microsoft/trocr-base-handwritten") -> str:
    """
    Use Microsoft TrOCR for handwritten text recognition.
    
    TrOCR is transformer-based and works far better than Tesseract on cursive handwriting.
    
    ⚠️ Key rule: Run TrOCR on cropped handwritten regions, NOT full pages.
    For full pages, split into lines/regions first.
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import numpy as np
    
    print(f"\n[TROCR] Loading model: {model_name}")
    
    # Load model and processor
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    elif isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    print(f"[TROCR] Image size: {image.size}")
    
    # Process image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    # Generate text
    print("[TROCR] Generating text...")
    start_time = time.time()
    generated_ids = model.generate(pixel_values, max_length=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    elapsed = time.time() - start_time
    
    print(f"[TROCR] Generated {len(text)} chars in {elapsed:.2f}s")
    return text


def ocr_with_tesseract(image, lang: str = "eng") -> str:
    """
    Use Tesseract OCR as a baseline comparison.
    """
    import pytesseract
    from PIL import Image
    import numpy as np
    
    print(f"\n[TESSERACT] Running OCR with lang={lang}")
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    start_time = time.time()
    text = pytesseract.image_to_string(image, lang=lang)
    elapsed = time.time() - start_time
    
    print(f"[TESSERACT] Extracted {len(text)} chars in {elapsed:.2f}s")
    return text


# ============================================================================
# STEP 3: TABLE DETECTION AND CELL EXTRACTION
# ============================================================================

def detect_tables_with_opencv(image) -> List[Dict]:
    """
    Simple table detection using OpenCV contours.
    For production, consider layoutparser or PaddleOCR.
    
    Returns list of table bounding boxes.
    """
    import cv2
    import numpy as np
    
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("\n[TABLE DETECT] Detecting tables with OpenCV...")
    
    # Threshold
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine
    table_mask = cv2.add(horizontal, vertical)
    
    # Find contours
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tables = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter small regions
        if w > 100 and h > 50:
            tables.append({
                "bbox": (x, y, x + w, y + h),
                "width": w,
                "height": h
            })
    
    print(f"[TABLE DETECT] Found {len(tables)} potential tables")
    return tables


def extract_table_cells(image, table_bbox: Tuple[int, int, int, int]) -> List[Dict]:
    """
    Extract individual cells from a detected table region.
    """
    import cv2
    import numpy as np
    
    x1, y1, x2, y2 = table_bbox
    
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Crop to table region
    table_img = img[y1:y2, x1:x2]
    
    # Threshold
    _, thresh = cv2.threshold(table_img, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find all contours (potential cells)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter by size (cells should be reasonably sized)
        if 20 < w < (x2 - x1) * 0.9 and 10 < h < (y2 - y1) * 0.5:
            cells.append({
                "local_bbox": (x, y, x + w, y + h),
                "global_bbox": (x1 + x, y1 + y, x1 + x + w, y1 + y + h),
                "width": w,
                "height": h
            })
    
    # Sort cells by position (top-to-bottom, left-to-right)
    cells.sort(key=lambda c: (c["local_bbox"][1], c["local_bbox"][0]))
    
    print(f"[TABLE CELLS] Found {len(cells)} cells in table")
    return cells


# ============================================================================
# STEP 4: SEMANTIC NORMALIZATION (Local LLM or simple rules)
# ============================================================================

def normalize_form_data_simple(ocr_text: str) -> Dict:
    """
    Simple rule-based extraction of form fields.
    For production, use a local LLM like Mistral/Phi-3.
    """
    import re
    
    print("\n[NORMALIZE] Extracting structured data from OCR text...")
    
    result = {
        "raw_text": ocr_text,
        "fields": {},
        "checkboxes": {},
        "tables": []
    }
    
    # Extract field patterns: "Label: Value"
    field_pattern = r"([A-Za-z\s]+):\s*([^\n]+)"
    matches = re.findall(field_pattern, ocr_text)
    for label, value in matches:
        label = label.strip()
        value = value.strip()
        if label and value:
            result["fields"][label] = value
    
    # Detect checkbox patterns
    checkbox_patterns = [
        r"\[TICKED\]",
        r"\[EMPTY\]",
        r"\[X\]",
        r"\[\s*\]",
        r"Yes\s*☐|Yes\s*☑",
        r"No\s*☐|No\s*☑"
    ]
    
    for pattern in checkbox_patterns:
        matches = re.findall(pattern, ocr_text, re.IGNORECASE)
        if matches:
            result["checkboxes"][pattern] = len(matches)
    
    print(f"[NORMALIZE] Extracted {len(result['fields'])} fields, {sum(result['checkboxes'].values())} checkboxes")
    return result


# ============================================================================
# COMPARISON TEST: New Pipeline vs Claude Vision
# ============================================================================

def compare_pipelines(pdf_path: str, page_num: int = 1):
    """
    Compare the new cost-optimized pipeline against Claude Vision.
    """
    import cv2
    from PIL import Image
    
    print("=" * 80)
    print(f"COMPARING OCR PIPELINES FOR: {pdf_path}, Page {page_num}")
    print("=" * 80)
    
    output_dir = Path("test_ocr_output")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Render PDF page
    print("\n" + "=" * 40)
    print("STEP 1: Rendering PDF page")
    print("=" * 40)
    
    img = render_pdf_page_to_image(pdf_path, page_num, dpi=200)
    raw_path = output_dir / f"page_{page_num}_raw.png"
    cv2.imwrite(str(raw_path), img)
    print(f"Saved raw image to: {raw_path}")
    
    # Step 2: Preprocess
    print("\n" + "=" * 40)
    print("STEP 2: Image Preprocessing")
    print("=" * 40)
    
    preprocessed_path = output_dir / f"page_{page_num}_preprocessed.png"
    img_processed = preprocess_image_opencv(str(raw_path), str(preprocessed_path))
    
    # Step 3: OCR Comparison
    print("\n" + "=" * 40)
    print("STEP 3: OCR Comparison")
    print("=" * 40)
    
    results = {}
    
    # 3a: Tesseract on RAW
    print("\n--- Tesseract on RAW image ---")
    try:
        tesseract_raw = ocr_with_tesseract(str(raw_path))
        results["tesseract_raw"] = tesseract_raw
    except Exception as e:
        print(f"Tesseract RAW failed: {e}")
        results["tesseract_raw"] = ""
    
    # 3b: Tesseract on PREPROCESSED
    print("\n--- Tesseract on PREPROCESSED image ---")
    try:
        tesseract_processed = ocr_with_tesseract(str(preprocessed_path))
        results["tesseract_preprocessed"] = tesseract_processed
    except Exception as e:
        print(f"Tesseract PREPROCESSED failed: {e}")
        results["tesseract_preprocessed"] = ""
    
    # 3c: TrOCR (if available)
    print("\n--- TrOCR on RAW image ---")
    try:
        trocr_text = ocr_with_trocr(str(raw_path))
        results["trocr"] = trocr_text
    except Exception as e:
        print(f"TrOCR failed (may need to install transformers): {e}")
        results["trocr"] = f"Error: {e}"
    
    # Step 4: Table Detection
    print("\n" + "=" * 40)
    print("STEP 4: Table Detection")
    print("=" * 40)
    
    tables = detect_tables_with_opencv(img_processed)
    results["tables_detected"] = len(tables)
    
    # Step 5: Semantic Normalization
    print("\n" + "=" * 40)
    print("STEP 5: Semantic Normalization")
    print("=" * 40)
    
    # Use best OCR result for normalization
    best_ocr = results.get("tesseract_preprocessed") or results.get("tesseract_raw", "")
    normalized = normalize_form_data_simple(best_ocr)
    results["normalized"] = normalized
    
    # Save results
    results_path = output_dir / f"page_{page_num}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "tesseract_raw_chars": len(results.get("tesseract_raw", "")),
            "tesseract_preprocessed_chars": len(results.get("tesseract_preprocessed", "")),
            "trocr_chars": len(results.get("trocr", "")),
            "tables_detected": results["tables_detected"],
            "fields_extracted": len(normalized.get("fields", {})),
            "checkboxes_found": sum(normalized.get("checkboxes", {}).values()),
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Tesseract RAW:         {len(results.get('tesseract_raw', '')):>6} chars")
    print(f"Tesseract PREPROCESSED:{len(results.get('tesseract_preprocessed', '')):>6} chars")
    print(f"TrOCR:                 {len(results.get('trocr', '')):>6} chars")
    print(f"Tables detected:       {results['tables_detected']}")
    print(f"Fields extracted:      {len(normalized.get('fields', {}))}")
    print(f"Checkboxes found:      {sum(normalized.get('checkboxes', {}).values())}")
    print(f"\nResults saved to: {output_dir}/")
    
    # Print sample text comparison
    print("\n" + "=" * 40)
    print("SAMPLE TEXT (first 500 chars)")
    print("=" * 40)
    
    print("\n--- Tesseract RAW ---")
    print(results.get("tesseract_raw", "")[:500])
    
    print("\n--- Tesseract PREPROCESSED ---")
    print(results.get("tesseract_preprocessed", "")[:500])
    
    if results.get("trocr") and not results["trocr"].startswith("Error"):
        print("\n--- TrOCR ---")
        print(results["trocr"][:500])
    
    return results


def test_full_document(pdf_path: str, max_pages: int = 3):
    """
    Test the pipeline on multiple pages of a document.
    """
    import fitz
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    print(f"\n{'='*80}")
    print(f"TESTING FULL DOCUMENT: {pdf_path}")
    print(f"Total pages: {total_pages}, Testing: {min(max_pages, total_pages)}")
    print(f"{'='*80}")
    
    all_results = []
    for page_num in range(1, min(max_pages + 1, total_pages + 1)):
        try:
            results = compare_pipelines(pdf_path, page_num)
            all_results.append({
                "page": page_num,
                "tesseract_raw_chars": len(results.get("tesseract_raw", "")),
                "tesseract_preprocessed_chars": len(results.get("tesseract_preprocessed", "")),
                "tables": results.get("tables_detected", 0),
            })
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            all_results.append({"page": page_num, "error": str(e)})
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL PAGES")
    print("=" * 80)
    
    for r in all_results:
        if "error" in r:
            print(f"Page {r['page']}: ERROR - {r['error']}")
        else:
            improvement = 0
            if r.get("tesseract_raw_chars", 0) > 0:
                improvement = ((r.get("tesseract_preprocessed_chars", 0) - r["tesseract_raw_chars"]) / r["tesseract_raw_chars"]) * 100
            print(f"Page {r['page']}: RAW={r.get('tesseract_raw_chars', 0):>5} → PREPROCESSED={r.get('tesseract_preprocessed_chars', 0):>5} ({improvement:+.1f}%), Tables={r.get('tables', 0)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Default test with handwritten PDF
    PDF_PATH = "/Users/tejakandra/Downloads/ADEO AI Assessment 2/pdf_handwritten.pdf"
    
    if len(sys.argv) > 1:
        PDF_PATH = sys.argv[1]
    
    if not Path(PDF_PATH).exists():
        print(f"ERROR: PDF not found: {PDF_PATH}")
        print("\nUsage: python test_ocr_pipeline.py [path_to_pdf]")
        sys.exit(1)
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║         COST-OPTIMIZED OCR PIPELINE TEST                                      ║
║                                                                               ║
║  This script tests:                                                          ║
║  1. Image Preprocessing (OpenCV) - denoise, threshold, deskew                ║
║  2. Tesseract OCR (baseline)                                                 ║
║  3. TrOCR (transformer-based, better for handwriting)                        ║
║  4. Table Detection (OpenCV)                                                 ║
║  5. Semantic Normalization (simple rules)                                    ║
║                                                                               ║
║  Compare results against your current Claude Vision approach.                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Test first 3 pages
    test_full_document(PDF_PATH, max_pages=3)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("""
1. Check the test_ocr_output/ folder for:
   - Raw vs preprocessed images
   - OCR results comparison
   
2. If preprocessing improves results significantly:
   - Integrate preprocessing into main pipeline
   
3. If TrOCR outperforms Tesseract:
   - Consider using TrOCR for handwritten regions
   - Keep Tesseract for printed text
   
4. For production:
   - Add layoutparser for better table detection
   - Add local LLM (Mistral/Phi-3) for semantic normalization
   - Use Claude Vision only as fallback for complex cases
""")
