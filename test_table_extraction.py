#!/usr/bin/env python3
"""Test script to check Docling table extraction."""

import sys
sys.path.insert(0, "/Users/tejakandra/Downloads/AI-project-app/RAG_AI_agent/src")

from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Path to test PDF
PDF_PATH = "/Users/tejakandra/Downloads/ADEO AI Assessment 2/pdf_a.pdf"

def main():
    print(f"Testing table extraction from: {PDF_PATH}")
    print("=" * 80)
    
    # Configure pipeline
    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = False
    pipeline_opts.do_table_structure = True
    
    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )
    
    # Convert document
    result = converter.convert(PDF_PATH)
    doc = result.document
    
    # Check for tables
    tables = getattr(doc, "tables", []) or []
    print(f"\n[INFO] Found {len(tables)} tables in document")
    
    for idx, table in enumerate(tables):
        print(f"\n{'='*80}")
        print(f"TABLE {idx + 1}")
        print("=" * 80)
        
        # Get page number
        prov = getattr(table, "prov", []) or []
        page_no = None
        for p in prov:
            page_no = getattr(p, "page_no", None)
            if page_no:
                break
        print(f"Page: {page_no}")
        
        # Check table data
        data = getattr(table, "data", None)
        if data:
            # Check for table_cells
            cells = getattr(data, "table_cells", []) or []
            print(f"Number of cells: {len(cells)}")
            
            if cells:
                # Build grid
                grid = {}
                max_row = 0
                max_col = 0
                for c in cells:
                    r = getattr(c, "start_row_offset_idx", 0)
                    col = getattr(c, "start_col_offset_idx", 0)
                    txt = getattr(c, "text", "") or ""
                    if r not in grid:
                        grid[r] = {}
                    grid[r][col] = txt.strip()[:50]  # Truncate for display
                    max_row = max(max_row, r)
                    max_col = max(max_col, col)
                
                print(f"Grid size: {max_row + 1} rows x {max_col + 1} cols")
                print("\nTable content:")
                for row_idx in sorted(grid.keys()):
                    row_cells = []
                    for col_idx in range(max_col + 1):
                        cell_text = grid[row_idx].get(col_idx, "")
                        row_cells.append(cell_text[:30].ljust(30))
                    print("| " + " | ".join(row_cells) + " |")
                    if row_idx == 0:
                        print("| " + " | ".join(["-" * 30] * (max_col + 1)) + " |")
            else:
                print("No table_cells found!")
                
            # Check for grid attribute
            grid_attr = getattr(data, "grid", None)
            if grid_attr:
                print(f"\nGrid attribute exists with {len(grid_attr)} rows")
                
            # Try to_dataframe
            try:
                df_method = getattr(data, "to_dataframe", None)
                if callable(df_method):
                    df = df_method()
                    print(f"\nDataFrame extraction works: {len(df)} rows, {len(df.columns)} columns")
                    print("Columns:", list(df.columns))
                    print(df.to_markdown(index=False))
            except Exception as e:
                print(f"DataFrame extraction failed: {e}")
        else:
            print("No data attribute on table!")
            
        # Check table.text attribute
        table_text = getattr(table, "text", None)
        if table_text:
            print(f"\nTable text attribute (first 500 chars):\n{table_text[:500]}")
    
    # Also check markdown export for tables
    print("\n" + "=" * 80)
    print("MARKDOWN EXPORT (checking for table formatting)")
    print("=" * 80)
    
    md = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""
    
    # Find lines with | that indicate tables
    lines = md.split("\n")
    in_table = False
    table_lines = []
    for line in lines:
        if "|" in line:
            in_table = True
            table_lines.append(line)
        elif in_table and line.strip() == "":
            # End of table
            if table_lines:
                print("\nFound table in markdown:")
                for tl in table_lines[:15]:  # Show first 15 lines
                    print(tl)
                if len(table_lines) > 15:
                    print(f"... ({len(table_lines) - 15} more lines)")
            table_lines = []
            in_table = False

if __name__ == "__main__":
    main()
