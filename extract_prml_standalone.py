#!/usr/bin/env python3
"""
Standalone PRML Textbook Extraction
Uses PyMuPDF (fitz) for reliable text extraction
No complex dependencies required
"""

import os
import sys
from pathlib import Path

# Add PDF-Extract-Kit to path for PyMuPDF
temp_path = sys.path.copy()
sys.path.insert(0, "C:/Users/uyenl/PDF-Extract-Kit")
sys.path.insert(0, "C:/Users/uyenl/PDF-Extract-Kit/venv/Lib/site-packages")

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF not found. Make sure PDF-Extract-Kit is installed.")
    sys.exit(1)


# Chapter definitions with page ranges
CHAPTERS = [
    {"num": 1, "name": "01 - Introduction", "start": 1, "end": 66},
    {"num": 2, "name": "02 - Probability Distributions", "start": 67, "end": 136},
    {"num": 3, "name": "03 - Linear Models for Regression", "start": 137, "end": 178},
    {"num": 4, "name": "04 - Linear Models for Classification", "start": 179, "end": 224},
    {"num": 5, "name": "05 - Neural Networks", "start": 225, "end": 290},
    {"num": 6, "name": "06 - Kernel Methods", "start": 291, "end": 324},
    {"num": 7, "name": "07 - Sparse Kernel Machines", "start": 325, "end": 358},
    {"num": 8, "name": "08 - Graphical Models", "start": 359, "end": 422},
    {"num": 9, "name": "09 - Mixture Models and EM", "start": 423, "end": 460},
    {"num": 10, "name": "10 - Approximate Inference", "start": 461, "end": 522},
    {"num": 11, "name": "11 - Sampling Methods", "start": 523, "end": 558},
    {"num": 12, "name": "12 - Continuous Latent Variables", "start": 559, "end": 604},
    {"num": 13, "name": "13 - Sequential Data", "start": 605, "end": 652},
    {"num": 14, "name": "14 - Combining Models", "start": 653, "end": 676},
]


def extract_chapter(doc, chapter, output_dir):
    """Extract a single chapter and save as markdown."""
    
    chapter_name = chapter['name']
    start_page = chapter['start'] - 1  # 0-indexed
    end_page = chapter['end']  # Exclusive
    
    output_file = Path(output_dir) / f"{chapter_name}.md"
    
    print(f"  Extracting {chapter_name} (pages {chapter['start']}-{chapter['end']})...")
    
    lines = []
    lines.append(f"# {chapter_name}\n")
    lines.append(f"*Pages {chapter['start']}-{chapter['end']} from Pattern Recognition and Machine Learning*\n")
    
    for page_num in range(start_page, min(end_page, len(doc))):
        page = doc[page_num]
        
        # Get text
        text = page.get_text()
        
        # Add page marker
        lines.append(f"\n---\n")
        lines.append(f"**Page {page_num + 1}**\n")
        lines.append(text)
        lines.append("\n")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))
    
    file_size = output_file.stat().st_size / 1024
    print(f"    [OK] Saved: {output_file.name} ({file_size:.1f} KB)")
    
    return True


def main():
    # Configuration
    pdf_path = Path("C:/Users/uyenl/Downloads/Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf")
    output_dir = Path("C:/Users/uyenl/knowledge_base/Pattern Recognition and Machine Learning")
    
    print("=" * 60)
    print("PRML TEXTBOOK EXTRACTION")
    print("Using PyMuPDF (fitz) via PDF-Extract-Kit environment")
    print("=" * 60)
    print()
    
    # Validate input
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at: {pdf_path}")
        sys.exit(1)
    
    print(f"Input PDF: {pdf_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open PDF
    print("Opening PDF...")
    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        print(f"[OK] PDF opened: {total_pages} pages")
        print()
    except Exception as e:
        print(f"ERROR: Could not open PDF: {e}")
        sys.exit(1)
    
    # Extract chapters
    print("Extracting chapters...")
    print("-" * 60)
    
    success_count = 0
    for chapter in CHAPTERS:
        try:
            if extract_chapter(doc, chapter, output_dir):
                success_count += 1
        except Exception as e:
            print(f"    [ERROR] Failed to extract {chapter['name']}: {e}")
    
    # Close PDF
    doc.close()
    
    # Summary
    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Successfully extracted: {success_count}/{len(CHAPTERS)} chapters")
    print()
    print("Output files:")
    
    for md_file in sorted(output_dir.glob("*.md")):
        size = md_file.stat().st_size / 1024
        print(f"  {md_file.name:<45} {size:>8.1f} KB")
    
    print()
    print(f"All chapters saved to: {output_dir}")


if __name__ == "__main__":
    main()
