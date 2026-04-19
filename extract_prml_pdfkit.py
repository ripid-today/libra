#!/usr/bin/env python3
"""
PRML Textbook Extraction Script using PDF-Extract-Kit
Extracts chapters from PRML PDF and saves as Markdown
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path

# PDF-Extract-Kit paths
PDFKIT_DIR = Path("C:/Users/uyenl/PDF-Extract-Kit")
VENV_PYTHON = PDFKIT_DIR / "venv/Scripts/python.exe"
EXTRACT_BRIDGE = PDFKIT_DIR / "extract_bridge.py"
MODELS_DIR = PDFKIT_DIR / "models"

# Chapter definitions
CHAPTERS = [
    {"name": "01 - Introduction", "start": 1, "end": 66},
    {"name": "02 - Probability Distributions", "start": 67, "end": 136},
    {"name": "03 - Linear Models for Regression", "start": 137, "end": 178},
    {"name": "04 - Linear Models for Classification", "start": 179, "end": 224},
    {"name": "05 - Neural Networks", "start": 225, "end": 290},
    {"name": "06 - Kernel Methods", "start": 291, "end": 324},
    {"name": "07 - Sparse Kernel Machines", "start": 325, "end": 358},
    {"name": "08 - Graphical Models", "start": 359, "end": 422},
    {"name": "09 - Mixture Models and EM", "start": 423, "end": 460},
    {"name": "10 - Approximate Inference", "start": 461, "end": 522},
    {"name": "11 - Sampling Methods", "start": 523, "end": 558},
    {"name": "12 - Continuous Latent Variables", "start": 559, "end": 604},
    {"name": "13 - Sequential Data", "start": 605, "end": 652},
    {"name": "14 - Combining Models", "start": 653, "end": 676},
]


def extract_pages_to_pdf(input_pdf: str, start_page: int, end_page: int, output_pdf: str):
    """Extract specific pages from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(input_pdf)
        new_doc = fitz.open()
        
        # Pages are 0-indexed in PyMuPDF
        for page_num in range(start_page - 1, min(end_page, len(doc))):
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        
        new_doc.save(output_pdf)
        new_doc.close()
        doc.close()
        
        return True
    except Exception as e:
        print(f"Error extracting pages {start_page}-{end_page}: {e}")
        return False


def extract_chapter_with_pdfkit(chapter_pdf: str, output_md: str):
    """Extract content using PDF-Extract-Kit bridge script."""
    import subprocess
    
    cmd = [
        str(VENV_PYTHON),
        str(EXTRACT_BRIDGE),
        chapter_pdf,
        "-o", output_md,
        "--models", str(MODELS_DIR)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per chapter
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"Extraction error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Extraction timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def extract_chapter_simple(chapter_pdf: str, output_md: str):
    """Simple extraction using just PyMuPDF text extraction."""
    try:
        import fitz
        
        doc = fitz.open(chapter_pdf)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_content.append(f"\n--- Page {page_num + 1} ---\n")
            text_content.append(text)
        
        doc.close()
        
        # Write to markdown
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(f"# {Path(output_md).stem}\n\n")
            f.write("".join(text_content))
        
        return True
    except Exception as e:
        print(f"Simple extraction error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Extract PRML textbook chapters using PDF-Extract-Kit'
    )
    parser.add_argument(
        'input_pdf',
        help='Path to PRML PDF file',
        default="C:/Users/uyenl/Downloads/Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for chapters',
        default="C:/Users/uyenl/knowledge_base/Pattern Recognition and Machine Learning"
    )
    parser.add_argument(
        '-c', '--chapters',
        help='Chapter numbers to extract (e.g., 1,3,5 or 1-5)',
        default='all'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple PyMuPDF extraction (faster, no OCR/formulas)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=30,
        help='Number of pages per batch (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_pdf):
        print(f"Error: Input PDF not found: {args.input_pdf}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix='prml_extract_'))
    
    try:
        # Determine which chapters to extract
        if args.chapters == 'all':
            chapters_to_extract = CHAPTERS
        else:
            # Parse chapter selection
            selected = []
            for part in args.chapters.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected.extend(range(start, end + 1))
                else:
                    selected.append(int(part))
            chapters_to_extract = [c for c in CHAPTERS if int(c['name'][:2]) in selected]
        
        print("=" * 60)
        print("PRML Textbook Extraction")
        print("=" * 60)
        print(f"Input: {args.input_pdf}")
        print(f"Output: {output_dir}")
        print(f"Chapters: {len(chapters_to_extract)}")
        print(f"Method: {'Simple (PyMuPDF)' if args.simple else 'PDF-Extract-Kit'}")
        print("=" * 60)
        print()
        
        # Extract each chapter
        for i, chapter in enumerate(chapters_to_extract, 1):
            chapter_name = chapter['name']
            start_page = chapter['start']
            end_page = chapter['end']
            
            print(f"[{i}/{len(chapters_to_extract)}] Extracting: {chapter_name}")
            print(f"    Pages: {start_page}-{end_page}")
            
            # Create chapter PDF
            chapter_pdf = temp_dir / f"chapter_{i:02d}.pdf"
            output_md = output_dir / f"{chapter_name}.md"
            
            # Skip if already exists and has content
            if output_md.exists() and output_md.stat().st_size > 1000:
                print(f"    ✓ Already exists, skipping")
                continue
            
            # Extract pages from main PDF
            print(f"    Extracting pages...", end=' ')
            if extract_pages_to_pdf(args.input_pdf, start_page, end_page, str(chapter_pdf)):
                print("✓")
            else:
                print("✗ Failed")
                continue
            
            # Extract content
            print(f"    Converting to Markdown...", end=' ')
            if args.simple:
                success = extract_chapter_simple(str(chapter_pdf), str(output_md))
            else:
                success = extract_chapter_with_pdfkit(str(chapter_pdf), str(output_md))
            
            if success:
                size = output_md.stat().st_size / 1024
                print(f"✓ ({size:.1f} KB)")
            else:
                print("✗ Failed")
            
            print()
        
        print("=" * 60)
        print("Extraction Complete!")
        print("=" * 60)
        
        # List output files
        print("\nGenerated files:")
        for md_file in sorted(output_dir.glob('*.md')):
            size = md_file.stat().st_size / 1024
            print(f"  {md_file.name:<45} {size:>8.1f} KB")
        
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nTemp files cleaned up")


if __name__ == '__main__':
    main()
