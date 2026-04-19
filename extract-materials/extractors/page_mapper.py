"""
Simple page utilities using digital page numbers directly.
"""
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium


def get_total_pages(pdf_path: Path) -> int:
    """
    Get total number of pages in PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        int: Total page count
    """
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        count = len(doc)
        doc.close()
        return count
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF page count: {e}")


def resolve_page_range(
    digital_start: Optional[int],
    digital_end: Optional[int],
    total_pages: int
) -> tuple:
    """
    Convert digital page range to 0-based physical indices.
    
    Args:
        digital_start: Requested start page (1-based digital page number) or None
        digital_end: Requested end page (1-based digital page number) or None
        total_pages: Total number of pages in PDF
        
    Returns:
        tuple: (physical_start, physical_end) - 0-based, INCLUSIVE
    """
    # Convert to 0-based indices
    start = (digital_start - 1) if digital_start else 0
    end = (digital_end - 1) if digital_end else (total_pages - 1)
    
    # Clamp to valid range
    start = max(0, min(start, total_pages - 1))
    end = max(0, min(end, total_pages - 1))
    
    # Ensure start <= end
    if start > end:
        start, end = end, start
    
    return (start, end)
