"""
Core marker library integration for document extraction.
Simplified version: No OCR, No LLM.
"""
import os
from pathlib import Path
from typing import Optional, Tuple

from marker.models import create_model_dict
from marker.converters.pdf import PdfConverter

from config import MARKER_CONFIG

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def extract_document(
    pdf_path: Path,
    page_range: Optional[Tuple[int, int]] = None,
) -> 'MarkdownOutput':
    """
    Extract content from PDF using marker library.
    OCR is disabled by default for faster text extraction.
    
    Args:
        pdf_path: Path to PDF file
        page_range: (start_idx, end_idx) 0-based inclusive, or None for all pages
        
    Returns:
        MarkdownOutput: Object with markdown content
    """
    # Load models with default CPU config
    artifact_dict = create_model_dict(device='cpu')
    
    # Build configuration - OCR disabled
    config = MARKER_CONFIG.copy()
    config['force_ocr'] = False
    
    # Add page range if specified
    if page_range is not None:
        start, end = page_range
        config['page_range'] = list(range(start, end + 1))
    
    # Create converter and extract
    converter = PdfConverter(
        config=config,
        artifact_dict=artifact_dict
    )
    
    rendered = converter(str(pdf_path))
    
    return rendered
