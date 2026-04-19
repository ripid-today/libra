"""
Configuration constants and settings for extract-materials.
"""
import os
from pathlib import Path

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_NO_INPUT = "Please input a textbook or academic material"
ERROR_NO_OUTPUT_PATH = "Please indicate where I can extract the material"
ERROR_INVALID_FORMAT = "Please retry your materials with the valid format."

# =============================================================================
# SUPPORTED FORMATS
# =============================================================================

SUPPORTED_OFFICE_FORMATS = ['.docx', '.doc', '.pptx', '.ppt', '.odt', '.ods', '.odp', '.rtf']
SUPPORTED_EBOOK_FORMATS = ['.epub', '.mobi', '.azw', '.azw3']
SUPPORTED_INPUT_FORMATS = ['.pdf'] + SUPPORTED_OFFICE_FORMATS + SUPPORTED_EBOOK_FORMATS

# =============================================================================
# CONVERSION TOOLS
# =============================================================================

REQUIRED_TOOLS = {
    'libreoffice': ['soffice', '--version'],
    'pandoc': ['pandoc', '--version']
}

# =============================================================================
# MARKER CONFIGURATION
# =============================================================================

MARKER_CONFIG = {
    'paginate_output': False,
    'pdftext_workers': 1,
    'layout_batch_size': 1,
    'recognition_batch_size': 1,
}

# =============================================================================
# CAPTION PATTERNS (for tables and equations only)
# =============================================================================

CAPTION_PATTERNS = {
    'table': [
        r'Table\s+(\d+(?:\.\d+)*)',
        r'Tbl\.?\s+(\d+(?:\.\d+)*)',
        r'TABLE\s+(\d+(?:\.\d+)*)',
    ],
    'equation': [
        r'Equation\s+(\d+(?:\.\d+)*)',
        r'\((\d+(?:\.\d+)*)\)',
        r'\[(\d+(?:\.\d+)*)\]',
    ]
}

# =============================================================================
# TEMP DIRECTORY
# =============================================================================

TEMP_DIR = Path(os.getenv('TEMP', '/tmp')) / 'extract_materials'
TEMP_DIR.mkdir(parents=True, exist_ok=True)
