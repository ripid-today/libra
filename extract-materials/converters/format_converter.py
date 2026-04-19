"""
Format conversion utilities for non-PDF documents.
Converts DOCX, PPTX, EPUB, etc. to PDF using LibreOffice and Pandoc.
"""
import subprocess
import shutil
import tempfile
from pathlib import Path

from config import TEMP_DIR, ERROR_INVALID_FORMAT
from utils.validators import validate_conversion_tools


def convert_to_pdf(input_path: Path) -> Path:
    """
    Convert input file to PDF based on its format.
    
    Args:
        input_path: Path to input file (office doc, ebook, etc.)
        
    Returns:
        Path: Path to generated PDF file (in temp directory)
        
    Raises:
        ValueError: If format is invalid or conversion fails
        RuntimeError: If conversion tools are missing
    """
    file_ext = input_path.suffix.lower()
    
    # Validate tools are available
    validate_conversion_tools()
    
    if file_ext in ['.docx', '.doc', '.pptx', '.ppt', '.odt', '.ods', '.odp', '.rtf']:
        return convert_office_to_pdf(input_path)
    elif file_ext in ['.epub', '.mobi', '.azw', '.azw3']:
        return convert_ebook_to_pdf(input_path)
    elif file_ext == '.pdf':
        return input_path
    else:
        raise ValueError(f"{ERROR_INVALID_FORMAT} Cannot convert: {file_ext}")


def convert_office_to_pdf(input_path: Path) -> Path:
    """
    Convert Office documents to PDF using LibreOffice.
    
    Args:
        input_path: Path to Office document
        
    Returns:
        Path: Path to generated PDF
        
    Raises:
        RuntimeError: If conversion fails
    """
    # Create temp directory for this conversion
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR, prefix='office_convert_')
    
    try:
        # LibreOffice command
        cmd = [
            'soffice',
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', temp_dir,
            str(input_path)
        ]
        
        # Run conversion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"LibreOffice conversion failed: {result.stderr}"
            )
        
        # Find generated PDF
        input_stem = input_path.stem
        possible_names = [
            input_stem + '.pdf',
            input_stem.replace(' ', '_') + '.pdf'
        ]
        
        pdf_path = None
        for name in possible_names:
            candidate = Path(temp_dir) / name
            if candidate.exists():
                pdf_path = candidate
                break
        
        # If not found by name, take first PDF in directory
        if not pdf_path:
            pdfs = list(Path(temp_dir).glob('*.pdf'))
            if pdfs:
                pdf_path = pdfs[0]
        
        if not pdf_path or not pdf_path.exists():
            raise RuntimeError("PDF conversion succeeded but output file not found")
        
        return pdf_path
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("LibreOffice conversion timed out (5 minutes)")
    except Exception as e:
        # Cleanup on failure
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Office conversion failed: {e}")


def convert_ebook_to_pdf(input_path: Path) -> Path:
    """
    Convert eBook formats to PDF using Pandoc.
    
    Args:
        input_path: Path to eBook file
        
    Returns:
        Path: Path to generated PDF
        
    Raises:
        RuntimeError: If conversion fails
    """
    # Create output path in temp directory
    output_path = TEMP_DIR / f"{input_path.stem}_converted.pdf"
    
    try:
        # Pandoc command
        cmd = [
            'pandoc',
            '-f', 'epub',  # or auto-detect format
            '-t', 'pdf',
            '-o', str(output_path),
            str(input_path)
        ]
        
        # Adjust format based on extension
        file_ext = input_path.suffix.lower()
        if file_ext == '.mobi':
            cmd[1] = 'mobi'
        elif file_ext in ['.azw', '.azw3']:
            cmd[1] = 'epub'  # Pandoc treats AZW as EPUB
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Pandoc conversion failed: {result.stderr}")
        
        if not output_path.exists():
            raise RuntimeError("PDF conversion succeeded but output file not found")
        
        return output_path
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Pandoc conversion timed out (5 minutes)")
    except Exception as e:
        raise RuntimeError(f"eBook conversion failed: {e}")


def cleanup_temp_files(temp_paths: list) -> None:
    """
    Clean up temporary files and directories.
    
    Args:
        temp_paths: List of Path objects to remove
    """
    for path in temp_paths:
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            # Ignore cleanup errors
            pass
