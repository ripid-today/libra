"""
Input validation utilities for extract-materials.
"""
import os
import subprocess
from pathlib import Path

from config import (
    SUPPORTED_INPUT_FORMATS,
    SUPPORTED_OFFICE_FORMATS,
    SUPPORTED_EBOOK_FORMATS,
    REQUIRED_TOOLS,
    ERROR_INVALID_FORMAT
)


def validate_input_file(input_path: Path) -> None:
    """
    Validate that input file exists and has supported extension.
    
    Args:
        input_path: Path to input file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    
    file_ext = input_path.suffix.lower()
    if file_ext not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(f"{ERROR_INVALID_FORMAT} Unsupported format: {file_ext}")


def validate_output_path(output_path: Path) -> None:
    """
    Validate output path and create directory if needed.
    Per requirements: create automatically if doesn't exist.
    
    Args:
        output_path: Path to output directory
        
    Raises:
        PermissionError: If directory cannot be created or written to
    """
    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {output_path}")
        # Test write permission
        test_file = output_path / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            raise PermissionError(f"No write permission for output directory: {output_path}")
    else:
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"Cannot create output directory: {output_path}")


def check_file_format(input_path: Path) -> str:
    """
    Determine file format category.
    
    Args:
        input_path: Path to input file
        
    Returns:
        str: 'pdf', 'office', or 'ebook'
        
    Raises:
        ValueError: If format is not supported
    """
    file_ext = input_path.suffix.lower()
    
    if file_ext == '.pdf':
        return 'pdf'
    elif file_ext in SUPPORTED_OFFICE_FORMATS:
        return 'office'
    elif file_ext in SUPPORTED_EBOOK_FORMATS:
        return 'ebook'
    else:
        raise ValueError(f"{ERROR_INVALID_FORMAT} Format: {file_ext}")


def validate_conversion_tools() -> dict:
    """
    Check if required conversion tools are installed.
    
    Returns:
        dict: Tool availability status {tool_name: bool}
        
    Raises:
        RuntimeError: If critical tools are missing with installation instructions
    """
    availability = {}
    missing_tools = []
    
    for tool_name, check_cmd in REQUIRED_TOOLS.items():
        try:
            result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            availability[tool_name] = result.returncode == 0
            if result.returncode != 0:
                missing_tools.append(tool_name)
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            availability[tool_name] = False
            missing_tools.append(tool_name)
    
    if missing_tools:
        install_instructions = []
        if 'libreoffice' in missing_tools:
            install_instructions.append(
                "- LibreOffice: Download from https://www.libreoffice.org/download/download/"
            )
        if 'pandoc' in missing_tools:
            install_instructions.append(
                "- Pandoc: Download from https://pandoc.org/installing.html"
            )
        
        raise RuntimeError(
            f"Missing required tools: {', '.join(missing_tools)}\n"
            f"Installation instructions:\n" + "\n".join(install_instructions)
        )
    
    return availability


def is_tool_available(tool_name: str) -> bool:
    """
    Check if a specific tool is available.
    
    Args:
        tool_name: Name of the tool to check
        
    Returns:
        bool: True if tool is available
    """
    if tool_name not in REQUIRED_TOOLS:
        return False
    
    check_cmd = REQUIRED_TOOLS[tool_name]
    try:
        result = subprocess.run(
            check_cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return False
