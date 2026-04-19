"""
Output file naming utilities for extract-materials.
"""
import re
from pathlib import Path


def generate_output_filename(output_path: Path, requested_name: str = None) -> Path:
    """
    Generate output filename based on requested name or auto-increment pattern.
    
    Args:
        output_path: Directory for output file
        requested_name: Optional specific filename (without extension)
        
    Returns:
        Path: Full path for output markdown file
    """
    if requested_name:
        # Ensure .md extension
        if not requested_name.endswith('.md'):
            requested_name += '.md'
        return output_path / requested_name
    else:
        # Auto-increment: find highest output_N.md and add 1
        next_number = find_next_output_number(output_path)
        return output_path / f"output_{next_number}.md"


def find_next_output_number(output_path: Path) -> int:
    """
    Find the next available output number by scanning existing files.
    Pattern: output_(
    
    Args:
        output_path: Directory to scan (flat search)
        
    Returns:
        int: Next available number (1 if no files exist)
    """
    pattern = re.compile(r'^output_(\d+)\.md$')
    max_number = 0
    
    if output_path.exists() and output_path.is_dir():
        for item in output_path.iterdir():
            if item.is_file():
                match = pattern.match(item.name)
                if match:
                    number = int(match.group(1))
                    max_number = max(max_number, number)
    
    return max_number + 1


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe filesystem usage.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename
