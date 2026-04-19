"""
Content post-processing utilities.
Adds semantic headings for tables and equations based on caption detection.
Images are discarded (not processed).
"""
import re
from typing import Optional

from config import CAPTION_PATTERNS


def process_content(markdown_content: str) -> str:
    """
    Main post-processing pipeline for extracted markdown.
    Images are ignored/discarded.
    
    Args:
        markdown_content: Raw markdown from marker
        
    Returns:
        str: Processed markdown with semantic headings
    """
    content = markdown_content
    
    # Process tables - add ## Table X.X headers if captions detected
    content = detect_and_label_tables(content)
    
    # Process equations - add ## Equation X.X headers if numbered
    content = detect_and_label_equations(content)
    
    # Note: Images are discarded (not processed)
    
    return content


def detect_and_label_tables(content: str) -> str:
    """
    Find markdown tables and add ## Table X.X headers if captions are detected.
    Skip if no caption is found.
    
    Args:
        content: Markdown content
        
    Returns:
        str: Modified content with table headers
    """
    lines = content.split('\n')
    result_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts a table (contains |)
        if '|' in line and i + 1 < len(lines) and '---' in lines[i + 1]:
            # Look for caption in lines before table
            caption_num = None
            context_start = max(0, i - 5)
            context = '\n'.join(lines[context_start:i])
            
            caption_num = extract_caption_number(context, 'table')
            
            if caption_num:
                # Insert header before table
                result_lines.append(f"## Table {caption_num}")
                result_lines.append("")
        
        result_lines.append(line)
        i += 1
    
    return '\n'.join(result_lines)


def detect_and_label_equations(content: str) -> str:
    """
    Find equations and add ## Equation X.X headers if numbered.
    Skip if no number found.
    
    Args:
        content: Markdown content
        
    Returns:
        str: Modified content with equation headers
    """
    # Pattern for block math: $$...$$
    block_math_pattern = r'\$\$([\s\S]*?)\$\$'
    
    def replace_equation(match):
        equation_content = match.group(1).strip()
        
        # Look for equation number in or around the equation
        caption_num = extract_caption_number(equation_content, 'equation')
        
        if caption_num:
            return f"## Equation {caption_num}\n\n$$\n{equation_content}\n$$"
        else:
            # No caption, return as-is
            return match.group(0)
    
    content = re.sub(block_math_pattern, replace_equation, content)
    return content


def extract_caption_number(text: str, content_type: str) -> Optional[str]:
    """
    Extract caption number from text using regex patterns.
    
    Args:
        text: Text to search
        content_type: 'table' or 'equation'
        
    Returns:
        str: Caption number (e.g., "1.2") or None
    """
    patterns = CAPTION_PATTERNS.get(content_type, [])
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def cleanup_markdown(content: str) -> str:
    """
    Clean up markdown formatting artifacts.
    
    Args:
        content: Markdown content
        
    Returns:
        str: Cleaned content
    """
    # Remove excessive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Fix spacing around headers
    content = re.sub(r'^(#{1,6}.*?)\n([^\n])', r'\1\n\n\2', content, flags=re.MULTILINE)
    
    # Ensure proper spacing before tables
    content = re.sub(r'([^\n])\n(\|)', r'\1\n\n\2', content)
    
    return content.strip()
