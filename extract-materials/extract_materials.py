#!/usr/bin/env python3
"""
extract-materials: Extract textbooks and academic materials to structured Markdown.
Simplified version: No OCR, No LLM, No images. Digital page numbers only.

Usage:
    extract_materials.py <input_file> --output <output_path> [options]
    
    extract_materials.py textbook.pdf --output ./extracted --start 10 --end 50 --name "chapter2"
    extract_materials.py chapter.docx --output ./notes --start 5 --end 20
"""
import argparse
import sys
from pathlib import Path

# Local imports
from config import (
    ERROR_NO_INPUT,
    ERROR_NO_OUTPUT_PATH,
    ERROR_INVALID_FORMAT,
)
from utils.validators import (
    validate_input_file,
    validate_output_path,
    check_file_format
)
from utils.file_naming import generate_output_filename
from converters.format_converter import convert_to_pdf, cleanup_temp_files
from extractors.page_mapper import (
    resolve_page_range,
    get_total_pages
)
from extractors.marker_extractor import extract_document
from extractors.content_processor import process_content, cleanup_markdown


def parse_arguments():
    """
    Parse and validate command line arguments.
    
    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Extract textbooks to structured Markdown (No OCR, No images)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s textbook.pdf --output ./extracted
  %(prog)s chapter.docx --output ./notes --start 5 --end 20
  %(prog)s book.epub --output ./notes --name "extracted"
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
        help='Input file path (PDF, DOCX, EPUB, etc.)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory path (required)'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=None,
        help='Start page number (1-based digital page number)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End page number (1-based digital page number)'
    )
    
    parser.add_argument(
        '--name', '-n',
        type=str,
        default=None,
        help='Output filename without extension (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if args.input_file is None:
        print(f"Error: {ERROR_NO_INPUT}", file=sys.stderr)
        sys.exit(1)
    
    return args


def run_extraction_pipeline(
    input_file: Path,
    output_path: Path,
    start_page: int = None,
    end_page: int = None,
    output_name: str = None,
) -> Path:
    """
    Main extraction pipeline orchestrator.
    
    Args:
        input_file: Path to input file
        output_path: Directory for output
        start_page: Digital start page (1-based, optional)
        end_page: Digital end page (1-based, optional)
        output_name: Specific output filename (optional)
        
    Returns:
        Path: Path to generated markdown file
    """
    temp_files = []
    pdf_path = None
    
    try:
        # Step 1: Validate input
        print(f"Validating input: {input_file}")
        validate_input_file(input_file)
        
        # Step 2: Validate/create output directory
        print(f"Preparing output directory: {output_path}")
        validate_output_path(output_path)
        
        # Step 3: Detect format and convert to PDF if needed
        print("Detecting file format...")
        file_format = check_file_format(input_file)
        
        if file_format == 'pdf':
            pdf_path = input_file
        elif file_format in ['office', 'ebook']:
            print(f"Converting {file_format} file to PDF...")
            pdf_path = convert_to_pdf(input_file)
            temp_files.append(pdf_path)
        else:
            raise ValueError(ERROR_INVALID_FORMAT)
        
        # Step 4: Get total pages and resolve page range (digital numbers)
        total_pages = get_total_pages(pdf_path)
        physical_range = resolve_page_range(start_page, end_page, total_pages)
        
        print(f"  Extracting digital pages {physical_range[0] + 1}-{physical_range[1] + 1}")
        
        # Step 5: Extract with marker (OCR disabled)
        print("Extracting content (text only, no OCR)...")
        rendered = extract_document(
            pdf_path=pdf_path,
            page_range=physical_range,
        )
        
        markdown_content = rendered.markdown
        
        print(f"  Extracted {len(markdown_content)} characters")
        
        # Step 6: Post-process content (no images)
        print("Processing content...")
        processed_content = process_content(markdown_content)
        processed_content = cleanup_markdown(processed_content)
        
        # Step 7: Generate output filename
        output_file = generate_output_filename(output_path, output_name)
        print(f"Output file: {output_file.name}")
        
        # Step 8: Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        print(f"\n✓ Extraction complete: {output_file}")
        return output_file
        
    finally:
        # Cleanup temp files
        if temp_files:
            print("Cleaning up temporary files...")
            cleanup_temp_files(temp_files)


def main():
    """Main entry point."""
    args = parse_arguments()
    
    input_file = Path(args.input_file).resolve()
    output_path = Path(args.output).resolve()
    
    try:
        output_file = run_extraction_pipeline(
            input_file=input_file,
            output_path=output_path,
            start_page=args.start,
            end_page=args.end,
            output_name=args.name,
        )
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"Error: {ERROR_NO_INPUT}", file=sys.stderr)
        print(f"  Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    except ValueError as e:
        error_msg = str(e)
        if ERROR_INVALID_FORMAT in error_msg:
            print(f"Error: {error_msg}", file=sys.stderr)
        else:
            print(f"Error: {ERROR_INVALID_FORMAT}", file=sys.stderr)
            print(f"  Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    except RuntimeError as e:
        print(f"Error: Conversion failed", file=sys.stderr)
        print(f"  Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: Unexpected error occurred", file=sys.stderr)
        print(f"  Details: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
