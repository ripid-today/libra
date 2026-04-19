# extract-materials

A robust CLI tool for extracting academic textbooks and materials into structured Markdown using the `marker` library with intelligent page number mapping and multi-format support.

## Features

- **Multi-format Input**: Supports PDF, DOCX, PPTX, images (PNG/JPG/TIFF), EPUB, MOBI, and more
- **Logical Page Numbers**: Extracts by printed page numbers (not physical indices)
- **Content Type Detection**: Separates and labels Tables, Images, and Equations
- **Image Embedding**: Embeds images as base64 data URIs in the markdown
- **OCR Support**: Automatic OCR for scanned documents and images
- **Smart Naming**: Auto-increment output filenames (output_1.md, output_2.md, ...)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install system dependencies:
- **LibreOffice**: For converting Office documents (DOCX, PPTX, etc.)
  - Download: https://www.libreoffice.org/download/download/
- **Pandoc**: For converting eBook formats (EPUB, MOBI)
  - Download: https://pandoc.org/installing.html

## Usage

### Basic Usage

```bash
# Extract entire PDF
python extract_materials.py textbook.pdf --output ./extracted

# Extract specific page range (logical page numbers)
python extract_materials.py textbook.pdf --output ./notes --start 10 --end 50

# Specify output name
python extract_materials.py chapter1.pdf --output ./notes --name "introduction"
```

### Supported Input Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Native support, best results |
| Images | `.png`, `.jpg`, `.tiff`, `.bmp`, `.webp` | OCR automatically enabled |
| Office | `.docx`, `.doc`, `.pptx`, `.ppt`, `.odt` | Converted via LibreOffice |
| eBooks | `.epub`, `.mobi`, `.azw`, `.azw3` | Converted via Pandoc |

### Command Line Options

```
positional arguments:
  input_file            Input file path (PDF, DOCX, image, etc.)

required arguments:
  --output, -o          Output directory path

optional arguments:
  --start               Start page number (logical/printed)
  --end                 End page number (logical/printed)
  --name, -n            Output filename without extension
  --force-ocr           Force OCR processing for all pages
  --use-llm             Use LLM for higher quality extraction
```

## Output Format

The tool produces Markdown files with semantic structure:

```markdown
## Table 1.2
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |

## Image 2.4
![Figure 2.4](data:image/jpeg;base64,/9j/4AAQ...)

## Equation 3.1
$$
E = mc^2
$$

Regular text content...
```

**Note**: Tables, Images, and Equations are only labeled with headers if their captions/numbers are detected in the original document (per design requirement).

## Error Handling

| Scenario | Error Message |
|----------|---------------|
| No input file | "Please input a textbook or academic material" |
| No output path | "Please indicate where I can extract the material" |
| Invalid format | "Please retry your materials with the valid format." |
| Page not found | Falls back to extracting all pages |

## Architecture

```
extract-materials/
├── extract_materials.py          # Main CLI entry point
├── config.py                     # Configuration constants
├── converters/
│   ├── format_converter.py       # DOCX/PPTX/EPUB → PDF
│   └── image_converter.py        # Images → PDF
├── extractors/
│   ├── marker_extractor.py       # Core marker integration
│   ├── page_mapper.py            # Logical → Physical page mapping
│   └── content_processor.py      # Post-processing and labeling
└── utils/
    ├── validators.py             # Input validation
    ├── file_naming.py            # Auto-increment naming
    └── image_encoder.py          # Base64 encoding
```

## How It Works

1. **Input Validation**: Checks file exists and format is supported
2. **Format Conversion**: Converts non-PDF files to PDF (if needed)
3. **Page Mapping**: Extracts logical page numbers from PDF metadata or text
4. **Content Extraction**: Uses marker library to extract text, tables, images, equations
5. **Post-Processing**: Adds semantic headers for detected captions
6. **Image Embedding**: Converts images to base64 for inline embedding
7. **Output Generation**: Saves as markdown with auto-increment naming

## Testing

Happy path examples:
```bash
# Simple PDF extraction
python extract_materials.py textbook.pdf -o ./output

# Page range extraction
python extract_materials.py book.pdf -o ./out --start 5 --end 25

# Auto-increment naming (run multiple times)
python extract_materials.py chap1.pdf -o ./notes
python extract_materials.py chap2.pdf -o ./notes  # Creates output_2.md

# Named output
python extract_materials.py thesis.pdf -o ./output -n "my_thesis"
```

## License

MIT License
