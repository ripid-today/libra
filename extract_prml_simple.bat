@echo off
chcp 65001 >nul
REM PRML Extraction Batch Script using PDF-Extract-Kit Python environment
REM This extracts chapters using PyMuPDF (fitz) which is already installed

echo ============================================================
echo PRML TEXTBOOK EXTRACTION - BATCH SCRIPT
echo ============================================================
echo.
echo This script will extract all 14 chapters from the PRML textbook
echo using the PDF-Extract-Kit Python environment.
echo.
echo Requirements:
echo - PDF-Extract-Kit must be installed at C:\Users\uyenl\PDF-Extract-Kit
echo - Python venv must be set up (already done)
echo.
echo ============================================================
echo.

set PDF_PATH=C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf
set OUTPUT_DIR=C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning
set PYTHON=C:\Users\uyenl\PDF-Extract-Kit\venv\Scripts\python.exe

echo Checking requirements...

if not exist "%PDF_PATH%" (
    echo ERROR: PDF file not found!
    echo Expected: %PDF_PATH%
    pause
    exit /b 1
)

echo ✓ PDF file found

if not exist "%PYTHON%" (
    echo ERROR: Python not found in PDF-Extract-Kit venv!
    echo Expected: %PYTHON%
    pause
    exit /b 1
)

echo ✓ Python environment found
echo.

REM Create output directory if not exists
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ============================================================
echo Starting extraction...
echo This will take 2-3 hours for all chapters
echo ============================================================
echo.

REM Extract all chapters using simple mode (faster, no model loading)
"%PYTHON%" "C:\Users\uyenl\knowledge_base\extract_prml_pdfkit.py" "%PDF_PATH%" -o "%OUTPUT_DIR%" --simple
echo.

echo ============================================================
echo Extraction complete!
echo ============================================================
echo.
echo Output files are in:
echo %OUTPUT_DIR%
echo.
pause