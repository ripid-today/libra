@echo off
chcp 65001 >nul
REM PRML Textbook Extraction - Run this file!

echo ============================================================
echo PRML TEXTBOOK EXTRACTION - STANDALONE BATCH SCRIPT
echo ============================================================
echo.
echo This will extract all 14 chapters from the PRML textbook
echo directly to: C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\
echo.
echo Features:
echo - Uses PyMuPDF (already installed in PDF-Extract-Kit)
echo - No complex model loading (fast and reliable)
echo - Extracts all 14 chapters automatically
echo - Creates proper Markdown files
echo.
echo ============================================================
echo.

set "PYTHON=C:\Users\uyenl\PDF-Extract-Kit\venv\Scripts\python.exe"
set "SCRIPT=C:\Users\uyenl\knowledge_base\extract_prml_standalone.py"
set "PDF=C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf"

echo Checking requirements...
echo.

if not exist "%PYTHON%" (
    echo [ERROR] Python not found in PDF-Extract-Kit venv
    echo Expected: %PYTHON%
    echo.
    echo Please ensure PDF-Extract-Kit is installed correctly.
    pause
    exit /b 1
)
echo [OK] Python found

if not exist "%SCRIPT%" (
    echo [ERROR] Extraction script not found
    echo Expected: %SCRIPT%
    pause
    exit /b 1
)
echo [OK] Extraction script found

if not exist "%PDF%" (
    echo [ERROR] PRML PDF not found
    echo Expected: %PDF%
    echo.
    echo Please download the PDF to the Downloads folder first.
    pause
    exit /b 1
)
echo [OK] PRML PDF found

echo.
echo ============================================================
echo Starting extraction...
echo This will take approximately 10-15 minutes for all chapters
echo ============================================================
echo.

"%PYTHON%" "%SCRIPT%"

echo.
echo ============================================================
echo DONE!
echo ============================================================
echo.
echo Check your output at:
echo C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\
echo.
pause