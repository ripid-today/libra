@echo off
REM DEBUG VERSION - Shows errors before closing
chcp 65001 >nul

echo ============================================================
echo PRML TEXTBOOK EXTRACTION - DEBUG MODE
echo ============================================================
echo.

set "PYTHON=C:\Users\uyenl\PDF-Extract-Kit\venv\Scripts\python.exe"
set "SCRIPT=C:\Users\uyenl\knowledge_base\extract_prml_standalone.py"
set "PDF=C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf"

echo Checking requirements...
echo.

if not exist "%PYTHON%" (
    echo [ERROR] Python not found
    echo Expected: %PYTHON%
    pause
    exit /b 1
)
echo [OK] Python found at %PYTHON%

if not exist "%SCRIPT%" (
    echo [ERROR] Script not found
    echo Expected: %SCRIPT%
    pause
    exit /b 1
)
echo [OK] Script found at %SCRIPT%

if not exist "%PDF%" (
    echo [ERROR] PDF not found
    echo Expected: %PDF%
    pause
    exit /b 1
)
echo [OK] PDF found at %PDF%

echo.
echo ============================================================
echo Running extraction...
echo ============================================================
echo.

REM Run with output visible
"%PYTHON%" "%SCRIPT%"

if errorlevel 1 (
    echo.
    echo [ERROR] Extraction failed with error code %errorlevel%
    pause
    exit /b 1
)

echo.
echo ============================================================
echo DONE!
echo ============================================================
pause
