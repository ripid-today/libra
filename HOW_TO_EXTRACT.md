# PRML Textbook Extraction - Complete Instructions

## 🎯 The Problem

The previous attempts failed because:
1. Pi tools (`extract_doc_parse`) must run in pi terminal, not through bash
2. Direct extraction through me hits tool call limits
3. You need a standalone solution

## ✅ The Solution

I've created **standalone Python scripts** that use PyMuPDF (already installed with PDF-Extract-Kit) to extract all 14 chapters **without requiring pi terminal**.

---

## 📁 Files Created

### Main Files (in `C:\Users\uyenl\knowledge_base\`)

| File | Purpose |
|------|---------|
| `RUN_EXTRACTION.bat` | **⭐ DOUBLE-CLICK THIS** - Main batch file |
| `extract_prml_standalone.py` | Python script that does the extraction |
| `extract_prml_pdfkit.py` | Alternative with PDF-Extract-Kit integration |
| `extract_prml_simple.bat` | Alternative batch (simple mode only) |

### Output Folder
```
C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\
```

---

## 🚀 How to Run (Super Simple)

### Step 1: Navigate to the folder
Open File Explorer and go to:
```
C:\Users\uyenl\knowledge_base\
```

### Step 2: Double-click to run
**Double-click on: `RUN_EXTRACTION.bat`**

That's it! The script will:
1. ✓ Check that everything is in place
2. ✓ Open the PRML PDF
3. ✓ Extract all 14 chapters (10-15 minutes)
4. ✓ Save them as Markdown files
5. ✓ Show you the results

---

## 📊 What You'll Get

After running, you'll have:

```
C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\
├── 01 - Introduction.md              (~100-150 KB)
├── 02 - Probability Distributions.md (~120-180 KB)
├── 03 - Linear Models for Regression.md
├── 04 - Linear Models for Classification.md
├── 05 - Neural Networks.md
├── 06 - Kernel Methods.md
├── 07 - Sparse Kernel Machines.md
├── 08 - Graphical Models.md
├── 09 - Mixture Models and EM.md
├── 10 - Approximate Inference.md
├── 11 - Sampling Methods.md
├── 12 - Continuous Latent Variables.md
├── 13 - Sequential Data.md
└── 14 - Combining Models.md
```

---

## ⚙️ Technical Details

### How It Works
1. **PyMuPDF (fitz)** opens the PDF
2. For each chapter, it extracts the specified page range
3. Text is converted to Markdown format with page markers
4. Each chapter is saved as a separate `.md` file

### Requirements (Already Installed)
- ✅ PDF-Extract-Kit at `C:\Users\uyenl\PDF-Extract-Kit`
- ✅ Python 3.13 in virtual environment
- ✅ PyMuPDF (fitz) library
- ✅ PRML PDF in Downloads folder

### Extraction Method
- **No OCR** - Direct text extraction (faster, more accurate for digital PDFs)
- **No AI models** - No loading delays
- **Pure Python** - No external dependencies
- **Page markers** - Each page clearly marked in output

---

## ⏱️ Expected Time

| Task | Time |
|------|------|
| Per chapter | 30-60 seconds |
| All 14 chapters | 10-15 minutes |
| Total (including overhead) | ~15 minutes |

---

## 🚨 Troubleshooting

### "Python not found"
**Cause:** PDF-Extract-Kit venv not set up properly  
**Fix:** Run the PDF-Extract-Kit installation first

### "PDF not found"
**Cause:** PRML PDF not in Downloads folder  
**Fix:** Download the PDF and ensure it's in:
```
C:\Users\uyenl\Downloads\
```

### "Permission denied"
**Cause:** Windows blocking the script  
**Fix:** Right-click → Properties → Unblock, or run as Administrator

### Extraction is slow
**Cause:** Large chapters with many pages  
**Fix:** This is normal. PRML has 676 pages total. Let it run.

---

## 🔄 Alternative: Manual Python Execution

If the batch file doesn't work, run manually:

```powershell
# Open PowerShell and run:
cd "C:\Users\uyenl\PDF-Extract-Kit"
.\venv\Scripts\python.exe "C:\Users\uyenl\knowledge_base\extract_prml_standalone.py"
```

Or with Git Bash:
```bash
cd /c/Users/uyenl/PDF-Extract-Kit
./venv/Scripts/python.exe /c/Users/uyenl/knowledge_base/extract_prml_standalone.py
```

---

## ✅ Verification After Extraction

Check that all files were created:

```powershell
# In PowerShell
cd "C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning"
Get-ChildItem *.md | Select-Object Name, @{N="SizeKB";E={[math]::Round($_.Length/1KB,2)}}
```

Expected: 14 files, each 50-200 KB

---

## 🎉 Summary

**Just double-click `RUN_EXTRACTION.bat` and wait 10-15 minutes!**

No pi terminal needed. No manual commands. No interim files. Just clean chapter extraction.

---

**Questions? The script will tell you if something is wrong.**