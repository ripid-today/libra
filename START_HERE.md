# PRML Textbook Extraction - START HERE

## 📁 Files Created

| File | Purpose |
|------|---------|
| `PRML_EXTRACTION_COMMANDS.txt` | **USE THIS** - Contains all 32 extraction commands |
| `GENERATE_PRML_COMMANDS.bat` | Batch file that generates the commands (already done) |
| `Pattern Recognition and Machine Learning/` | Folder where chapters will be saved |

## 🚀 How to Extract (Simple Steps)

### Step 1: Open the Command File
Open this file in Notepad:
```
C:\Users\uyenl\knowledge_base\PRML_EXTRACTION_COMMANDS.txt
```

### Step 2: Copy Commands One by One
1. **Copy ONE command** from the text file
2. **Paste into pi terminal**
3. **Wait 2-3 minutes** for completion
4. **Copy next command**
5. **Repeat** until all 32 commands are done

### Step 3: Merge Chapter 1 Parts
Chapter 1 is split into 3 parts. After those 3 commands complete, run:
```
read "C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\01-Introduction-part1.md" "C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\01-Introduction-part2.md" "C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\01-Introduction-part3.md" | save "C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\01 - Introduction.md"
```

Then delete the part files:
```
rm "C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\01-Introduction-part*.md"
```

## 📊 What You'll Get

After running all commands, you'll have:

```
C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\
├── 01 - Introduction.md (merged from 3 parts)
├── 02 - Probability Distributions.md
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

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Each command | 2-3 minutes |
| Each chapter | 4-9 minutes |
| All 14 chapters | 2-3 hours |

## 💡 Tips for Overnight Run

1. **Keep laptop plugged in** (don't run on battery)
2. **Keep lid open** or set power options to "Do nothing" when closing lid
3. **Lock screen** (Windows+L) - extraction continues
4. **Don't close the pi terminal window**
5. **Check progress** in the morning by looking at file sizes

## ✅ Verify Progress

To check if extraction is working:

```bash
ls -lh "/c/Users/uyenl/knowledge_base/Pattern Recognition and Machine Learning/"
```

File sizes should be growing (expect 50-200 KB per chapter).

## 🚨 If Something Goes Wrong

1. **Check file sizes** - empty files (0 bytes) mean that command didn't run
2. **Re-run failed commands** - just copy that specific command again
3. **Don't worry about overwriting** - `save` creates new files, `tee -a` appends

## ❓ Need Help?

- Chapter 1 has 3 commands (pages 1-30, 31-60, 61-66)
- Commands use `save` for first batch, `tee -a` for subsequent batches
- Each command extracts ~30 pages to stay within time limits

---

**Ready? Open `PRML_EXTRACTION_COMMANDS.txt` and start copying commands!**