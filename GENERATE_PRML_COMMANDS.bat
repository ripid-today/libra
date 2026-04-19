@echo off
chcp 65001 >nul
REM PRML Extraction Command Generator
REM Run this to get all the commands you need to paste into pi terminal

set OUTPUT_FILE=C:\Users\uyenl\knowledge_base\PRML_EXTRACTION_COMMANDS.txt

echo Generating extraction commands...
echo Results will be saved to: %OUTPUT_FILE%
echo.

(
echo ================================================================================
echo PRML TEXTBOOK EXTRACTION COMMANDS
echo ================================================================================
echo Generated: %date% %time%
echo.
echo INSTRUCTIONS:
echo 1. Copy ONE command at a time from this file
echo 2. Paste it into the pi terminal
echo 3. Wait for completion (2-3 minutes per command)
echo 4. Copy the next command
echo 5. Repeat until all 32 commands are done
echo.
echo Estimated total time: 2-3 hours
echo Recommended: Do this overnight
echo.
echo ================================================================================
echo.

rem Chapter 1: Introduction (Pages 1-66) - 3 batches
echo --- CHAPTER 1: INTRODUCTION (Pages 1-66) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="1-30" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\01-Introduction-part1.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="31-60" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\01-Introduction-part2.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="61-66" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\01-Introduction-part3.md
echo.
echo # Then merge: type "C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\01-Introduction-part*.md" ^> "C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\01 - Introduction.md"
echo.
echo.

rem Chapter 2: Probability Distributions (Pages 67-136) - 3 batches
echo --- CHAPTER 2: PROBABILITY DISTRIBUTIONS (Pages 67-136) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="67-96" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\02-Probability-Distributions.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="97-126" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\02-Probability-Distributions.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="127-136" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\02-Probability-Distributions.md
echo.
echo rename "C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\02-Probability-Distributions.md" "02 - Probability Distributions.md"
echo.
echo.

rem Chapter 3: Linear Models for Regression (Pages 137-178) - 2 batches
echo --- CHAPTER 3: LINEAR MODELS FOR REGRESSION (Pages 137-178) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="137-166" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\03 - Linear Models for Regression.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="167-178" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\03 - Linear Models for Regression.md
echo.
echo.

rem Chapter 4: Linear Models for Classification (Pages 179-224) - 2 batches
echo --- CHAPTER 4: LINEAR MODELS FOR CLASSIFICATION (Pages 179-224) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="179-208" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\04 - Linear Models for Classification.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="209-224" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\04 - Linear Models for Classification.md
echo.
echo.

rem Chapter 5: Neural Networks (Pages 225-290) - 3 batches
echo --- CHAPTER 5: NEURAL NETWORKS (Pages 225-290) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="225-254" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\05 - Neural Networks.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="255-284" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\05 - Neural Networks.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="285-290" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\05 - Neural Networks.md
echo.
echo.

rem Chapter 6: Kernel Methods (Pages 291-324) - 2 batches
echo --- CHAPTER 6: KERNEL METHODS (Pages 291-324) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="291-320" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\06 - Kernel Methods.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="321-324" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\06 - Kernel Methods.md
echo.
echo.

rem Chapter 7: Sparse Kernel Machines (Pages 325-358) - 2 batches
echo --- CHAPTER 7: SPARSE KERNEL MACHINES (Pages 325-358) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="325-354" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\07 - Sparse Kernel Machines.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="355-358" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\07 - Sparse Kernel Machines.md
echo.
echo.

rem Chapter 8: Graphical Models (Pages 359-422) - 3 batches
echo --- CHAPTER 8: GRAPHICAL MODELS (Pages 359-422) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="359-388" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\08 - Graphical Models.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="389-418" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\08 - Graphical Models.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="419-422" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\08 - Graphical Models.md
echo.
echo.

rem Chapter 9: Mixture Models and EM (Pages 423-460) - 2 batches
echo --- CHAPTER 9: MIXTURE MODELS AND EM (Pages 423-460) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="423-452" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\09 - Mixture Models and EM.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="453-460" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\09 - Mixture Models and EM.md
echo.
echo.

rem Chapter 10: Approximate Inference (Pages 461-522) - 3 batches
echo --- CHAPTER 10: APPROXIMATE INFERENCE (Pages 461-522) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="461-490" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\10 - Approximate Inference.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="491-520" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\10 - Approximate Inference.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="521-522" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\10 - Approximate Inference.md
echo.
echo.

rem Chapter 11: Sampling Methods (Pages 523-558) - 2 batches
echo --- CHAPTER 11: SAMPLING METHODS (Pages 523-558) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="523-552" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\11 - Sampling Methods.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="553-558" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\11 - Sampling Methods.md
echo.
echo.

rem Chapter 12: Continuous Latent Variables (Pages 559-604) - 2 batches
echo --- CHAPTER 12: CONTINUOUS LATENT VARIABLES (Pages 559-604) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="559-588" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\12 - Continuous Latent Variables.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="589-604" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\12 - Continuous Latent Variables.md
echo.
echo.

rem Chapter 13: Sequential Data (Pages 605-652) - 2 batches
echo --- CHAPTER 13: SEQUENTIAL DATA (Pages 605-652) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="605-634" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\13 - Sequential Data.md
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="635-652" format="text" ^| tee -a C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\13 - Sequential Data.md
echo.
echo.

rem Chapter 14: Combining Models (Pages 653-676) - 1 batch
echo --- CHAPTER 14: COMBINING MODELS (Pages 653-676) ---
echo.
echo extract_doc_parse file="C:\Users\uyenl\Downloads\Pattern Recognition and Machine Learning (Christopher M. Bishop) (z-library.sk, 1lib.sk, z-lib.sk).pdf" pages="653-676" format="text" ^| save C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\14 - Combining Models.md
echo.
echo.

echo ================================================================================
echo CLEANUP COMMANDS (run after all extractions complete)
echo ================================================================================
echo.
echo # Delete temporary part files from Chapter 1
echo del "C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\01-Introduction-part*.md"
echo.
echo # Verify all files exist
echo dir "C:\Users\uyenl\knowledge_base\Pattern Recognition and Machine Learning\*.md"
echo.
echo ================================================================================
echo.
) > %OUTPUT_FILE%

echo Commands generated successfully!
echo.
echo Open this file to see all commands:
echo %OUTPUT_FILE%
echo.
echo Next steps:
echo 1. Open %OUTPUT_FILE% in Notepad
echo 2. Copy commands ONE AT A TIME
echo 3. Paste into pi terminal
echo 4. Wait 2-3 minutes per command
echo 5. Repeat for all 32 commands
echo.
pause