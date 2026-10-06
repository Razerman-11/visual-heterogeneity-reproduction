@echo off
chcp 65001 >nul
echo ============================================
echo  Running the full pipeline
echo ============================================
echo.
echo [0/3] Checking image quality ...
python check_image_quality.py
echo.
echo [1/3] Step 1: Semantic segmentation ...
python step1_semantic_segmentation.py
echo.
echo [2/3] Step 2: Visual indicators ...
python step2_visual_indicators.py
echo.
echo [3/3] Step 3: Visual heterogeneity ...
python step3_heterogeneity.py
echo.
echo ============================================
echo  All done. Results are in the outputs folder.
echo ============================================
pause
