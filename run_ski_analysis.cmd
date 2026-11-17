@echo off
setlocal
cd /d "%~dp0"
if not exist "analyze_ski.py" goto missing
if not exist "ski2dpose_labels.json.zip" goto missing
if not exist "ski2dpose_images_webp.zip" goto missing
if not exist ".ski-venv\Scripts\python.exe" (
    python -m venv ".ski-venv" || goto failed
)
set "SKIPY=.ski-venv\Scripts\python.exe"
if not exist "pose_landmarker_lite.task" (
    curl.exe --fail --location --retry 2 --output "pose_landmarker_lite.task.part" "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" || goto failed
    "%SKIPY%" -c "import zipfile; z=zipfile.ZipFile('pose_landmarker_lite.task.part'); assert z.testzip() is None and any(n.endswith('.tflite') for n in z.namelist())" || goto failed
    move /y "pose_landmarker_lite.task.part" "pose_landmarker_lite.task" >nul || goto failed
)
"%SKIPY%" -m pip install --only-binary=:all: mediapipe numpy || goto failed
"%SKIPY%" -c "import cv2" 2>nul || "%SKIPY%" -m pip install --only-binary=:all: opencv-python || goto failed
"%SKIPY%" analyze_ski.py --model pose_landmarker_lite.task || goto failed
echo Finished. Results: ski_comparison.csv
pause
exit /b 0
:missing
echo Place this file in the AI cv folder beside analyze_ski.py and both dataset ZIPs.
:failed
echo Analysis did not finish. See the error above.
pause
exit /b 1
