@echo off
REM Setup script for Infant Breathing Monitor (Windows)
REM Creates virtual environment and installs dependencies

echo ==========================================
echo Infant Breathing Monitor - Setup
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.7 or higher.
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist venv (
    echo Warning: Virtual environment already exists.
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "%RECREATE%"=="y" (
        rmdir /s /q venv
        python -m venv venv
        echo Virtual environment recreated
    ) else (
        echo Using existing virtual environment
    )
) else (
    python -m venv venv
    echo Virtual environment created
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Verify installation
echo.
echo Verifying installation...
python -c "import cv2; import numpy; import mediapipe; print('All packages installed successfully')"

REM Create activation helper
echo @echo off > activate.bat
echo call venv\Scripts\activate.bat >> activate.bat
echo echo Virtual environment activated >> activate.bat
echo echo Run: python breathing_monitor.py >> activate.bat

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To activate the virtual environment:
echo   venv\Scripts\activate
echo   REM or simply:
echo   activate.bat
echo.
echo To test your camera:
echo   python test_camera.py
echo.
echo To run the monitor:
echo   python breathing_monitor.py
echo   REM or advanced version:
echo   python breathing_monitor_advanced.py
echo.
echo To deactivate when done:
echo   deactivate
echo.
pause

