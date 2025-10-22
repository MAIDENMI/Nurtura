@echo off
REM Quick run script - automatically activates venv and runs the monitor

REM Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found!
    echo Please run setup first: setup.bat
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Parse arguments
if "%1"=="advanced" (
    echo Starting Advanced Breathing Monitor...
    python breathing_monitor_advanced.py
) else if "%1"=="graph" (
    echo Starting Graphical Breathing Monitor with Real-Time Graphs...
    python breathing_monitor_graphical.py
) else if "%1"=="graphical" (
    echo Starting Graphical Breathing Monitor with Real-Time Graphs...
    python breathing_monitor_graphical.py
) else if "%1"=="test" (
    echo Starting Camera Test...
    python test_camera.py
) else if "%1"=="config" (
    echo Validating Configuration...
    python config.py
) else (
    echo Starting Basic Breathing Monitor...
    echo.
    echo Other options:
    echo   run.bat graph     - With real-time graphs
    echo   run.bat advanced  - Advanced features
    echo   run.bat test      - Test camera
    echo.
    python breathing_monitor.py
)

