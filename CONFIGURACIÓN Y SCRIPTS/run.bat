@echo off
REM FIFA World Cup 2018 Predictor - Windows Run Script

echo ========================================
echo FIFA World Cup 2018 Predictor
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed
    echo Please install Python 3.7 or higher
    pause
    exit /b 1
)

echo [OK] Python found
python --version

REM Check if data files exist
if not exist "data\matches.csv" (
    echo Error: data\matches.csv not found
    echo Please ensure all data files are in the data\ directory
    pause
    exit /b 1
)

if not exist "data\teams.csv" (
    echo Error: data\teams.csv not found
    pause
    exit /b 1
)

if not exist "data\qualified.csv" (
    echo Error: data\qualified.csv not found
    pause
    exit /b 1
)

echo [OK] All data files found

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo Installing dependencies...
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo [OK] Dependencies installed

REM Run the main script
echo.
echo ========================================
echo Starting prediction model...
echo ========================================
echo.

python main.py

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo [OK] Simulation completed successfully!
    echo ========================================
    echo.
    echo Results have been displayed above.
    echo For detailed analysis, check RESULTS.md
) else (
    echo.
    echo ========================================
    echo [ERROR] Simulation failed
    echo ========================================
)

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

pause
