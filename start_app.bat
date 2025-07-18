@echo off
echo ============================================
echo   🚀 Advanced Stock Forecasting App
echo ============================================
echo.
echo Starting AI-powered stock prediction system...
echo.
echo Features loading:
echo ✓ Real-time data fetching
echo ✓ LSTM + XGBoost models
echo ✓ Technical indicators
echo ✓ Trading signals
echo ✓ Interactive charts
echo.
echo The application will open in your default web browser.
echo Go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo.

set PYTHON_PATH="C:\Users\Ranjeet verma\AppData\Local\Programs\Python\Python311\python.exe"
set STREAMLIT_PATH="C:\Users\Ranjeet verma\AppData\Local\Programs\Python\Python311\Scripts\streamlit.exe"

%PYTHON_PATH% -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Streamlit is not installed.
    echo Please run setup.bat first to install dependencies.
    pause
    exit /b 1
)

echo Starting the application...
%STREAMLIT_PATH% run app.py

if %errorlevel% neq 0 (
    echo.
    echo Trying alternative method...
    %PYTHON_PATH% -m streamlit run app.py
)
