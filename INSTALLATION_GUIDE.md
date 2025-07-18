# NSE Stock Analyzer - Installation Guide

## Prerequisites

Before installing the NSE Stock Analyzer, you need to have Python installed on your system.

### Step 1: Install Python

1. **Download Python**:
   - Visit https://python.org/downloads/
   - Download the latest Python 3.x version (3.7 or higher)

2. **Install Python**:
   - Run the downloaded installer
   - **IMPORTANT**: Check the box "Add Python to PATH" during installation
   - Click "Install Now"

3. **Verify Installation**:
   - Open Command Prompt or PowerShell
   - Type: `python --version`
   - You should see the Python version number

### Step 2: Install NSE Stock Analyzer

#### Option A: Automatic Installation (Recommended)

1. **Navigate to the project folder**:
   ```
   cd "C:\Users\Ranjeet verma\nse-stock-analyzer"
   ```

2. **Run the setup script**:
   - Double-click `setup.bat` OR
   - Run in Command Prompt: `setup.bat`

3. **Wait for installation to complete**

#### Option B: Manual Installation

1. **Open Command Prompt or PowerShell**

2. **Navigate to the project directory**:
   ```
   cd "C:\Users\Ranjeet verma\nse-stock-analyzer"
   ```

3. **Install dependencies**:
   ```
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Running the Application

### Option A: Using the Start Script (Recommended)

1. **Double-click `start_app.bat`** OR
2. **Run in Command Prompt**: `start_app.bat`

### Option B: Manual Start

1. **Open Command Prompt or PowerShell**

2. **Navigate to the project directory**:
   ```
   cd "C:\Users\Ranjeet verma\nse-stock-analyzer"
   ```

3. **Run the application**:
   ```
   streamlit run app.py
   ```

4. **Open your browser** and go to: `http://localhost:8501`

## Using the Application

1. **Select a stock** from the dropdown menu in the sidebar
2. **Choose a time period** (1 month to 5 years)
3. **Click "Analyze Stock"** to generate the analysis
4. **View the results**:
   - Key metrics (price, volume, 52-week high/low)
   - Interactive candlestick charts
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Trading signals
   - Raw data (optional)

## Troubleshooting

### Common Issues

1. **"Python is not recognized"**:
   - Python is not installed or not in PATH
   - Reinstall Python with "Add Python to PATH" checked

2. **"pip is not recognized"**:
   - Use `python -m pip` instead of `pip`

3. **Installation errors**:
   - Update pip: `python -m pip install --upgrade pip`
   - Try installing packages individually

4. **Application won't start**:
   - Check if all dependencies are installed
   - Verify Python version (should be 3.7+)

5. **No data displayed**:
   - Check internet connection
   - Stock symbols might be temporarily unavailable

### Getting Help

- Check the console output for error messages
- Ensure you have an active internet connection
- Try restarting the application
- Contact support if issues persist

## System Requirements

- **Operating System**: Windows 10 or later
- **Python**: Version 3.7 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **Internet**: Active connection required for stock data

## Security Note

This application fetches stock data from public APIs. No personal or financial information is stored or transmitted beyond what's necessary for stock data retrieval.

## Next Steps

Once installed and running:
1. Explore different stocks and timeframes
2. Learn about the technical indicators
3. Use the trading signals as educational tools
4. Remember: This is for educational purposes only!

## Support

If you encounter any issues:
1. Check this guide first
2. Review the README.md file
3. Check the console for error messages
4. Ensure all prerequisites are met
