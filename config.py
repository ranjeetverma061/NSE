# Configuration file for NSE Stock Analyzer

# Default stocks to display in the dropdown
DEFAULT_STOCKS = {
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'INFY': 'INFY.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'SBIN': 'SBIN.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'ASIANPAINT': 'ASIANPAINT.NS',
    'ITC': 'ITC.NS',
    'AXISBANK': 'AXISBANK.NS',
    'LT': 'LT.NS',
    'DMART': 'DMART.NS',
    'SUNPHARMA': 'SUNPHARMA.NS',
    'TITAN': 'TITAN.NS',
    'ULTRACEMCO': 'ULTRACEMCO.NS',
    'NESTLEIND': 'NESTLEIND.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'MARUTI': 'MARUTI.NS',
    'WIPRO': 'WIPRO.NS',
    'TECHM': 'TECHM.NS',
    'POWERGRID': 'POWERGRID.NS',
    'NTPC': 'NTPC.NS',
    'ONGC': 'ONGC.NS'
}

# Technical indicator parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

SMA_SHORT = 20
SMA_LONG = 50

EMA_SHORT = 12
EMA_LONG = 26

# Chart settings
CHART_HEIGHT = 800
CHART_THEME = 'plotly'  # 'plotly', 'plotly_white', 'plotly_dark'

# Colors
BULLISH_COLOR = '#00ff00'
BEARISH_COLOR = '#ff0000'
NEUTRAL_COLOR = '#ffff00'

# Data source settings
DATA_SOURCE = 'yfinance'  # 'yfinance' or 'nsepy'
UPDATE_INTERVAL = 300  # seconds

# App settings
APP_TITLE = "NSE Stock Analyzer"
APP_ICON = "📈"
LAYOUT = "wide"  # "wide" or "centered"

# Disclaimer text
DISCLAIMER = """
**Disclaimer:** This tool is for educational and informational purposes only. 
It should not be considered as financial advice. Always consult with a qualified 
financial advisor before making investment decisions. Past performance does not 
guarantee future results.
"""
