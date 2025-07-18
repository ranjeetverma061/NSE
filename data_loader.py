"""
Data Loader Module for Stock Forecasting Application
Handles data fetching, preprocessing, and validation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
warnings.filterwarnings('ignore')

class StockDataLoader:
    """Class to handle stock data loading and preprocessing"""
    
    def __init__(self):
        self.data = None
        self.symbol = None
        
    def fetch_stock_data(self, symbol, period="2y", interval="1d", max_retries=3):
        """
        Fetch stock data from Yahoo Finance with rate limiting and retry logic
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'TSLA')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            max_retries (int): Maximum number of retry attempts
        
        Returns:
            pandas.DataFrame: Stock data with OHLCV columns
        """
        self.symbol = symbol.upper()
        
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = random.uniform(1, 3) * (attempt + 1)
                    print(f"Retrying {symbol} in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                
                # Create ticker without custom session (let yfinance handle it)
                ticker = yf.Ticker(self.symbol)
                
                # Try different approaches if the first fails
                data = None
                
                # Method 1: Direct history call
                try:
                    data = ticker.history(period=period, interval=interval, auto_adjust=True, prepost=True)
                except Exception as e:
                    print(f"Method 1 failed for {symbol}: {str(e)}")
                
                # Method 2: Use yf.download if ticker.history fails
                if data is None or data.empty:
                    try:
                        print(f"Trying alternative method for {symbol}...")
                        data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
                    except Exception as e:
                        print(f"Method 2 failed for {symbol}: {str(e)}")
                
                # Method 3: Try shorter period if longer period fails
                if data is None or data.empty:
                    try:
                        shorter_periods = ['1y', '6mo', '3mo', '1mo']
                        for short_period in shorter_periods:
                            if short_period != period:
                                print(f"Trying shorter period {short_period} for {symbol}...")
                                data = ticker.history(period=short_period, interval=interval)
                                if not data.empty:
                                    print(f"Success with {short_period} period!")
                                    break
                    except Exception as e:
                        print(f"Method 3 failed for {symbol}: {str(e)}")
                
                if data is not None and not data.empty:
                    # Clean up the data
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    
                    # Ensure we have the required columns
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in data.columns for col in required_columns):
                        print(f"✅ Successfully fetched {len(data)} rows for {symbol}")
                        self.data = data
                        return data
                    else:
                        print(f"❌ Missing required columns for {symbol}: {data.columns.tolist()}")
                
                print(f"❌ No data available for {symbol} (attempt {attempt + 1})")
                
            except Exception as e:
                print(f"❌ Error fetching data for {symbol} (attempt {attempt + 1}): {str(e)}")
                if "429" in str(e) or "Too Many Requests" in str(e):
                    print("Rate limited - increasing delay...")
                    time.sleep(random.uniform(5, 10))
        
        print(f"❌ Failed to fetch data for {symbol} after {max_retries} attempts")
        return None
    
    def get_stock_info(self, symbol, max_retries=2):
        """
        Get stock information and metadata
        
        Args:
            symbol (str): Stock symbol
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            dict: Stock information
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(random.uniform(1, 2))
                
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return info if info else {}
            except Exception as e:
                print(f"Error fetching info for {symbol} (attempt {attempt + 1}): {str(e)}")
                if "429" in str(e):
                    time.sleep(random.uniform(3, 6))
        return {}
    
    def clean_data(self, data):
        """
        Clean and preprocess the stock data
        
        Args:
            data (pandas.DataFrame): Raw stock data
            
        Returns:
            pandas.DataFrame: Cleaned stock data
        """
        if data is None or data.empty:
            return None
            
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Remove timezone information if present
        if cleaned_data.index.tz is not None:
            cleaned_data.index = cleaned_data.index.tz_localize(None)
        
        # Handle missing values
        cleaned_data = self._handle_missing_values(cleaned_data)
        
        # Remove outliers
        cleaned_data = self._remove_outliers(cleaned_data)
        
        # Validate data integrity
        cleaned_data = self._validate_data(cleaned_data)
        
        return cleaned_data
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        # Forward fill missing values
        data = data.fillna(method='ffill')
        
        # Backward fill any remaining missing values
        data = data.fillna(method='bfill')
        
        # If there are still missing values, interpolate
        data = data.interpolate(method='linear')
        
        return data
    
    def _remove_outliers(self, data, z_threshold=3):
        """
        Remove outliers using Z-score method
        
        Args:
            data (pandas.DataFrame): Stock data
            z_threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pandas.DataFrame: Data with outliers removed
        """
        # Calculate Z-scores for price columns
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        for col in price_columns:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data.loc[z_scores > z_threshold, col] = np.nan
        
        # Handle the new missing values
        data = data.fillna(method='ffill')
        data = data.fillna(method='bfill')
        
        return data
    
    def _validate_data(self, data):
        """
        Validate data integrity
        
        Args:
            data (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Validated data
        """
        # Ensure High >= Low
        invalid_high_low = data['High'] < data['Low']
        if invalid_high_low.any():
            print(f"Warning: Found {invalid_high_low.sum()} rows where High < Low. Fixing...")
            # Swap High and Low values
            data.loc[invalid_high_low, ['High', 'Low']] = data.loc[invalid_high_low, ['Low', 'High']].values
        
        # Ensure Open and Close are within High and Low range
        data['Open'] = np.clip(data['Open'], data['Low'], data['High'])
        data['Close'] = np.clip(data['Close'], data['Low'], data['High'])
        
        # Remove rows with zero or negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            data = data[data[col] > 0]
        
        # Remove rows with zero volume (optional, as some stocks might have zero volume days)
        # data = data[data['Volume'] >= 0]
        
        return data
    
    def normalize_data(self, data, method='minmax'):
        """
        Normalize the data for machine learning
        
        Args:
            data (pandas.DataFrame): Stock data
            method (str): Normalization method ('minmax', 'zscore')
            
        Returns:
            pandas.DataFrame: Normalized data
            dict: Scaling parameters for inverse transformation
        """
        normalized_data = data.copy()
        scaling_params = {}
        
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        if method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            
            for col in price_columns:
                if col in data.columns:
                    scaler = MinMaxScaler()
                    normalized_data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1)).flatten()
                    scaling_params[col] = scaler
            
            # Normalize volume separately
            if 'Volume' in data.columns:
                volume_scaler = MinMaxScaler()
                normalized_data['Volume'] = volume_scaler.fit_transform(data['Volume'].values.reshape(-1, 1)).flatten()
                scaling_params['Volume'] = volume_scaler
                
        elif method == 'zscore':
            from sklearn.preprocessing import StandardScaler
            
            for col in price_columns:
                if col in data.columns:
                    scaler = StandardScaler()
                    normalized_data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1)).flatten()
                    scaling_params[col] = scaler
            
            # Normalize volume separately
            if 'Volume' in data.columns:
                volume_scaler = StandardScaler()
                normalized_data['Volume'] = volume_scaler.fit_transform(data['Volume'].values.reshape(-1, 1)).flatten()
                scaling_params['Volume'] = volume_scaler
        
        return normalized_data, scaling_params
    
    def inverse_normalize(self, data, scaling_params, columns=None):
        """
        Inverse normalize the data
        
        Args:
            data (numpy.array or pandas.DataFrame): Normalized data
            scaling_params (dict): Scaling parameters from normalize_data
            columns (list): Columns to inverse normalize
            
        Returns:
            numpy.array or pandas.DataFrame: Denormalized data
        """
        if columns is None:
            columns = ['Close']  # Default to Close price
        
        if isinstance(data, pd.DataFrame):
            denormalized_data = data.copy()
            for col in columns:
                if col in scaling_params and col in data.columns:
                    denormalized_data[col] = scaling_params[col].inverse_transform(
                        data[col].values.reshape(-1, 1)
                    ).flatten()
            return denormalized_data
        else:
            # Assume it's a numpy array for Close price
            if 'Close' in scaling_params:
                return scaling_params['Close'].inverse_transform(data.reshape(-1, 1)).flatten()
            else:
                return data
    
    def split_data(self, data, train_ratio=0.8, val_ratio=0.1):
        """
        Split data into train, validation, and test sets
        
        Args:
            data (pandas.DataFrame): Stock data
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        return train_data, val_data, test_data
    
    def get_popular_stocks(self):
        """
        Get a list of popular stock symbols
        
        Returns:
            dict: Dictionary of stock names and symbols
        """
        return {
            # Technology
            'Apple': 'AAPL',
            'Microsoft': 'MSFT',
            'Google (Alphabet)': 'GOOGL',
            'Tesla': 'TSLA',
            'Amazon': 'AMZN',
            'Meta (Facebook)': 'META',
            'Netflix': 'NFLX',
            'NVIDIA': 'NVDA',
            
            # Indian Stocks (NSE)
            'Reliance Industries': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'HDFC Bank': 'HDFCBANK.NS',
            'Infosys': 'INFY.NS',
            'ICICI Bank': 'ICICIBANK.NS',
            'Hindustan Unilever': 'HINDUNILVR.NS',
            'State Bank of India': 'SBIN.NS',
            'Kotak Mahindra Bank': 'KOTAKBANK.NS',
            'Bharti Airtel': 'BHARTIARTL.NS',
            'Asian Paints': 'ASIANPAINT.NS',
            'ITC': 'ITC.NS',
            'Axis Bank': 'AXISBANK.NS',
            'Larsen & Toubro': 'LT.NS',
            'Sun Pharma': 'SUNPHARMA.NS',
            'Titan Company': 'TITAN.NS',
            'UltraTech Cement': 'ULTRACEMCO.NS',
            'Nestle India': 'NESTLEIND.NS',
            'Bajaj Finance': 'BAJFINANCE.NS',
            'Maruti Suzuki': 'MARUTI.NS',
            'Avenue Supermarts (DMart)': 'DMART.NS',
            'Wipro': 'WIPRO.NS',
            'HCL Technologies': 'HCLTECH.NS',
            'Bajaj Auto': 'BAJAJ-AUTO.NS',
            'Bharat Petroleum': 'BPCL.NS',
            'Hindalco': 'HINDALCO.NS',
            'Adani Ports': 'ADANIPORTS.NS',
            'Vedanta': 'VEDL.NS',
            'Hero MotoCorp': 'HEROMOTOCO.NS',
            'Tata Steel': 'TATASTEEL.NS',
            'Tata Motors': 'TATAMOTORS.NS',
            'HDFC': 'HDFC.NS',
            'Zee Entertainment': 'ZEEL.NS',
            'Power Grid Corporation': 'POWERGRID.NS',
            'Eicher Motors': 'EICHERMOT.NS',
            'NTPC': 'NTPC.NS',
            'Dr Reddys Laboratories': 'DRREDDY.NS',
            'Britannia Industries': 'BRITANNIA.NS',
            'Grasim Industries': 'GRASIM.NS',
            'JSW Steel': 'JSWSTEEL.NS',
            'Steel Authority of India': 'SAIL.NS',
            'IndusInd Bank': 'INDUSINDBK.NS',
            'Coal India': 'COALINDIA.NS',
            'Oil & Natural Gas Corporation': 'ONGC.NS',
            'Bajaj Finserv': 'BAJAJFINSV.NS',
            'Mahindra & Mahindra': 'M&M.NS',
            'Tech Mahindra': 'TECHM.NS',
            'Cipla': 'CIPLA.NS',
            'Shree Cement': 'SHREECEM.NS',
            'Apollo Hospitals': 'APOLLOHOSP.NS',
            'Divis Laboratories': 'DIVISLAB.NS',
            'SBI Life Insurance': 'SBILIFE.NS',
            'HDFC Life Insurance': 'HDFCLIFE.NS',
            'ICICI Prudential Life': 'ICICIPRULI.NS',
            'Adani Green Energy': 'ADANIGREEN.NS',
            'Adani Enterprises': 'ADANIENT.NS',
            'Adani Total Gas': 'ATGL.NS',
            'Havells India': 'HAVELLS.NS',
            'Pidilite Industries': 'PIDILITIND.NS',
            'Berger Paints': 'BERGEPAINT.NS',
            'Godrej Consumer Products': 'GODREJCP.NS',
            'Dabur India': 'DABUR.NS',
            'Marico': 'MARICO.NS',
            'Colgate-Palmolive': 'COLPAL.NS',
            'Hindustan Zinc': 'HINDZINC.NS',
            'NMDC': 'NMDC.NS',
            'Gail India': 'GAIL.NS',
            'Indian Oil Corporation': 'IOC.NS',
            'Hindustan Petroleum': 'HINDPETRO.NS',
            'Bank of Baroda': 'BANKBARODA.NS',
            'Punjab National Bank': 'PNB.NS',
            'Union Bank of India': 'UNIONBANK.NS',
            'Canara Bank': 'CANBK.NS',
            'Yes Bank': 'YESBANK.NS',
            'IDFC First Bank': 'IDFCFIRSTB.NS',
            'Federal Bank': 'FEDERALBNK.NS',
            'Bandhan Bank': 'BANDHANBNK.NS',
            'Paytm': 'PAYTM.NS',
            'Zomato': 'ZOMATO.NS',
            'Nykaa': 'NYKAA.NS',
            'PB Fintech (PolicyBazaar)': 'PBFINTECH.NS',
            'Delhivery': 'DELHIVERY.NS',
            'LIC of India': 'LICI.NS',
            'JSW Energy': 'JSWENERGY.NS',
            'Torrent Pharmaceuticals': 'TORNTPHARM.NS',
            'Lupin': 'LUPIN.NS',
            'Aurobindo Pharma': 'AUROPHARMA.NS',
            'Biocon': 'BIOCON.NS',
            'Cadila Healthcare': 'CADILAHC.NS',
            'Abbott India': 'ABBOTINDIA.NS',
            'Glaxo SmithKline Pharmaceuticals': 'GLAXO.NS',
            'Pfizer': 'PFIZER.NS',
            'Tata Consumer Products': 'TATACONSUM.NS',
            'United Spirits': 'UBL.NS',
            'United Breweries': 'UBL.NS',
            'Page Industries': 'PAGEIND.NS',
            'MRF': 'MRF.NS',
            'Balkrishna Industries': 'BALKRISIND.NS',
            'Bharat Forge': 'BHARATFORG.NS',
            'Motherson Sumi Systems': 'MOTHERSUMI.NS',
            'Bosch': 'BOSCHLTD.NS',
            'Siemens': 'SIEMENS.NS',
            'ABB India': 'ABB.NS',
            'L&T Infotech': 'LTI.NS',
            'Mphasis': 'MPHASIS.NS',
            'Mindtree': 'MINDTREE.NS',
            'Persistent Systems': 'PERSISTENT.NS',
            'L&T Technology Services': 'LTTS.NS',
            'Coforge': 'COFORGE.NS',
            'HDFC Asset Management': 'HDFCAMC.NS',
            'Nippon Life India Asset Management': 'NAM-INDIA.NS',
            'SBI Cards': 'SBICARD.NS',
            'Bajaj Holdings': 'BAJAJHLDNG.NS',
            'Muthoot Finance': 'MUTHOOTFIN.NS',
            'Manappuram Finance': 'MANAPPURAM.NS',
            'PVR INOX': 'PVRINOX.NS',
            'IndiGo (InterGlobe Aviation)': 'INDIGO.NS',
            'SpiceJet': 'SPICEJET.NS',
            'IRB Infrastructure': 'IRB.NS',
            'GMR Infrastructure': 'GMRINFRA.NS',
            'Oberoi Realty': 'OBEROIRLTY.NS',
            'DLF': 'DLF.NS',
            'Godrej Properties': 'GODREJPROP.NS',
            'Sobha': 'SOBHA.NS',
            'Brigade Enterprises': 'BRIGADE.NS',
            'Phoenix Mills': 'PFS.NS',
            'Aditya Birla Fashion': 'ABFRL.NS',
            'Trent': 'TRENT.NS',
            'V-Mart Retail': 'VMART.NS',
            'Future Retail': 'FRETAIL.NS',
            
            # Other Popular Stocks
            'Berkshire Hathaway': 'BRK-B',
            'Johnson & Johnson': 'JNJ',
            'JPMorgan Chase': 'JPM',
            'Visa': 'V',
            'Procter & Gamble': 'PG',
            'Mastercard': 'MA',
            'UnitedHealth': 'UNH',
            'Home Depot': 'HD',
            'Bank of America': 'BAC',
            'Coca-Cola': 'KO'
        }
