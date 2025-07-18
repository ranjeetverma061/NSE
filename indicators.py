"""
Technical Indicators Module for Stock Forecasting Application
Calculates various technical indicators and trading signals
"""

import pandas as pd
import numpy as np
import ta
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator

class TechnicalIndicators:
    """Class to calculate technical indicators and generate trading signals"""
    
    def __init__(self):
        pass
    
    def calculate_all_indicators(self, data):
        """
        Calculate all technical indicators
        
        Args:
            data (pandas.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pandas.DataFrame: Data with all technical indicators
        """
        if data is None or data.empty:
            return None
            
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Momentum Indicators
        df = self.calculate_rsi(df)
        df = self.calculate_stoch_rsi(df)
        
        # Trend Indicators
        df = self.calculate_macd(df)
        df = self.calculate_moving_averages(df)
        df = self.calculate_adx(df)
        
        # Volatility Indicators
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_atr(df)
        
        # Volume Indicators
        df = self.calculate_volume_indicators(df)
        
        # Price-based indicators
        df = self.calculate_price_indicators(df)
        
        return df
    
    def calculate_rsi(self, data, window=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data (pandas.DataFrame): Stock data
            window (int): RSI period
            
        Returns:
            pandas.DataFrame: Data with RSI
        """
        rsi_indicator = RSIIndicator(close=data['Close'], window=window)
        data['RSI'] = rsi_indicator.rsi()
        
        # RSI signals
        data['RSI_Overbought'] = data['RSI'] > 70
        data['RSI_Oversold'] = data['RSI'] < 30
        
        return data
    
    def calculate_stoch_rsi(self, data, window=14, smooth1=3, smooth2=3):
        """
        Calculate Stochastic RSI
        
        Args:
            data (pandas.DataFrame): Stock data
            window (int): RSI period
            smooth1 (int): First smoothing period
            smooth2 (int): Second smoothing period
            
        Returns:
            pandas.DataFrame: Data with Stochastic RSI
        """
        stoch_rsi = StochRSIIndicator(
            close=data['Close'], 
            window=window, 
            smooth1=smooth1, 
            smooth2=smooth2
        )
        data['StochRSI'] = stoch_rsi.stochrsi()
        data['StochRSI_K'] = stoch_rsi.stochrsi_k()
        data['StochRSI_D'] = stoch_rsi.stochrsi_d()
        
        return data
    
    def calculate_macd(self, data, window_slow=26, window_fast=12, window_sign=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data (pandas.DataFrame): Stock data
            window_slow (int): Slow EMA period
            window_fast (int): Fast EMA period
            window_sign (int): Signal line period
            
        Returns:
            pandas.DataFrame: Data with MACD
        """
        macd = MACD(
            close=data['Close'],
            window_slow=window_slow,
            window_fast=window_fast,
            window_sign=window_sign
        )
        
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()
        
        # MACD signals
        data['MACD_Bullish'] = (data['MACD'] > data['MACD_Signal']) & (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))
        data['MACD_Bearish'] = (data['MACD'] < data['MACD_Signal']) & (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1))
        
        return data
    
    def calculate_moving_averages(self, data):
        """
        Calculate various moving averages
        
        Args:
            data (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Data with moving averages
        """
        # Simple Moving Averages
        data['SMA_5'] = SMAIndicator(close=data['Close'], window=5).sma_indicator()
        data['SMA_10'] = SMAIndicator(close=data['Close'], window=10).sma_indicator()
        data['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
        data['SMA_50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()
        data['SMA_100'] = SMAIndicator(close=data['Close'], window=100).sma_indicator()
        data['SMA_200'] = SMAIndicator(close=data['Close'], window=200).sma_indicator()
        
        # Exponential Moving Averages
        data['EMA_5'] = EMAIndicator(close=data['Close'], window=5).ema_indicator()
        data['EMA_10'] = EMAIndicator(close=data['Close'], window=10).ema_indicator()
        data['EMA_12'] = EMAIndicator(close=data['Close'], window=12).ema_indicator()
        data['EMA_20'] = EMAIndicator(close=data['Close'], window=20).ema_indicator()
        data['EMA_26'] = EMAIndicator(close=data['Close'], window=26).ema_indicator()
        data['EMA_50'] = EMAIndicator(close=data['Close'], window=50).ema_indicator()
        data['EMA_100'] = EMAIndicator(close=data['Close'], window=100).ema_indicator()
        data['EMA_200'] = EMAIndicator(close=data['Close'], window=200).ema_indicator()
        
        # Moving average signals
        data['SMA_20_50_Golden'] = (data['SMA_20'] > data['SMA_50']) & (data['SMA_20'].shift(1) <= data['SMA_50'].shift(1))
        data['SMA_20_50_Death'] = (data['SMA_20'] < data['SMA_50']) & (data['SMA_20'].shift(1) >= data['SMA_50'].shift(1))
        data['Price_Above_SMA_20'] = data['Close'] > data['SMA_20']
        data['Price_Above_SMA_50'] = data['Close'] > data['SMA_50']
        
        return data
    
    def calculate_bollinger_bands(self, data, window=20, window_dev=2):
        """
        Calculate Bollinger Bands
        
        Args:
            data (pandas.DataFrame): Stock data
            window (int): Moving average period
            window_dev (int): Standard deviation multiplier
            
        Returns:
            pandas.DataFrame: Data with Bollinger Bands
        """
        bb = BollingerBands(close=data['Close'], window=window, window_dev=window_dev)
        
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Middle'] = bb.bollinger_mavg()
        data['BB_Lower'] = bb.bollinger_lband()
        data['BB_Width'] = bb.bollinger_wband()
        data['BB_Percent'] = bb.bollinger_pband()
        
        # Bollinger Band signals
        data['BB_Squeeze'] = data['BB_Width'] < data['BB_Width'].rolling(20).mean() * 0.5
        data['BB_Upper_Break'] = data['Close'] > data['BB_Upper']
        data['BB_Lower_Break'] = data['Close'] < data['BB_Lower']
        
        return data
    
    def calculate_atr(self, data, window=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            data (pandas.DataFrame): Stock data
            window (int): ATR period
            
        Returns:
            pandas.DataFrame: Data with ATR
        """
        data['ATR'] = ta.volatility.AverageTrueRange(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=window
        ).average_true_range()
        
        return data
    
    def calculate_adx(self, data, window=14):
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            data (pandas.DataFrame): Stock data
            window (int): ADX period
            
        Returns:
            pandas.DataFrame: Data with ADX
        """
        adx = ADXIndicator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=window
        )
        
        data['ADX'] = adx.adx()
        data['ADX_Pos'] = adx.adx_pos()
        data['ADX_Neg'] = adx.adx_neg()
        
        # ADX signals
        data['ADX_Strong_Trend'] = data['ADX'] > 25
        data['ADX_Weak_Trend'] = data['ADX'] < 20
        
        return data
    
    def calculate_volume_indicators(self, data):
        """
        Calculate volume-based indicators
        
        Args:
            data (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Data with volume indicators
        """
        # Volume SMA (manual calculation)
        data['Volume_SMA_10'] = data['Volume'].rolling(window=10).mean()
        data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
        
        # On Balance Volume
        data['OBV'] = OnBalanceVolumeIndicator(
            close=data['Close'],
            volume=data['Volume']
        ).on_balance_volume()
        
        # Chaikin Money Flow
        data['CMF'] = ChaikinMoneyFlowIndicator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            volume=data['Volume'],
            window=20
        ).chaikin_money_flow()
        
        # Volume oscillator
        data['Volume_Oscillator'] = ((data['Volume'].rolling(5).mean() - data['Volume'].rolling(10).mean()) / 
                                    data['Volume'].rolling(10).mean()) * 100
        
        # Volume signals
        data['High_Volume'] = data['Volume'] > data['Volume'].rolling(20).mean() * 1.5
        data['Low_Volume'] = data['Volume'] < data['Volume'].rolling(20).mean() * 0.5
        
        return data
    
    def calculate_price_indicators(self, data):
        """
        Calculate price-based indicators
        
        Args:
            data (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Data with price indicators
        """
        # Price change indicators
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_5D'] = data['Close'].pct_change(5)
        data['Price_Change_10D'] = data['Close'].pct_change(10)
        
        # Volatility
        data['Volatility_10D'] = data['Price_Change'].rolling(10).std()
        data['Volatility_20D'] = data['Price_Change'].rolling(20).std()
        
        # Support and Resistance levels
        data['Resistance_20D'] = data['High'].rolling(20).max()
        data['Support_20D'] = data['Low'].rolling(20).min()
        
        # Gap indicators
        data['Gap_Up'] = data['Open'] > data['Close'].shift(1) * 1.02  # 2% gap up
        data['Gap_Down'] = data['Open'] < data['Close'].shift(1) * 0.98  # 2% gap down
        
        return data
    
    def generate_trading_signals(self, data):
        """
        Generate comprehensive trading signals
        
        Args:
            data (pandas.DataFrame): Stock data with indicators
            
        Returns:
            pandas.DataFrame: Data with trading signals
        """
        # Initialize signal columns
        data['Signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
        data['Signal_Strength'] = 0  # Signal strength from 0 to 1
        data['Signal_Reason'] = ''
        
        # Buy signals
        buy_conditions = []
        buy_reasons = []
        
        # RSI oversold
        rsi_oversold = data['RSI'] < 30
        buy_conditions.append(rsi_oversold)
        buy_reasons.append('RSI Oversold')
        
        # MACD bullish crossover
        macd_bullish = data['MACD_Bullish']
        buy_conditions.append(macd_bullish)
        buy_reasons.append('MACD Bullish')
        
        # Price above moving averages
        price_above_sma = data['Price_Above_SMA_20'] & data['Price_Above_SMA_50']
        buy_conditions.append(price_above_sma)
        buy_reasons.append('Price Above MA')
        
        # Bollinger Band bounce
        bb_bounce = (data['Close'] < data['BB_Lower']) & (data['Close'].shift(1) >= data['BB_Lower'].shift(1))
        buy_conditions.append(bb_bounce)
        buy_reasons.append('BB Bounce')
        
        # Golden cross
        golden_cross = data['SMA_20_50_Golden']
        buy_conditions.append(golden_cross)
        buy_reasons.append('Golden Cross')
        
        # Sell signals
        sell_conditions = []
        sell_reasons = []
        
        # RSI overbought
        rsi_overbought = data['RSI'] > 70
        sell_conditions.append(rsi_overbought)
        sell_reasons.append('RSI Overbought')
        
        # MACD bearish crossover
        macd_bearish = data['MACD_Bearish']
        sell_conditions.append(macd_bearish)
        sell_reasons.append('MACD Bearish')
        
        # Bollinger Band break
        bb_break = data['BB_Upper_Break']
        sell_conditions.append(bb_break)
        sell_reasons.append('BB Upper Break')
        
        # Death cross
        death_cross = data['SMA_20_50_Death']
        sell_conditions.append(death_cross)
        sell_reasons.append('Death Cross')
        
        # Calculate signal strength and final signal
        for i in range(len(data)):
            buy_score = sum([condition.iloc[i] if condition.iloc[i] == condition.iloc[i] else False 
                           for condition in buy_conditions])
            sell_score = sum([condition.iloc[i] if condition.iloc[i] == condition.iloc[i] else False 
                            for condition in sell_conditions])
            
            if buy_score > sell_score and buy_score > 0:
                data.iloc[i, data.columns.get_loc('Signal')] = 1
                data.iloc[i, data.columns.get_loc('Signal_Strength')] = min(buy_score / len(buy_conditions), 1.0)
                active_reasons = [buy_reasons[j] for j, cond in enumerate(buy_conditions) 
                                if cond.iloc[i] == cond.iloc[i] and cond.iloc[i]]
                data.iloc[i, data.columns.get_loc('Signal_Reason')] = ', '.join(active_reasons)
            elif sell_score > buy_score and sell_score > 0:
                data.iloc[i, data.columns.get_loc('Signal')] = -1
                data.iloc[i, data.columns.get_loc('Signal_Strength')] = min(sell_score / len(sell_conditions), 1.0)
                active_reasons = [sell_reasons[j] for j, cond in enumerate(sell_conditions) 
                                if cond.iloc[i] == cond.iloc[i] and cond.iloc[i]]
                data.iloc[i, data.columns.get_loc('Signal_Reason')] = ', '.join(active_reasons)
            else:
                data.iloc[i, data.columns.get_loc('Signal')] = 0
                data.iloc[i, data.columns.get_loc('Signal_Strength')] = 0.0
                data.iloc[i, data.columns.get_loc('Signal_Reason')] = 'Hold'
        
        return data
    
    def get_current_signals(self, data):
        """
        Get current trading signals and analysis
        
        Args:
            data (pandas.DataFrame): Stock data with indicators and signals
            
        Returns:
            dict: Current trading analysis
        """
        if data is None or data.empty:
            return {}
            
        latest = data.iloc[-1]
        
        analysis = {
            'current_price': latest['Close'],
            'signal': int(latest['Signal']),
            'signal_strength': latest['Signal_Strength'],
            'signal_reason': latest['Signal_Reason'],
            'rsi': latest['RSI'],
            'rsi_status': 'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral',
            'macd': latest['MACD'],
            'macd_signal': latest['MACD_Signal'],
            'macd_histogram': latest['MACD_Histogram'],
            'bb_position': 'Above Upper' if latest['Close'] > latest['BB_Upper'] else 
                          'Below Lower' if latest['Close'] < latest['BB_Lower'] else 'Within Bands',
            'trend_short': 'Bullish' if latest['Close'] > latest['SMA_20'] else 'Bearish',
            'trend_medium': 'Bullish' if latest['Close'] > latest['SMA_50'] else 'Bearish',
            'trend_long': 'Bullish' if latest['Close'] > latest['SMA_200'] else 'Bearish' if 'SMA_200' in data.columns and not pd.isna(latest['SMA_200']) else 'Unknown',
            'volume_status': 'High' if latest.get('High_Volume', False) else 'Low' if latest.get('Low_Volume', False) else 'Normal',
            'volatility': latest['Volatility_20D'] if 'Volatility_20D' in data.columns else None,
            'support_level': latest['Support_20D'] if 'Support_20D' in data.columns else None,
            'resistance_level': latest['Resistance_20D'] if 'Resistance_20D' in data.columns else None
        }
        
        return analysis
