#!/usr/bin/env python3
"""
Test script to debug data fetching issues
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_yahoo_finance():
    """Test Yahoo Finance data fetching"""
    print("Testing Yahoo Finance data fetching...")
    print("=" * 50)
    
    # Test symbols
    test_symbols = [
        'AAPL',           # US stock
        'RELIANCE.NS',    # Indian stock
        'TSLA',           # Another US stock
        'TCS.NS',         # Another Indian stock
    ]
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing symbol: {symbol}")
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Test different methods
            print(f"  📊 Fetching historical data...")
            
            # Try different periods
            periods_to_test = ['1d', '5d', '1mo', '3mo', '6mo', '1y']
            
            for period in periods_to_test:
                try:
                    data = ticker.history(period=period)
                    if not data.empty:
                        print(f"    ✅ {period}: {len(data)} rows, latest date: {data.index[-1].date()}")
                        if period == '1mo':  # Show sample data
                            print(f"       Sample: Close={data['Close'].iloc[-1]:.2f}, Volume={data['Volume'].iloc[-1]:,.0f}")
                        break
                    else:
                        print(f"    ❌ {period}: No data")
                except Exception as e:
                    print(f"    ❌ {period}: Error - {str(e)}")
            
            # Test ticker info
            try:
                info = ticker.info
                if info and 'longName' in info:
                    print(f"  📋 Company: {info.get('longName', 'N/A')}")
                    print(f"  💰 Current Price: {info.get('currentPrice', 'N/A')}")
                else:
                    print(f"  📋 Info: Limited data available")
            except Exception as e:
                print(f"  ❌ Info error: {str(e)}")
                
        except Exception as e:
            print(f"  ❌ Failed to create ticker: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")

def test_network_connectivity():
    """Test network connectivity to Yahoo Finance"""
    print("\n🌐 Testing network connectivity...")
    try:
        import requests
        response = requests.get('https://finance.yahoo.com', timeout=10)
        print(f"✅ Yahoo Finance accessible (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Network issue: {str(e)}")

def test_with_different_approach():
    """Test with different yfinance approach"""
    print("\n🔄 Testing alternative approach...")
    
    try:
        # Try downloading multiple symbols at once
        symbols = ['AAPL', 'RELIANCE.NS']
        data = yf.download(symbols, period='1mo', progress=False)
        
        if not data.empty:
            print(f"✅ Downloaded {len(data)} rows for symbols: {symbols}")
            for symbol in symbols:
                if symbol in data.columns.get_level_values(1):
                    symbol_data = data.xs(symbol, level=1, axis=1)
                    print(f"  {symbol}: {len(symbol_data)} rows")
        else:
            print("❌ No data downloaded")
            
    except Exception as e:
        print(f"❌ Alternative approach failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Yahoo Finance Data Fetching Test")
    print(f"📅 Current time: {datetime.now()}")
    
    test_network_connectivity()
    test_yahoo_finance()
    test_with_different_approach()
    
    print("\n💡 If all tests fail, possible issues:")
    print("   - Internet connectivity problems")
    print("   - Yahoo Finance API changes or restrictions")
    print("   - Firewall blocking the requests")
    print("   - Need to update yfinance library")
