from data_loader import StockDataLoader

# Test the updated data loader
loader = StockDataLoader()

# Test with RELIANCE.NS
print("Testing RELIANCE.NS...")
data = loader.fetch_stock_data('RELIANCE.NS', period='1mo')

if data is not None:
    print(f"✅ Data shape: {data.shape}")
    print(f"✅ Latest close: {data['Close'].iloc[-1]:.2f}")
    print(f"✅ Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"✅ Columns: {list(data.columns)}")
else:
    print("❌ No data fetched")

# Test with Apple
print("\nTesting AAPL...")
data2 = loader.fetch_stock_data('AAPL', period='1mo')

if data2 is not None:
    print(f"✅ Data shape: {data2.shape}")
    print(f"✅ Latest close: {data2['Close'].iloc[-1]:.2f}")
else:
    print("❌ No data fetched")
