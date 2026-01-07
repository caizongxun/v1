# ZigZag Early Prediction System - Installation Guide

## System Requirements

Python 3.8 or higher
Minimum 4GB RAM recommended
Internet connection for HuggingFace dataset access

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/caizongxun/v1.git
cd v1
```

### 2. Create Virtual Environment

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation fails, try:
```bash
pip install tensorflow --no-cache-dir
```

### 4. Verify Installation

```bash
python -c "import pandas; import numpy; print('Installation successful!')"
```

## Configuration

### 1. Create Data Directory

```bash
mkdir -p data/cache
mkdir -p logs
```

### 2. Configure Parameters

Edit `config/config.yaml`:

```yaml
indicators:
  zigzag:
    threshold: 0.5              # Adjust sensitivity (0.3-1.0)
    predict_bars: 3             # How many bars ahead to predict
    confidence_threshold: 0.65   # Minimum confidence for signals

models:
  ensemble:
    weights:
      zigzag: 0.3               # ZigZag weight
      technical: 0.3            # Technical indicators weight
      lstm: 0.4                 # ML model weight
```

### 3. Set HuggingFace Credentials (Optional)

```bash
huggingface-cli login
```

Or set environment variable:
```bash
export HF_TOKEN=your_token_here
```

## Quick Start

### 1. Analyze Single Pair

```bash
python main.py analyze --pair BTCUSDT --timeframe 15m
```

### 2. Backtest Single Pair

```bash
python main.py backtest --pair BTCUSDT --capital 10000
```

### 3. Compare Multiple Pairs

```bash
python main.py compare --pairs "BTCUSDT,ETHUSDT,BNBUSDT" --timeframe 1h
```

## Python API Usage

### Example: Basic Analysis

```python
from data.loader import HFDataLoader
from indicators.zigzag_predictive import PredictiveZigZag
from signals.signal_generator import SignalGenerator

# Load data
loader = HFDataLoader()
df = loader.fetch_pair_data('BTCUSDT', timeframe='15m', limit=500)

# Initialize predictor
zigzag = PredictiveZigZag(threshold=0.5, predict_bars=3)
df = zigzag.calculate_with_prediction(df)

# Generate signals
generator = SignalGenerator(zigzag)
df = generator.generate(df)

# Get signals
signals = generator.get_signals()
for signal in signals:
    print(f"{signal.timestamp}: {signal.signal_type} confidence={signal.confidence:.2%}")
```

### Example: Backtesting

```python
from signals.backtester import BackTester

# Run backtest
backtester = BackTester(initial_capital=10000, risk_per_trade=0.02)
results = backtester.backtest(df)

# Print results
backtester.print_results(results)
```

## Troubleshooting

### Issue: Import Errors

**Solution:** Ensure virtual environment is activated and all dependencies installed:
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: HuggingFace Download Fails

**Solution:** Check internet connection and HF token:
```bash
huggingface-cli repo info zongowo111/v2-crypto-ohlcv-data
```

### Issue: Out of Memory

**Solution:** Reduce limit parameter:
```bash
python main.py analyze --pair BTCUSDT --limit 500
```

### Issue: TensorFlow GPU Support

**GPU Installation:**
```bash
pip install tensorflow[and-cuda]
```

## Performance Optimization

### 1. Cache Data Locally

```bash
data/
cache/
  BTCUSDT_15m.parquet
  ETHUSDT_15m.parquet
```

### 2. Parallel Processing

Edit `config.yaml` for multi-threading.

### 3. Batch Analysis

```python
pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
data = loader.get_multiple_pairs(pairs, timeframe='15m')
```

## Advanced Configuration

### Custom Thresholds

```python
zigzag = PredictiveZigZag(
    threshold=0.3,           # More sensitive
    predict_bars=5,          # Predict further ahead
    confidence_threshold=0.7  # Higher confidence requirement
)
```

### Different Timeframes

Supported timeframes from HuggingFace dataset:
- 15m (15 minutes)
- 1h (1 hour)

### Custom Signal Filters

```python
generator = SignalGenerator(
    zigzag,
    rsi_threshold=(25, 75),      # Custom RSI levels
    confidence_threshold=0.75,   # Strict confidence
    multi_timeframe=True         # Check multiple timeframes
)
```

## Monitoring and Logging

### View Logs

```bash
tail -f logs/zigzag_system.log
```

### Debug Mode

```bash
python main.py analyze --pair BTCUSDT --verbose
```

## Next Steps

1. Review examples in `examples/` directory
2. Customize parameters in `config/config.yaml`
3. Backtest on your preferred pairs
4. Integrate with trading system

## Support

For issues and questions:
- Check GitHub Issues
- Review documentation
- Inspect logs for errors

## License

MIT License - See LICENSE file
