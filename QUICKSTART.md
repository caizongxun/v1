# Quick Start Guide

## 30 Seconds to First Signal

### Step 1: Install
```bash
git clone https://github.com/caizongxun/v1.git
cd v1
pip install -r requirements.txt
```

### Step 2: Analyze
```bash
python main.py analyze --pair BTCUSDT --timeframe 15m
```

### Step 3: View Results
```
Signals Generated: 5

Recent Signals:
  2025-01-07 15:00 - strong_buy: 42500.25 (Conf: 82%)
  2025-01-07 14:45 - sell: 42480.50 (Conf: 71%)
  2025-01-07 14:30 - buy: 42510.00 (Conf: 68%)
```

## Python API Quick Start

```python
from data.loader import HFDataLoader
from indicators.zigzag_predictive import PredictiveZigZag
from signals.signal_generator import SignalGenerator

# Load data
loader = HFDataLoader()
df = loader.fetch_pair_data('BTCUSDT', timeframe='15m', limit=500)

# Predict
zigzag = PredictiveZigZag(threshold=0.5)
df = zigzag.calculate_with_prediction(df)

# Generate signals
generator = SignalGenerator(zigzag)
df = generator.generate(df)

# Get signals
for signal in generator.get_signals():
    print(f"{signal.timestamp}: {signal.signal_type}")
```

## Common Use Cases

### 1. Single Pair Analysis
```bash
python main.py analyze --pair BTCUSDT --timeframe 15m
```

### 2. Backtest
```bash
python main.py backtest --pair BTCUSDT --capital 10000
```

### 3. Multiple Pairs
```bash
python main.py compare --pairs "BTCUSDT,ETHUSDT,BNBUSDT"
```

### 4. Custom Parameters
```bash
python main.py analyze --pair BTCUSDT \
  --threshold 0.3 \
  --predict-bars 5 \
  --confidence 0.70
```

## Parameter Tuning

### Threshold (Sensitivity)
- **0.3**: Very sensitive, more signals, more false positives
- **0.5**: Balanced (default)
- **1.0**: Conservative, fewer signals, higher accuracy

### Predict Bars
- **2-3**: Short-term reversals
- **4-5**: Longer lead time
- **1**: Immediate signals

### Confidence Threshold
- **0.60**: Liberal filtering
- **0.65**: Default
- **0.75**: Strict filtering

## Output Interpretation

### Signal Types
- **strong_buy**: Confidence > 75%
- **buy**: Confidence 65-75%
- **sell**: Short confirmation
- **strong_sell**: High probability short

### Confidence Levels
- Green (75-100%): Trust this signal
- Yellow (65-75%): Use with caution
- Red (<65%): Skip or reduce size

## Backtest Metrics Explained

```
Win Rate:        % of profitable trades
Profit Factor:   Gross Profit / Gross Loss
Sharpe Ratio:    Risk-adjusted returns
Max Drawdown:    Largest peak-to-trough decline
Avg Win/Loss:    Average profit per winning/losing trade
```

## Troubleshooting

### No signals generated
```
Increase --threshold to make it more sensitive:
python main.py analyze --pair BTCUSDT --threshold 0.3
```

### Too many false signals
```
Increase --confidence requirement:
python main.py analyze --pair BTCUSDT --confidence 0.75
```

### Data loading slow
```
Reduce data size:
python main.py analyze --pair BTCUSDT --limit 500
```

## Next Steps

1. Read INSTALLATION.md for advanced setup
2. Check ZIGZAG_LAG_SOLUTION.md for technical details
3. Review examples/ directory for more scripts
4. Customize config/config.yaml for your needs
5. Backtest on your preferred pairs

## Key Features Recap

ZigZag Early Prediction System provides:

✓ Early reversal detection (2-5 bars ahead)
✓ Multi-layer confirmation
✓ Technical indicator integration
✓ Comprehensive backtesting
✓ HuggingFace dataset support
✓ Real-time capability
✓ Easy customization
✓ Production-ready code

## Getting Help

- Check logs: `cat logs/zigzag_system.log`
- Verbose mode: `python main.py analyze --pair BTCUSDT --verbose`
- GitHub Issues: Report bugs or ask questions
- Documentation: Read INSTALLATION.md and ZIGZAG_LAG_SOLUTION.md

## Performance Tips

1. Use cached data when possible
2. Reduce --limit for faster analysis
3. Run backtests with --limit 1000 initially
4. Monitor logs for errors
5. Test parameters on different pairs

Happy trading!
