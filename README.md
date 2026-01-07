# ZigZag Early Prediction System (v1)

A comprehensive cryptocurrency trend prediction system designed to solve ZigZag indicator lag problems through multi-layer prediction architecture.

## Overview

This system provides early prediction of price trends by:
- Analyzing OHLCV data from HuggingFace datasets
- Combining multiple technical indicators with machine learning
- Implementing lag-compensation algorithms
- Real-time trend prediction capabilities

## System Architecture

### 1. Data Pipeline
- **Source**: HuggingFace crypto OHLCV datasets
- **Supported Timeframes**: 15m, 1h
- **Data Format**: Parquet files for efficient processing
- **Update Frequency**: Real-time streaming capability

### 2. ZigZag Enhancement Layer
- **Traditional ZigZag**: Identifies swing highs/lows with lag
- **Predictive ZigZag**: Anticipates reversals before they form
- **Lag Compensation**: Early warning signals for trend changes

### 3. Prediction Components
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Price Action**: Support/Resistance levels, breakout detection
- **ML Models**: LSTM for sequence prediction, Random Forest for feature importance
- **Ensemble Methods**: Combines multiple predictions for reliability

### 4. Early Signal Detection
- Pre-reversal pattern recognition
- Volume-price confirmation
- Multi-timeframe analysis
- Probability-based scoring

## Project Structure

```
v1/
├── README.md
├── requirements.txt
├── config/
│   ├── config.yaml
│   └── pairs_config.json
├── data/
│   ├── loader.py
│   ├── processor.py
│   └── cache/
├── indicators/
│   ├── zigzag_base.py
│   ├── zigzag_predictive.py
│   ├── technical_indicators.py
│   └── price_action.py
├── models/
│   ├── lstm_predictor.py
│   ├── ensemble_model.py
│   └── feature_extractor.py
├── signals/
│   ├── signal_generator.py
│   ├── early_warning.py
│   └── backtester.py
├── utils/
│   ├── helpers.py
│   ├── logger.py
│   └── metrics.py
└── examples/
    ├── single_pair_analysis.py
    ├── backtest_example.py
    └── live_monitoring.py
```

## Key Features

### Lag-Free Prediction
- Predicts trend changes 2-5 candles before traditional ZigZag
- Early reversal warning system
- Support/resistance breakthrough alerts

### Multi-Layer Confirmation
- Technical indicators
- Price action patterns
- Machine learning confidence scores
- Volume confirmation

### Backtesting Framework
- Historical performance analysis
- Win rate and profit factor calculation
- Drawdown analysis
- Signal reliability metrics

### Real-Time Capabilities
- Stream data processing
- Live prediction updates
- Alert generation system

## Installation

```bash
git clone https://github.com/caizongxun/v1.git
cd v1
pip install -r requirements.txt
```

## Quick Start

```python
from data.loader import HFDataLoader
from indicators.zigzag_predictive import PredictiveZigZag
from signals.signal_generator import SignalGenerator

# Load data
loader = HFDataLoader()
df = loader.fetch_pair_data('BTCUSDT', timeframe='15m')

# Initialize predictor
zigzag = PredictiveZigZag(
    threshold=0.5,
    lookback=50,
    predict_bars=3
)

# Generate signals
predictor = SignalGenerator(zigzag)
signals = predictor.generate(df)
```

## Configuration

Edit `config/config.yaml` to customize:
- Data source paths
- Model parameters
- Indicator thresholds
- Alert settings

## HuggingFace Dataset Structure

```
Root: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/
├── klines/
│   ├── BTCUSDT/
│   │   ├── BTC_15m.parquet
│   │   └── BTC_1h.parquet
│   ├── ETHUSDT/
│   │   ├── ETH_15m.parquet
│   │   └── ETH_1h.parquet
│   └── [other pairs]/
```

## Performance Metrics

The system provides:
- Prediction accuracy on historical data
- Early signal lead time
- False signal rate
- Win/loss ratio
- Sharpe ratio approximation

## Contributing

Contributions welcome! Areas for enhancement:
- Additional timeframe support
- More ML models
- Advanced pattern recognition
- Real-time data streaming

## License

MIT License - See LICENSE file for details

## Contact

Author: caizongxun
GitHub: https://github.com/caizongxun/v1
