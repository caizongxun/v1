#!/usr/bin/env python3
"""
Single Pair ZigZag Analysis Example

Demonstrates how to load data, apply predictive ZigZag,
generate signals, and visualize results for a single cryptocurrency pair.
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from data.loader import HFDataLoader
from indicators.zigzag_predictive import PredictiveZigZag
from indicators.technical_indicators import TechnicalIndicators
from signals.signal_generator import SignalGenerator


def setup_logger():
    """
    Configure logging
    """
    logger.remove()
    logger.add(
        "logs/analysis.log",
        level="INFO",
        rotation="500 MB"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO"
    )


def analyze_pair(
    pair: str,
    timeframe: str = '15m',
    limit: int = 1000,
    threshold: float = 0.5,
    predict_bars: int = 3
):
    """
    Analyze a single cryptocurrency pair
    
    Args:
        pair: Cryptocurrency pair (e.g., 'BTCUSDT')
        timeframe: Timeframe ('15m' or '1h')
        limit: Number of candles to analyze
        threshold: ZigZag threshold percentage
        predict_bars: Bars to predict ahead
    """
    logger.info(f"Starting analysis for {pair} {timeframe}")
    
    # Load data
    loader = HFDataLoader()
    try:
        df = loader.fetch_pair_data(pair, timeframe=timeframe, limit=limit)
        logger.info(f"Loaded {len(df)} candles")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return
    
    # Initialize predictive ZigZag
    zigzag = PredictiveZigZag(
        threshold=threshold,
        min_length=20,
        predict_bars=predict_bars,
        confidence_threshold=0.65
    )
    
    # Calculate predictions
    df = zigzag.calculate_with_prediction(df)
    
    # Add technical indicators
    df = TechnicalIndicators.add_indicators_to_df(df)
    
    # Generate signals
    signal_gen = SignalGenerator(zigzag, confidence_threshold=0.65)
    df = signal_gen.generate(df)
    
    # Display results
    print("\n" + "="*80)
    print(f"ANALYSIS RESULTS - {pair} {timeframe}")
    print("="*80)
    
    # Swing points
    swing_points = zigzag.get_swing_points(df)
    print(f"\nSwing Points Found: {len(swing_points)}")
    if swing_points:
        for i, point in enumerate(swing_points[-5:]):
            print(f"  {i+1}. {point['timestamp']} - {point['direction'].upper()} at {point['price']:.2f}")
    
    # Early signals
    signals = signal_gen.get_signals()
    print(f"\nEarly Signals Generated: {len(signals)}")
    if signals:
        for signal in signals[-5:]:
            print(f"  {signal.timestamp} - {signal.signal_type.upper()} at {signal.price:.2f}")
            print(f"    Confidence: {signal.confidence:.2%}")
            print(f"    Target: {signal.target:.2f} | Stop: {signal.stop_loss:.2f}")
    
    # Current trend
    trend = zigzag.get_current_trend()
    print(f"\nCurrent Trend: {trend.upper()}")
    
    # Support/Resistance
    support, resistance = zigzag.get_support_resistance()
    print(f"\nSupport/Resistance:")
    print(f"  Support: {support:.2f}")
    print(f"  Resistance: {resistance:.2f}")
    
    # Latest statistics
    print(f"\nLatest Data (Last 5 candles):")
    print(df[['Close', 'RSI', 'MACD_Hist', 'ATR', 'Signal_Type', 'Confidence']].tail().to_string())
    
    # Save results
    output_file = f"analysis_{pair}_{timeframe}.csv"
    df.to_csv(output_file)
    logger.info(f"Results saved to {output_file}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    setup_logger()
    
    # Analyze Bitcoin
    analyze_pair(
        pair='BTCUSDT',
        timeframe='15m',
        limit=500,
        threshold=0.5,
        predict_bars=3
    )
    
    # Analyze Ethereum
    analyze_pair(
        pair='ETHUSDT',
        timeframe='15m',
        limit=500,
        threshold=0.5,
        predict_bars=3
    )
    
    logger.info("Analysis complete")
