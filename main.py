#!/usr/bin/env python3
"""
ZigZag Early Prediction System - Main Entry Point

This script serves as the main entry point for the complete ZigZag
prediction system with lag-free cryptocurrency trend analysis.
"""

import argparse
import sys
from pathlib import Path
from typing import List
from datetime import datetime
from loguru import logger

from data.loader import HFDataLoader
from indicators.zigzag_predictive import PredictiveZigZag
from indicators.technical_indicators import TechnicalIndicators
from signals.signal_generator import SignalGenerator
from signals.backtester import BackTester
from utils.logger import configure_logger


def setup_logging(verbose: bool = False):
    """
    Setup logging system
    """
    level = "DEBUG" if verbose else "INFO"
    configure_logger(level=level)


def analyze_single_pair(args):
    """
    Analyze a single cryptocurrency pair
    """
    logger.info(f"Analyzing {args.pair} on {args.timeframe} timeframe")
    
    loader = HFDataLoader()
    df = loader.fetch_pair_data(
        args.pair,
        timeframe=args.timeframe,
        limit=args.limit
    )
    
    zigzag = PredictiveZigZag(
        threshold=args.threshold,
        predict_bars=args.predict_bars,
        confidence_threshold=args.confidence
    )
    
    df = zigzag.calculate_with_prediction(df)
    df = TechnicalIndicators.add_indicators_to_df(df)
    
    signal_gen = SignalGenerator(
        zigzag,
        confidence_threshold=args.confidence
    )
    df = signal_gen.generate(df)
    
    # Print results
    print("\n" + "="*80)
    print(f"ZIGZAG ANALYSIS - {args.pair} {args.timeframe}")
    print("="*80)
    
    signals = signal_gen.get_signals()
    print(f"\nSignals Generated: {len(signals)}")
    
    if signals:
        print("\nRecent Signals:")
        for sig in signals[-5:]:
            print(f"  {sig.timestamp} - {sig.signal_type}: {sig.price:.2f} "
                  f"(Conf: {sig.confidence:.2%})")
    
    trend = zigzag.get_current_trend()
    print(f"\nCurrent Trend: {trend.upper()}")
    
    support, resistance = zigzag.get_support_resistance()
    print(f"Support: {support:.2f} | Resistance: {resistance:.2f}")
    
    # Save analysis
    output_file = f"analysis_{args.pair}_{args.timeframe}.csv"
    df.to_csv(output_file)
    logger.info(f"Analysis saved to {output_file}")
    
    print("="*80 + "\n")


def backtest_pair(args):
    """
    Backtest on a single pair
    """
    logger.info(f"Backtesting {args.pair} on {args.timeframe} timeframe")
    
    loader = HFDataLoader()
    df = loader.fetch_pair_data(
        args.pair,
        timeframe=args.timeframe,
        limit=args.limit
    )
    
    zigzag = PredictiveZigZag(
        threshold=args.threshold,
        predict_bars=args.predict_bars,
        confidence_threshold=args.confidence
    )
    
    df = zigzag.calculate_with_prediction(df)
    df = TechnicalIndicators.add_indicators_to_df(df)
    
    signal_gen = SignalGenerator(
        zigzag,
        confidence_threshold=args.confidence
    )
    df = signal_gen.generate(df)
    
    backtester = BackTester(
        initial_capital=args.capital,
        risk_per_trade=args.risk
    )
    
    results = backtester.backtest(df)
    backtester.print_results(results)
    
    # Save backtest results
    trades_df = pd.DataFrame([
        {
            'Entry': t.entry_time,
            'Exit': t.exit_time,
            'Entry_Price': t.entry_price,
            'Exit_Price': t.exit_price,
            'Profit_Loss': t.profit_loss,
            'Return_Pct': t.profit_loss_pct
        }
        for t in results.trades
    ])
    
    trades_file = f"backtest_{args.pair}_{args.timeframe}.csv"
    trades_df.to_csv(trades_file, index=False)
    logger.info(f"Backtest results saved to {trades_file}")


def backtest_multiple(args):
    """
    Backtest multiple pairs and compare
    """
    pairs = args.pairs.split(',')
    logger.info(f"Backtesting {len(pairs)} pairs")
    
    results_list = []
    
    for pair in pairs:
        pair = pair.strip()
        logger.info(f"Processing {pair}...")
        
        try:
            loader = HFDataLoader()
            df = loader.fetch_pair_data(
                pair,
                timeframe=args.timeframe,
                limit=args.limit
            )
            
            zigzag = PredictiveZigZag(
                threshold=args.threshold,
                predict_bars=args.predict_bars,
                confidence_threshold=args.confidence
            )
            
            df = zigzag.calculate_with_prediction(df)
            df = TechnicalIndicators.add_indicators_to_df(df)
            
            signal_gen = SignalGenerator(zigzag, confidence_threshold=args.confidence)
            df = signal_gen.generate(df)
            
            backtester = BackTester(
                initial_capital=args.capital,
                risk_per_trade=args.risk
            )
            
            results = backtester.backtest(df)
            
            results_list.append({
                'Pair': pair,
                'Trades': results.total_trades,
                'Win_Rate': f"{results.win_rate:.2%}",
                'Return': f"{results.total_return_pct:.2%}",
                'Profit_Factor': f"{results.profit_factor:.2f}",
                'Sharpe': f"{results.sharpe_ratio:.2f}",
                'Drawdown': f"{results.max_drawdown:.2%}"
            })
        
        except Exception as e:
            logger.error(f"Error backtesting {pair}: {str(e)}")
    
    # Print comparison
    import pandas as pd
    comparison_df = pd.DataFrame(results_list)
    print("\n" + "="*100)
    print("BACKTEST COMPARISON")
    print("="*100)
    print(comparison_df.to_string(index=False))
    print("="*100 + "\n")
    
    # Save comparison
    comparison_df.to_csv('backtest_comparison.csv', index=False)
    logger.info("Comparison saved to backtest_comparison.csv")


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(
        description="ZigZag Early Prediction System for Cryptocurrency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py analyze --pair BTCUSDT
  python main.py backtest --pair BTCUSDT --capital 10000
  python main.py compare --pairs "BTCUSDT,ETHUSDT,BNBUSDT" --timeframe 1h
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single pair')
    analyze_parser.add_argument('--pair', required=True, help='Cryptocurrency pair')
    analyze_parser.add_argument('--timeframe', default='15m', help='Timeframe (15m, 1h)')
    analyze_parser.add_argument('--limit', type=int, default=1000, help='Number of candles')
    analyze_parser.add_argument('--threshold', type=float, default=0.5, help='ZigZag threshold')
    analyze_parser.add_argument('--predict-bars', type=int, default=3, help='Bars to predict')
    analyze_parser.add_argument('--confidence', type=float, default=0.65, help='Min confidence')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest a single pair')
    backtest_parser.add_argument('--pair', required=True, help='Cryptocurrency pair')
    backtest_parser.add_argument('--timeframe', default='15m', help='Timeframe')
    backtest_parser.add_argument('--limit', type=int, default=2000, help='Number of candles')
    backtest_parser.add_argument('--threshold', type=float, default=0.5, help='ZigZag threshold')
    backtest_parser.add_argument('--predict-bars', type=int, default=3, help='Bars to predict')
    backtest_parser.add_argument('--confidence', type=float, default=0.65, help='Min confidence')
    backtest_parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    backtest_parser.add_argument('--risk', type=float, default=0.02, help='Risk per trade')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple pairs')
    compare_parser.add_argument('--pairs', required=True, help='Pairs (comma-separated)')
    compare_parser.add_argument('--timeframe', default='15m', help='Timeframe')
    compare_parser.add_argument('--limit', type=int, default=1000, help='Number of candles')
    compare_parser.add_argument('--threshold', type=float, default=0.5, help='ZigZag threshold')
    compare_parser.add_argument('--predict-bars', type=int, default=3, help='Bars to predict')
    compare_parser.add_argument('--confidence', type=float, default=0.65, help='Min confidence')
    compare_parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    compare_parser.add_argument('--risk', type=float, default=0.02, help='Risk per trade')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging(args.verbose)
    
    try:
        if args.command == 'analyze':
            analyze_single_pair(args)
        elif args.command == 'backtest':
            backtest_pair(args)
        elif args.command == 'compare':
            backtest_multiple(args)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    import pandas as pd
    main()
