#!/usr/bin/env python3
"""
Backtest Example

Demonstrates how to backtest the ZigZag prediction system
and evaluate its performance.
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
from loguru import logger

from data.loader import HFDataLoader
from indicators.zigzag_predictive import PredictiveZigZag
from indicators.technical_indicators import TechnicalIndicators
from signals.signal_generator import SignalGenerator
from signals.backtester import BackTester


def setup_logger():
    logger.remove()
    logger.add(
        "logs/backtest.log",
        level="INFO",
        rotation="500 MB"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO"
    )


def run_backtest(
    pair: str,
    timeframe: str = '15m',
    limit: int = 2000,
    initial_capital: float = 10000,
    risk_per_trade: float = 0.02
):
    """
    Run backtest on cryptocurrency pair
    
    Args:
        pair: Cryptocurrency pair
        timeframe: Timeframe
        limit: Number of candles
        initial_capital: Starting capital
        risk_per_trade: Risk per trade
    """
    logger.info(f"Running backtest on {pair} {timeframe}")
    
    # Load data
    loader = HFDataLoader()
    try:
        df = loader.fetch_pair_data(pair, timeframe=timeframe, limit=limit)
        logger.info(f"Loaded {len(df)} candles for backtest")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return
    
    # Setup predictive system
    zigzag = PredictiveZigZag(
        threshold=0.5,
        min_length=20,
        predict_bars=3,
        confidence_threshold=0.65
    )
    
    # Add predictions and indicators
    df = zigzag.calculate_with_prediction(df)
    df = TechnicalIndicators.add_indicators_to_df(df)
    
    # Generate signals
    signal_gen = SignalGenerator(zigzag, confidence_threshold=0.65)
    df = signal_gen.generate(df)
    
    # Run backtest
    backtester = BackTester(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        commission=0.001,
        slippage=0.0005
    )
    
    results = backtester.backtest(df)
    
    # Print results
    backtester.print_results(results)
    
    # Print trades
    if results.trades:
        print("Top 10 Winning Trades:")
        winning_trades = sorted([t for t in results.trades if t.profit_loss > 0], 
                                key=lambda x: x.profit_loss, reverse=True)[:10]
        for i, trade in enumerate(winning_trades, 1):
            print(f"  {i}. Entry: {trade.entry_price:.2f} -> Exit: {trade.exit_price:.2f} | "
                  f"Profit: {trade.profit_loss:.2f} ({trade.profit_loss_pct:.2%})")
        
        print("\nTop 10 Losing Trades:")
        losing_trades = sorted([t for t in results.trades if t.profit_loss < 0], 
                               key=lambda x: x.profit_loss)[:10]
        for i, trade in enumerate(losing_trades, 1):
            print(f"  {i}. Entry: {trade.entry_price:.2f} -> Exit: {trade.exit_price:.2f} | "
                  f"Loss: {trade.profit_loss:.2f} ({trade.profit_loss_pct:.2%})")
    
    return results


def compare_pairs(pairs: list, timeframe: str = '15m', limit: int = 1000):
    """
    Compare backtest results across multiple pairs
    
    Args:
        pairs: List of cryptocurrency pairs
        timeframe: Timeframe
        limit: Number of candles
    """
    logger.info(f"Comparing {len(pairs)} pairs")
    
    comparison_data = []
    
    for pair in pairs:
        logger.info(f"Processing {pair}...")
        try:
            results = run_backtest(pair, timeframe, limit)
            
            comparison_data.append({
                'Pair': pair,
                'Total_Trades': results.total_trades,
                'Win_Rate': results.win_rate,
                'Total_Return': results.total_return_pct,
                'Profit_Factor': results.profit_factor,
                'Max_Drawdown': results.max_drawdown,
                'Sharpe_Ratio': results.sharpe_ratio
            })
        except Exception as e:
            logger.error(f"Failed to backtest {pair}: {str(e)}")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Win_Rate', ascending=False)
    
    print("\n" + "="*100)
    print("PAIR COMPARISON")
    print("="*100)
    print(comparison_df.to_string(index=False))
    print("="*100 + "\n")
    
    # Save comparison
    comparison_df.to_csv('backtest_comparison.csv', index=False)
    logger.info("Comparison saved to backtest_comparison.csv")


if __name__ == "__main__":
    setup_logger()
    
    # Single pair backtest
    logger.info("Starting single pair backtest...")
    run_backtest(
        pair='BTCUSDT',
        timeframe='15m',
        limit=1500,
        initial_capital=10000,
        risk_per_trade=0.02
    )
    
    # Compare multiple pairs
    logger.info("Starting pair comparison...")
    compare_pairs(
        pairs=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
        timeframe='15m',
        limit=1000
    )
    
    logger.info("Backtest complete")
