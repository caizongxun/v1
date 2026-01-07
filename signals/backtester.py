import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from loguru import logger
from datetime import datetime


@dataclass
class Trade:
    entry_index: int
    entry_price: float
    entry_time: pd.Timestamp
    entry_signal: str
    exit_index: int = 0
    exit_price: float = 0.0
    exit_time: pd.Timestamp = None
    exit_reason: str = ""
    profit_loss: float = 0.0
    profit_loss_pct: float = 0.0
    risk_reward: float = 0.0


@dataclass
class BacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    total_return_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    avg_holding_bars: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class BackTester:
    """
    Backtester for evaluating trading signals
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        risk_per_trade: float = 0.02,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            risk_per_trade: Risk per trade as fraction of capital
            commission: Commission per trade
            slippage: Slippage as fraction of price
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
    
    def backtest(
        self,
        df: pd.DataFrame,
        signals_col: str = 'Signal',
        target_col: str = 'Target',
        stop_col: str = 'StopLoss'
    ) -> BacktestResult:
        """
        Run backtest on DataFrame with signals
        
        Args:
            df: DataFrame with OHLCV and signals
            signals_col: Name of signal column
            target_col: Name of target column
            stop_col: Name of stop loss column
        
        Returns:
            BacktestResult object
        """
        try:
            current_position = None
            current_capital = self.initial_capital
            
            for i in range(len(df)):
                signal = df[signals_col].iloc[i]
                current_price = df['Close'].iloc[i]
                current_time = df.index[i]
                
                # Exit signal processing
                if current_position:
                    # Check stop loss
                    if signal == -1 and current_position.entry_signal == 'buy':
                        trade = self._close_trade(current_position, i, current_price, current_time, 'signal')
                        current_capital = self._apply_trade_result(current_capital, trade)
                        self.trades.append(trade)
                        current_position = None
                    
                    elif signal == 1 and current_position.entry_signal == 'sell':
                        trade = self._close_trade(current_position, i, current_price, current_time, 'signal')
                        current_capital = self._apply_trade_result(current_capital, trade)
                        self.trades.append(trade)
                        current_position = None
                    
                    # Check stop loss price
                    stop_loss = df[stop_col].iloc[i] if stop_col in df.columns else 0
                    if stop_loss > 0:
                        if current_position.entry_signal == 'buy' and current_price <= stop_loss:
                            trade = self._close_trade(current_position, i, stop_loss, current_time, 'stop_loss')
                            current_capital = self._apply_trade_result(current_capital, trade)
                            self.trades.append(trade)
                            current_position = None
                        
                        elif current_position.entry_signal == 'sell' and current_price >= stop_loss:
                            trade = self._close_trade(current_position, i, stop_loss, current_time, 'stop_loss')
                            current_capital = self._apply_trade_result(current_capital, trade)
                            self.trades.append(trade)
                            current_position = None
                    
                    # Check target price
                    target = df[target_col].iloc[i] if target_col in df.columns else 0
                    if target > 0:
                        if current_position.entry_signal == 'buy' and current_price >= target:
                            trade = self._close_trade(current_position, i, target, current_time, 'target')
                            current_capital = self._apply_trade_result(current_capital, trade)
                            self.trades.append(trade)
                            current_position = None
                        
                        elif current_position.entry_signal == 'sell' and current_price <= target:
                            trade = self._close_trade(current_position, i, target, current_time, 'target')
                            current_capital = self._apply_trade_result(current_capital, trade)
                            self.trades.append(trade)
                            current_position = None
                
                # Entry signal processing
                if signal != 0 and not current_position:
                    if signal == 1:
                        current_position = Trade(
                            entry_index=i,
                            entry_price=current_price * (1 + self.slippage),
                            entry_time=current_time,
                            entry_signal='buy'
                        )
                    elif signal == -1:
                        current_position = Trade(
                            entry_index=i,
                            entry_price=current_price * (1 - self.slippage),
                            entry_time=current_time,
                            entry_signal='sell'
                        )
                
                # Update equity curve
                self.equity_curve.append(current_capital)
            
            # Close any open position
            if current_position:
                final_price = df['Close'].iloc[-1]
                trade = self._close_trade(current_position, len(df) - 1, final_price, df.index[-1], 'end')
                current_capital = self._apply_trade_result(current_capital, trade)
                self.trades.append(trade)
            
            return self._calculate_results(current_capital)
        
        except Exception as e:
            logger.error(f"Backtest error: {str(e)}")
            return BacktestResult()
    
    def _close_trade(
        self,
        trade: Trade,
        exit_index: int,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str
    ) -> Trade:
        """
        Close a trade and calculate profit/loss
        """
        trade.exit_index = exit_index
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.exit_reason = exit_reason
        
        if trade.entry_signal == 'buy':
            trade.profit_loss = (exit_price - trade.entry_price) * 1  # 1 contract
            trade.profit_loss_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            trade.profit_loss = (trade.entry_price - exit_price) * 1  # 1 contract
            trade.profit_loss_pct = (trade.entry_price - exit_price) / trade.entry_price
        
        return trade
    
    def _apply_trade_result(self, capital: float, trade: Trade) -> float:
        """
        Apply trade result to capital
        """
        profit = trade.profit_loss
        commission_cost = trade.entry_price * self.commission
        net_profit = profit - commission_cost
        return capital + net_profit
    
    def _calculate_results(self, final_capital: float) -> BacktestResult:
        """
        Calculate backtest statistics
        """
        result = BacktestResult()
        result.equity_curve = self.equity_curve
        result.trades = self.trades
        
        if not self.trades:
            result.total_trades = 0
            return result
        
        # Basic statistics
        result.total_trades = len(self.trades)
        result.winning_trades = sum(1 for t in self.trades if t.profit_loss > 0)
        result.losing_trades = sum(1 for t in self.trades if t.profit_loss < 0)
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        
        # Profit statistics
        wins = [t.profit_loss for t in self.trades if t.profit_loss > 0]
        losses = [t.profit_loss for t in self.trades if t.profit_loss < 0]
        
        result.total_profit = final_capital - self.initial_capital
        result.total_return_pct = result.total_profit / self.initial_capital
        
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        result.average_win = np.mean(wins) if wins else 0
        result.average_loss = np.mean(losses) if losses else 0
        
        # Drawdown
        result.max_drawdown = self._calculate_max_drawdown()
        
        # Sharpe ratio
        result.sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Average holding bars
        holding_bars = [t.exit_index - t.entry_index for t in self.trades]
        result.avg_holding_bars = np.mean(holding_bars) if holding_bars else 0
        
        return result
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        running_max = np.maximum.accumulate(self.equity_curve)
        drawdown = (np.array(self.equity_curve) - running_max) / running_max
        return float(np.min(drawdown))
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        excess_returns = returns - risk_free_rate / 252
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)
    
    def print_results(self, results: BacktestResult):
        """
        Print backtest results
        """
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Trades: {results.total_trades}")
        print(f"Winning Trades: {results.winning_trades}")
        print(f"Losing Trades: {results.losing_trades}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Total Profit: {results.total_profit:.2f}")
        print(f"Total Return: {results.total_return_pct:.2%}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Average Win: {results.average_win:.2f}")
        print(f"Average Loss: {results.average_loss:.2f}")
        print(f"Average Holding Bars: {results.avg_holding_bars:.0f}")
        print("="*50 + "\n")


if __name__ == "__main__":
    from data.loader import HFDataLoader
    from indicators.zigzag_predictive import PredictiveZigZag
    from signals.signal_generator import SignalGenerator
    
    loader = HFDataLoader()
    df = loader.fetch_pair_data('BTCUSDT', timeframe='15m', limit=1000)
    
    zigzag = PredictiveZigZag(threshold=0.5, predict_bars=3)
    generator = SignalGenerator(zigzag)
    df = generator.generate(df)
    
    backtester = BackTester(initial_capital=10000)
    results = backtester.backtest(df)
    backtester.print_results(results)
