import numpy as np
import pandas as pd
from typing import Tuple, Dict
from loguru import logger


class TechnicalIndicators:
    """
    Collection of technical indicators for trend analysis and confirmation
    """
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index
        
        Args:
            prices: Price series
            period: RSI period
        
        Returns:
            RSI values (0-100)
        """
        if len(prices) < period + 1:
            return np.full_like(prices, np.nan, dtype=float)
        
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices, dtype=float)
        rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                up = delta
                down = 0.0
            else:
                up = 0.0
                down = -delta
            
            rs = (up + up * (period - 1) / period) / (down + down * (period - 1) / period)
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        
        return rsi
    
    @staticmethod
    def calculate_macd(
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        
        Returns:
            Tuple of (macd, signal_line, histogram)
        """
        if len(prices) < slow:
            return (
                np.full_like(prices, np.nan, dtype=float),
                np.full_like(prices, np.nan, dtype=float),
                np.full_like(prices, np.nan, dtype=float)
            )
        
        ema_fast = TechnicalIndicators._calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            period: MA period
            std_dev: Number of standard deviations
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if len(prices) < period:
            n = len(prices)
            return (
                np.full(n, np.nan, dtype=float),
                np.full(n, np.nan, dtype=float),
                np.full(n, np.nan, dtype=float)
            )
        
        middle = np.convolve(prices, np.ones(period) / period, mode='same')
        std = np.zeros_like(prices, dtype=float)
        
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1])
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
        
        Returns:
            ATR values
        """
        if len(high) < period + 1:
            return np.full_like(high, np.nan, dtype=float)
        
        tr = np.zeros_like(high, dtype=float)
        
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr[i] = max(tr1, tr2, tr3)
        
        atr = np.convolve(tr, np.ones(period) / period, mode='same')
        return atr
    
    @staticmethod
    def calculate_stochastic(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Stochastic period
            smooth_k: K smoothing
            smooth_d: D smoothing
        
        Returns:
            Tuple of (K line, D line)
        """
        if len(high) < period:
            n = len(high)
            return np.full(n, np.nan), np.full(n, np.nan)
        
        k = np.zeros_like(close, dtype=float)
        
        for i in range(period - 1, len(close)):
            lowest_low = np.min(low[i - period + 1:i + 1])
            highest_high = np.max(high[i - period + 1:i + 1])
            
            if highest_high != lowest_low:
                k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
            else:
                k[i] = 50
        
        k_smooth = TechnicalIndicators._calculate_sma(k, smooth_k)
        d_smooth = TechnicalIndicators._calculate_sma(k_smooth, smooth_d)
        
        return k_smooth, d_smooth
    
    @staticmethod
    def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average
        
        Args:
            prices: Price series
            period: EMA period
        
        Returns:
            EMA values
        """
        if len(prices) < period:
            return np.full_like(prices, np.nan, dtype=float)
        
        ema = np.zeros_like(prices, dtype=float)
        multiplier = 2.0 / (period + 1)
        
        ema[period - 1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = prices[i] * multiplier + ema[i - 1] * (1 - multiplier)
        
        ema[:period - 1] = np.nan
        return ema
    
    @staticmethod
    def _calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average
        
        Args:
            prices: Price series
            period: SMA period
        
        Returns:
            SMA values
        """
        if len(prices) < period:
            return np.full_like(prices, np.nan, dtype=float)
        
        return np.convolve(prices, np.ones(period) / period, mode='same')
    
    @staticmethod
    def add_indicators_to_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all indicators to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with indicators added
        """
        try:
            # RSI
            df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'].values, 14)
            
            # MACD
            macd, signal, hist = TechnicalIndicators.calculate_macd(df['Close'].values)
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = hist
            
            # Bollinger Bands
            upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df['Close'].values)
            df['BB_Upper'] = upper
            df['BB_Middle'] = middle
            df['BB_Lower'] = lower
            
            # ATR
            df['ATR'] = TechnicalIndicators.calculate_atr(
                df['High'].values,
                df['Low'].values,
                df['Close'].values
            )
            
            # Stochastic
            k, d = TechnicalIndicators.calculate_stochastic(
                df['High'].values,
                df['Low'].values,
                df['Close'].values
            )
            df['Stoch_K'] = k
            df['Stoch_D'] = d
            
            # Moving Averages
            df['SMA_20'] = TechnicalIndicators._calculate_sma(df['Close'].values, 20)
            df['SMA_50'] = TechnicalIndicators._calculate_sma(df['Close'].values, 50)
            df['EMA_12'] = TechnicalIndicators._calculate_ema(df['Close'].values, 12)
            df['EMA_26'] = TechnicalIndicators._calculate_ema(df['Close'].values, 26)
            
            logger.info("All indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
        
        return df


if __name__ == "__main__":
    import pandas as pd
    
    # Example
    prices = np.random.randn(100).cumsum() + 100
    
    rsi = TechnicalIndicators.calculate_rsi(prices)
    print(f"RSI shape: {rsi.shape}")
    print(f"RSI last value: {rsi[-1]:.2f}")
