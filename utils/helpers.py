import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta


def normalize_price(price: float, min_price: float, max_price: float) -> float:
    """
    Normalize price to 0-1 range
    
    Args:
        price: Price to normalize
        min_price: Minimum price
        max_price: Maximum price
    
    Returns:
        Normalized price
    """
    if max_price == min_price:
        return 0.5
    return (price - min_price) / (max_price - min_price)


def denormalize_price(normalized: float, min_price: float, max_price: float) -> float:
    """
    Denormalize price from 0-1 range
    
    Args:
        normalized: Normalized price
        min_price: Minimum price
        max_price: Maximum price
    
    Returns:
        Denormalized price
    """
    return normalized * (max_price - min_price) + min_price


def calculate_price_change(current: float, previous: float) -> float:
    """
    Calculate percentage price change
    
    Args:
        current: Current price
        previous: Previous price
    
    Returns:
        Percentage change
    """
    if previous == 0:
        return 0.0
    return (current - previous) / previous


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """
    Calculate current ATR value
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
    
    Returns:
        Current ATR value
    """
    tr = np.zeros(len(high))
    
    for i in range(1, len(high)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    return np.mean(tr[-period:])


def identify_support_resistance(
    prices: np.ndarray,
    window: int = 10,
    tolerance: float = 0.02
) -> Tuple[List[float], List[float]]:
    """
    Identify support and resistance levels
    
    Args:
        prices: Price array
        window: Window size for extrema detection
        tolerance: Tolerance for level clustering
    
    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    if len(prices) < window:
        return [], []
    
    support = []
    resistance = []
    
    for i in range(window, len(prices) - window):
        window_prices = prices[i-window:i+window]
        
        if prices[i] == np.min(window_prices):
            support.append(prices[i])
        elif prices[i] == np.max(window_prices):
            resistance.append(prices[i])
    
    # Cluster similar levels
    support = cluster_levels(support, tolerance)
    resistance = cluster_levels(resistance, tolerance)
    
    return support, resistance


def cluster_levels(levels: List[float], tolerance: float) -> List[float]:
    """
    Cluster similar support/resistance levels
    
    Args:
        levels: List of levels
        tolerance: Tolerance for clustering
    
    Returns:
        Clustered levels
    """
    if not levels:
        return []
    
    levels = sorted(levels)
    clustered = []
    current_cluster = [levels[0]]
    
    for level in levels[1:]:
        if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
            current_cluster.append(level)
        else:
            clustered.append(np.mean(current_cluster))
            current_cluster = [level]
    
    if current_cluster:
        clustered.append(np.mean(current_cluster))
    
    return clustered


def detect_divergence(
    prices: np.ndarray,
    indicator: np.ndarray,
    window: int = 5
) -> List[Tuple[int, str]]:
    """
    Detect price-indicator divergence
    
    Args:
        prices: Price array
        indicator: Indicator values
        window: Window size
    
    Returns:
        List of divergence points (index, type)
    """
    divergences = []
    
    for i in range(window, len(prices) - window):
        price_high = np.max(prices[i-window:i])
        price_low = np.min(prices[i-window:i])
        ind_high = np.max(indicator[i-window:i])
        ind_low = np.min(indicator[i-window:i])
        
        # Bullish divergence (price lower low, indicator higher low)
        if prices[i] < price_low and indicator[i] > ind_low:
            divergences.append((i, 'bullish'))
        
        # Bearish divergence (price higher high, indicator lower high)
        elif prices[i] > price_high and indicator[i] < ind_high:
            divergences.append((i, 'bearish'))
    
    return divergences


def get_market_regime(
    prices: np.ndarray,
    fast_ma: int = 20,
    slow_ma: int = 50
) -> str:
    """
    Determine market regime (trending/ranging)
    
    Args:
        prices: Price array
        fast_ma: Fast MA period
        slow_ma: Slow MA period
    
    Returns:
        'uptrend', 'downtrend', or 'ranging'
    """
    fast_ma_val = np.mean(prices[-fast_ma:])
    slow_ma_val = np.mean(prices[-slow_ma:])
    
    if fast_ma_val > slow_ma_val * 1.01:
        return 'uptrend'
    elif fast_ma_val < slow_ma_val * 0.99:
        return 'downtrend'
    else:
        return 'ranging'


def calculate_volatility(
    prices: np.ndarray,
    period: int = 20
) -> float:
    """
    Calculate price volatility (standard deviation of returns)
    
    Args:
        prices: Price array
        period: Period for calculation
    
    Returns:
        Volatility value
    """
    if len(prices) < period:
        return 0.0
    
    returns = np.diff(prices[-period:]) / prices[-period:-1]
    return np.std(returns)


def time_until_next_signal(
    last_signal_time: datetime,
    expected_interval: timedelta = timedelta(hours=1)
) -> timedelta:
    """
    Calculate time until next expected signal
    
    Args:
        last_signal_time: Time of last signal
        expected_interval: Expected signal interval
    
    Returns:
        Time remaining
    """
    next_signal_time = last_signal_time + expected_interval
    return next_signal_time - datetime.now()


def validate_signal(signal: dict, price: float) -> bool:
    """
    Validate signal integrity
    
    Args:
        signal: Signal dictionary
        price: Current price
    
    Returns:
        True if signal is valid
    """
    required_fields = ['timestamp', 'signal_type', 'confidence', 'price']
    
    for field in required_fields:
        if field not in signal:
            return False
    
    if signal['confidence'] < 0 or signal['confidence'] > 1:
        return False
    
    if signal['signal_type'] not in ['buy', 'sell', 'strong_buy', 'strong_sell']:
        return False
    
    return True


if __name__ == "__main__":
    # Test helpers
    prices = np.array([100, 102, 101, 103, 102, 104, 103, 105])
    
    support, resistance = identify_support_resistance(prices)
    print(f"Support levels: {support}")
    print(f"Resistance levels: {resistance}")
    print(f"Market regime: {get_market_regime(prices)}")
    print(f"Volatility: {calculate_volatility(prices):.4f}")
