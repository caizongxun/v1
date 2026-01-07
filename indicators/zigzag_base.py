import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
from loguru import logger


@dataclass
class ZigZagPoint:
    index: int
    price: float
    direction: str  # 'high' or 'low'
    bar_count: int = 0
    confirmed: bool = False
    timestamp: pd.Timestamp = None


class ZigZagBase:
    """
    Traditional ZigZag indicator implementation
    Identifies significant swing highs and lows
    """
    
    def __init__(self, threshold: float = 0.5, min_length: int = 20):
        """
        Initialize ZigZag indicator
        
        Args:
            threshold: Percentage threshold for reversal (0-100)
            min_length: Minimum bars between zigzag points
        """
        self.threshold = threshold / 100.0
        self.min_length = min_length
        self.points: List[ZigZagPoint] = []
    
    def calculate(self, df: pd.DataFrame, use_hl2: bool = True) -> pd.DataFrame:
        """
        Calculate ZigZag indicator
        
        Args:
            df: DataFrame with OHLC data
            use_hl2: Use (High+Low)/2 instead of High/Low
        
        Returns:
            DataFrame with ZigZag values
        """
        if len(df) < self.min_length:
            logger.warning(f"Insufficient data: {len(df)} < {self.min_length}")
            df['ZigZag'] = np.nan
            return df
        
        try:
            if use_hl2:
                highs = (df['High'] + df['Low']) / 2
                lows = (df['High'] + df['Low']) / 2
            else:
                highs = df['High']
                lows = df['Low']
            
            zigzag_values = self._calculate_zigzag(highs, lows)
            df['ZigZag'] = zigzag_values
            
            return df
        
        except Exception as e:
            logger.error(f"Error calculating ZigZag: {str(e)}")
            df['ZigZag'] = np.nan
            return df
    
    def _calculate_zigzag(self, highs: pd.Series, lows: pd.Series) -> np.ndarray:
        """
        Core ZigZag calculation algorithm
        
        Args:
            highs: High prices
            lows: Low prices
        
        Returns:
            Array of ZigZag values
        """
        n = len(highs)
        zigzag = np.full(n, np.nan)
        
        if n < self.min_length:
            return zigzag
        
        # Initialize
        last_high = highs.iloc[0]
        last_low = lows.iloc[0]
        last_zigzag_idx = 0
        last_zigzag_price = highs.iloc[0]
        trend = 0  # 0: unknown, 1: up, -1: down
        
        # Store points for reference
        self.points = []
        self.points.append(ZigZagPoint(
            index=0,
            price=last_zigzag_price,
            direction='high' if trend > 0 else 'low',
            timestamp=highs.index[0]
        ))
        
        for i in range(self.min_length, n):
            current_high = highs.iloc[i]
            current_low = lows.iloc[i]
            
            if trend == 0:
                # Determine initial trend
                if current_high > last_zigzag_price * (1 + self.threshold):
                    trend = 1
                    last_low = lows.iloc[i]
                elif current_low < last_zigzag_price * (1 - self.threshold):
                    trend = -1
                    last_high = highs.iloc[i]
            
            elif trend == 1:
                # Uptrend
                if current_high > last_high:
                    last_high = current_high
                    last_zigzag_idx = i
                    last_zigzag_price = current_high
                    zigzag[i] = current_high
                
                # Check for reversal
                if current_low < last_high * (1 - self.threshold):
                    trend = -1
                    last_low = current_low
                    zigzag[last_zigzag_idx] = last_high
                    
                    self.points.append(ZigZagPoint(
                        index=last_zigzag_idx,
                        price=last_high,
                        direction='high',
                        bar_count=last_zigzag_idx - self.points[-1].index,
                        timestamp=highs.index[last_zigzag_idx],
                        confirmed=True
                    ))
            
            else:  # trend == -1
                # Downtrend
                if current_low < last_low:
                    last_low = current_low
                    last_zigzag_idx = i
                    last_zigzag_price = current_low
                    zigzag[i] = current_low
                
                # Check for reversal
                if current_high > last_low * (1 + self.threshold):
                    trend = 1
                    last_high = current_high
                    zigzag[last_zigzag_idx] = last_low
                    
                    self.points.append(ZigZagPoint(
                        index=last_zigzag_idx,
                        price=last_low,
                        direction='low',
                        bar_count=last_zigzag_idx - self.points[-1].index,
                        timestamp=highs.index[last_zigzag_idx],
                        confirmed=True
                    ))
        
        return zigzag
    
    def get_swing_points(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract swing high and low points
        
        Args:
            df: DataFrame with ZigZag calculated
        
        Returns:
            List of swing point dictionaries
        """
        if 'ZigZag' not in df.columns:
            self.calculate(df)
        
        swing_points = []
        for point in self.points:
            swing_points.append({
                'index': point.index,
                'price': point.price,
                'direction': point.direction,
                'timestamp': point.timestamp,
                'bar_count': point.bar_count,
                'confirmed': point.confirmed
            })
        
        return swing_points
    
    def get_last_point(self) -> ZigZagPoint:
        """
        Get the last ZigZag point
        
        Returns:
            Last ZigZagPoint or None
        """
        return self.points[-1] if self.points else None
    
    def get_current_trend(self) -> str:
        """
        Determine current trend direction
        
        Returns:
            'uptrend', 'downtrend', or 'unknown'
        """
        if len(self.points) < 2:
            return 'unknown'
        
        last_direction = self.points[-1].direction
        if last_direction == 'high':
            return 'downtrend'
        elif last_direction == 'low':
            return 'uptrend'
        else:
            return 'unknown'
    
    def get_support_resistance(self, lookback: int = 5) -> Tuple[float, float]:
        """
        Get current support and resistance levels
        
        Args:
            lookback: Number of recent points to consider
        
        Returns:
            Tuple of (support, resistance)
        """
        if len(self.points) < 2:
            return (None, None)
        
        recent_points = self.points[-lookback:]
        highs = [p.price for p in recent_points if p.direction == 'high']
        lows = [p.price for p in recent_points if p.direction == 'low']
        
        support = min(lows) if lows else None
        resistance = max(highs) if highs else None
        
        return (support, resistance)
    
    def is_reversal_pending(self, current_price: float, margin: float = 0.01) -> bool:
        """
        Check if reversal is pending based on current price
        
        Args:
            current_price: Current market price
            margin: Percentage margin for reversal detection
        
        Returns:
            True if reversal appears pending
        """
        if len(self.points) < 2:
            return False
        
        last_point = self.points[-1]
        current_trend = self.get_current_trend()
        
        if current_trend == 'uptrend':
            # Check if price is approaching last resistance
            distance = (last_point.price - current_price) / current_price
            return distance < margin
        else:
            # Check if price is approaching last support
            distance = (current_price - last_point.price) / last_point.price
            return distance < margin


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    data = yf.download('BTC-USD', period='3mo', interval='1d')
    
    zigzag = ZigZagBase(threshold=2.0)
    data = zigzag.calculate(data)
    
    points = zigzag.get_swing_points(data)
    print(f"Found {len(points)} swing points")
    print(zigzag.get_current_trend())
    print(zigzag.get_support_resistance())
