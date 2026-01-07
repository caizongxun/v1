import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from indicators.zigzag_predictive import PredictiveZigZag
from indicators.technical_indicators import TechnicalIndicators


@dataclass
class Signal:
    index: int
    timestamp: pd.Timestamp
    signal_type: str  # 'buy', 'sell', 'strong_buy', 'strong_sell'
    price: float
    confidence: float
    components: Dict
    target: float
    stop_loss: float
    bars_ahead: int = 0


class SignalGenerator:
    """
    Generates trading signals by combining multiple prediction sources
    """
    
    def __init__(
        self,
        zigzag: Optional[PredictiveZigZag] = None,
        rsi_threshold: Tuple[float, float] = (30, 70),
        confidence_threshold: float = 0.65,
        multi_timeframe: bool = True
    ):
        """
        Initialize signal generator
        
        Args:
            zigzag: PredictiveZigZag instance
            rsi_threshold: (oversold, overbought) levels
            confidence_threshold: Minimum confidence for signal
            multi_timeframe: Use multiple timeframes
        """
        self.zigzag = zigzag or PredictiveZigZag()
        self.rsi_threshold = rsi_threshold
        self.confidence_threshold = confidence_threshold
        self.multi_timeframe = multi_timeframe
        self.signals: List[Signal] = []
        self.last_signal_idx = -1
    
    def generate(self, df: pd.DataFrame, min_bars_between: int = 2) -> pd.DataFrame:
        """
        Generate signals on DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            min_bars_between: Minimum bars between signals
        
        Returns:
            DataFrame with signals added
        """
        try:
            # Calculate zigzag predictions
            df = self.zigzag.calculate_with_prediction(df)
            
            # Add technical indicators
            df = TechnicalIndicators.add_indicators_to_df(df)
            
            # Initialize signal columns
            df['Signal'] = 0
            df['Signal_Type'] = ''
            df['Confidence'] = 0.0
            df['Target'] = 0.0
            df['StopLoss'] = 0.0
            
            # Generate signals
            for i in range(len(df)):
                if self.last_signal_idx >= 0 and i - self.last_signal_idx < min_bars_between:
                    continue
                
                signal = self._generate_signal_at_index(df, i)
                if signal:
                    self.signals.append(signal)
                    self.last_signal_idx = i
                    
                    df.loc[df.index[i], 'Signal'] = 1 if 'buy' in signal.signal_type else -1
                    df.loc[df.index[i], 'Signal_Type'] = signal.signal_type
                    df.loc[df.index[i], 'Confidence'] = signal.confidence
                    df.loc[df.index[i], 'Target'] = signal.target
                    df.loc[df.index[i], 'StopLoss'] = signal.stop_loss
            
            logger.info(f"Generated {len(self.signals)} signals")
            return df
        
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return df
    
    def _generate_signal_at_index(self, df: pd.DataFrame, idx: int) -> Optional[Signal]:
        """
        Generate signal at specific index
        
        Args:
            df: DataFrame
            idx: Index position
        
        Returns:
            Signal object or None
        """
        try:
            if idx < 30:
                return None
            
            row = df.iloc[idx]
            
            # Collect components
            components = {}
            
            # Component 1: ZigZag prediction
            zigzag_score = row.get('Early_Signal', 0)
            zigzag_confidence = row.get('Reversal_Confidence', 0)
            components['zigzag'] = zigzag_score
            components['zigzag_conf'] = zigzag_confidence
            
            # Component 2: RSI
            rsi = row.get('RSI', 50)
            rsi_score = self._analyze_rsi(rsi)
            components['rsi'] = rsi
            components['rsi_score'] = rsi_score
            
            # Component 3: MACD
            macd_hist = row.get('MACD_Hist', 0)
            macd_score = 1 if macd_hist > 0 else -1 if macd_hist < 0 else 0
            components['macd_hist'] = macd_hist
            components['macd_score'] = macd_score
            
            # Component 4: Bollinger Bands
            close = row['Close']
            bb_lower = row.get('BB_Lower', close)
            bb_upper = row.get('BB_Upper', close)
            bb_score = self._analyze_bollinger(close, bb_lower, bb_upper)
            components['bb_score'] = bb_score
            
            # Component 5: Volume
            volume_score = row.get('Volume', 0) > 0
            components['volume_score'] = volume_score
            
            # Calculate confidence
            confidence = self._calculate_signal_confidence(components)
            
            if confidence < self.confidence_threshold:
                return None
            
            # Determine signal direction
            signal_direction = self._determine_direction(
                zigzag_score,
                rsi_score,
                macd_score,
                bb_score
            )
            
            if signal_direction == 0:
                return None
            
            # Calculate targets and stops
            atr = row.get('ATR', (row['High'] - row['Low']))
            target, stop_loss = self._calculate_targets(close, signal_direction, atr)
            
            # Determine signal type (strong vs normal)
            signal_type = self._get_signal_type(signal_direction, confidence)
            
            bars_ahead = int(row.get('Reversal_Confidence', 1))
            
            signal = Signal(
                index=idx,
                timestamp=df.index[idx],
                signal_type=signal_type,
                price=close,
                confidence=confidence,
                components=components,
                target=target,
                stop_loss=stop_loss,
                bars_ahead=bars_ahead
            )
            
            return signal
        
        except Exception as e:
            logger.debug(f"Signal generation error at index {idx}: {str(e)}")
            return None
    
    def _analyze_rsi(self, rsi: float) -> float:
        """
        Analyze RSI for signal generation
        
        Returns: Score between -1 and 1
        """
        if pd.isna(rsi):
            return 0.0
        
        oversold, overbought = self.rsi_threshold
        
        if rsi < oversold:
            return (oversold - rsi) / oversold  # Oversold, bullish
        elif rsi > overbought:
            return -(rsi - overbought) / (100 - overbought)  # Overbought, bearish
        else:
            return 0.0
    
    def _analyze_bollinger(self, close: float, lower: float, upper: float) -> float:
        """
        Analyze Bollinger Band position
        
        Returns: Score between -1 and 1
        """
        if pd.isna(lower) or pd.isna(upper):
            return 0.0
        
        total_width = upper - lower
        if total_width == 0:
            return 0.0
        
        position = (close - lower) / total_width
        
        if position < 0.2:
            return 0.8  # Near lower band, bullish
        elif position > 0.8:
            return -0.8  # Near upper band, bearish
        else:
            return 0.0
    
    def _calculate_signal_confidence(self, components: Dict) -> float:
        """
        Calculate overall signal confidence
        
        Args:
            components: Signal components dictionary
        
        Returns:
            Confidence score 0-1
        """
        weights = {
            'zigzag_conf': 0.35,
            'rsi_score': 0.20,
            'macd_score': 0.20,
            'bb_score': 0.15,
            'volume_score': 0.10
        }
        
        confidence = 0.0
        for key, weight in weights.items():
            value = abs(components.get(key, 0))
            confidence += value * weight
        
        return np.clip(confidence, 0, 1)
    
    def _determine_direction(self, zigzag_score: float, rsi_score: float, macd_score: float, bb_score: float) -> int:
        """
        Determine signal direction based on components
        
        Returns: 1 (buy), -1 (sell), or 0 (no signal)
        """
        signals = [zigzag_score, rsi_score, macd_score, bb_score]
        
        bullish_count = sum(1 for s in signals if s > 0)
        bearish_count = sum(1 for s in signals if s < 0)
        
        if bullish_count > bearish_count:
            return 1
        elif bearish_count > bullish_count:
            return -1
        else:
            return 0
    
    def _get_signal_type(self, direction: int, confidence: float) -> str:
        """
        Determine signal type based on direction and confidence
        
        Returns: 'buy', 'strong_buy', 'sell', or 'strong_sell'
        """
        is_strong = confidence > 0.75
        
        if direction == 1:
            return 'strong_buy' if is_strong else 'buy'
        else:
            return 'strong_sell' if is_strong else 'sell'
    
    def _calculate_targets(self, entry_price: float, direction: int, atr: float) -> Tuple[float, float]:
        """
        Calculate target and stop loss levels
        
        Args:
            entry_price: Entry price
            direction: 1 for buy, -1 for sell
            atr: Average True Range
        
        Returns:
            Tuple of (target, stop_loss)
        """
        risk_reward_ratio = 2.0
        
        if direction == 1:
            # Buy signal
            stop_loss = entry_price - (atr * 1.0)
            target = entry_price + (atr * risk_reward_ratio)
        else:
            # Sell signal
            stop_loss = entry_price + (atr * 1.0)
            target = entry_price - (atr * risk_reward_ratio)
        
        return target, stop_loss
    
    def get_signals(self) -> List[Signal]:
        """
        Get all generated signals
        
        Returns:
            List of Signal objects
        """
        return self.signals
    
    def get_latest_signal(self) -> Optional[Signal]:
        """
        Get most recent signal
        
        Returns:
            Latest Signal or None
        """
        return self.signals[-1] if self.signals else None
    
    def filter_signals(self, min_confidence: float = 0.65) -> List[Signal]:
        """
        Filter signals by confidence threshold
        
        Args:
            min_confidence: Minimum confidence level
        
        Returns:
            Filtered list of signals
        """
        return [s for s in self.signals if s.confidence >= min_confidence]


if __name__ == "__main__":
    from data.loader import HFDataLoader
    from indicators.zigzag_predictive import PredictiveZigZag
    
    loader = HFDataLoader()
    df = loader.fetch_pair_data('BTCUSDT', timeframe='15m', limit=500)
    
    zigzag = PredictiveZigZag(threshold=0.5, predict_bars=3)
    generator = SignalGenerator(zigzag)
    
    df = generator.generate(df)
    signals = generator.get_signals()
    
    print(f"Total signals: {len(signals)}")
    for signal in signals[-5:]:
        print(f"{signal.timestamp}: {signal.signal_type} confidence={signal.confidence:.2f}")
