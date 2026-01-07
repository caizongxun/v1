import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import stats
from loguru import logger
from .zigzag_base import ZigZagBase, ZigZagPoint


@dataclass
class PredictedReversal:
    prediction_index: int
    predicted_reversal_index: int
    predicted_price: float
    confidence: float
    direction: str  # 'up' or 'down'
    signals: Dict
    bars_ahead: int = 0


class PredictiveZigZag(ZigZagBase):
    """
    Enhanced ZigZag with predictive capabilities
    Addresses lag by predicting reversals before they occur
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_length: int = 20,
        predict_bars: int = 3,
        confidence_threshold: float = 0.65
    ):
        """
        Initialize Predictive ZigZag
        
        Args:
            threshold: Percentage threshold for reversal
            min_length: Minimum bars between zigzag points
            predict_bars: How many bars ahead to predict reversals
            confidence_threshold: Minimum confidence for signal
        """
        super().__init__(threshold, min_length)
        self.predict_bars = predict_bars
        self.confidence_threshold = confidence_threshold
        self.predicted_reversals: List[PredictedReversal] = []
        self.early_signals: List[Dict] = []
    
    def calculate_with_prediction(self, df: pd.DataFrame, use_hl2: bool = True) -> pd.DataFrame:
        """
        Calculate ZigZag with predictive early signals
        
        Args:
            df: DataFrame with OHLC data
            use_hl2: Use (High+Low)/2
        
        Returns:
            DataFrame with ZigZag and prediction columns
        """
        # Calculate base ZigZag
        df = self.calculate(df, use_hl2)
        
        # Add prediction columns
        df['ZigZag_Predicted'] = np.nan
        df['Reversal_Confidence'] = 0.0
        df['Early_Signal'] = 0
        df['Signal_Type'] = ''
        
        try:
            # Predict reversals
            for i in range(self.min_length, len(df)):
                window_df = df.iloc[max(0, i - self.min_length):i + 1]
                prediction = self._predict_reversal(
                    window_df,
                    df.index[i]
                )
                
                if prediction:
                    # Store prediction
                    self.predicted_reversals.append(prediction)
                    
                    # Fill columns
                    if prediction.predicted_reversal_index < len(df):
                        df.loc[df.index[prediction.predicted_reversal_index], 'ZigZag_Predicted'] = prediction.predicted_price
                        df.loc[df.index[i], 'Reversal_Confidence'] = prediction.confidence
                    
                    if prediction.confidence >= self.confidence_threshold:
                        df.loc[df.index[i], 'Early_Signal'] = 1 if prediction.direction == 'up' else -1
                        df.loc[df.index[i], 'Signal_Type'] = f'{prediction.direction}_reversal'
                        self.early_signals.append({
                            'index': i,
                            'prediction': prediction,
                            'timestamp': df.index[i]
                        })
        
        except Exception as e:
            logger.error(f"Error in predictive calculation: {str(e)}")
        
        return df
    
    def _predict_reversal(
        self,
        window_df: pd.DataFrame,
        current_time: pd.Timestamp
    ) -> Optional[PredictedReversal]:
        """
        Predict upcoming reversal point
        
        Args:
            window_df: Recent price data window
            current_time: Current timestamp
        
        Returns:
            PredictedReversal object or None
        """
        try:
            if len(window_df) < self.min_length:
                return None
            
            current_idx = len(window_df) - 1
            
            # Collect signals
            signals = {}
            
            # Signal 1: Price momentum
            signals['momentum'] = self._analyze_momentum(
                window_df['Close'].values
            )
            
            # Signal 2: Volatility expansion
            signals['volatility'] = self._analyze_volatility(
                window_df['High'].values,
                window_df['Low'].values
            )
            
            # Signal 3: Volume confirmation
            if 'Volume' in window_df.columns:
                signals['volume'] = self._analyze_volume(
                    window_df['Volume'].values
                )
            else:
                signals['volume'] = 0.0
            
            # Signal 4: Price pattern
            signals['pattern'] = self._detect_reversal_pattern(
                window_df['High'].values,
                window_df['Low'].values
            )
            
            # Signal 5: Support/Resistance
            signals['sr_level'] = self._check_sr_proximity(
                window_df['Close'].values,
                window_df['High'].values,
                window_df['Low'].values
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(signals)
            
            if confidence < self.confidence_threshold:
                return None
            
            # Predict direction
            direction = 'up' if signals['momentum'] > 0 else 'down'
            
            # Estimate reversal price
            predicted_price = self._estimate_reversal_price(
                window_df,
                direction,
                signals
            )
            
            # Estimate reversal bars
            predicted_bars = min(self.predict_bars, len(window_df) // 4)
            predicted_idx = current_idx + predicted_bars
            
            return PredictedReversal(
                prediction_index=current_idx,
                predicted_reversal_index=predicted_idx,
                predicted_price=predicted_price,
                confidence=confidence,
                direction=direction,
                signals=signals,
                bars_ahead=predicted_bars
            )
        
        except Exception as e:
            logger.debug(f"Prediction error: {str(e)}")
            return None
    
    def _analyze_momentum(self, prices: np.ndarray) -> float:
        """
        Analyze price momentum
        
        Returns: Score between -1 and 1
        """
        if len(prices) < 5:
            return 0.0
        
        # Calculate momentum
        recent = prices[-5:]
        momentum = (recent[-1] - recent[0]) / recent[0]
        
        # Normalize to -1 to 1 range
        return np.clip(momentum * 100, -1, 1)
    
    def _analyze_volatility(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """
        Analyze volatility expansion (reversal signal)
        
        Returns: Score between 0 and 1
        """
        if len(highs) < 10:
            return 0.0
        
        ranges = highs[-10:] - lows[-10:]
        recent_range = ranges[-1]
        avg_range = np.mean(ranges[:-1])
        
        if avg_range == 0:
            return 0.0
        
        expansion = recent_range / avg_range
        return np.clip((expansion - 1) / 2, 0, 1)  # 0 to 1 range
    
    def _analyze_volume(self, volumes: np.ndarray) -> float:
        """
        Analyze volume confirmation
        
        Returns: Score between 0 and 1
        """
        if len(volumes) < 10:
            return 0.0
        
        recent_vol = volumes[-1]
        avg_vol = np.mean(volumes[:-1])
        
        if avg_vol == 0:
            return 0.0
        
        vol_ratio = recent_vol / avg_vol
        return np.clip(vol_ratio - 1, 0, 1)
    
    def _detect_reversal_pattern(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """
        Detect reversal patterns (e.g., double top/bottom)
        
        Returns: Score between 0 and 1
        """
        if len(highs) < 15:
            return 0.0
        
        recent_highs = highs[-15:]
        recent_lows = lows[-15:]
        
        # Check for pattern similarity
        score = 0.0
        
        # Double top pattern (two peaks close together)
        if abs(recent_highs[-1] - recent_highs[-8]) / recent_highs[-1] < 0.02:
            score += 0.5
        
        # Recent lower low (continuation of downtrend)
        if recent_lows[-1] < np.mean(recent_lows[:-5]):
            score += 0.3
        
        return np.clip(score, 0, 1)
    
    def _check_sr_proximity(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> float:
        """
        Check proximity to support/resistance
        
        Returns: Score between 0 and 1
        """
        if len(closes) < 20:
            return 0.0
        
        current = closes[-1]
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        resistance = np.mean(recent_highs[-5:])
        support = np.mean(recent_lows[-5:])
        
        # Distance to nearest level
        dist_to_res = abs(current - resistance) / resistance
        dist_to_sup = abs(current - support) / support
        
        min_dist = min(dist_to_res, dist_to_sup)
        return np.clip(1 - min_dist * 10, 0, 1)  # Closer = higher score
    
    def _calculate_confidence(self, signals: Dict) -> float:
        """
        Calculate overall prediction confidence
        
        Args:
            signals: Dictionary of signal components
        
        Returns: Confidence score between 0 and 1
        """
        weights = {
            'momentum': 0.25,
            'volatility': 0.25,
            'volume': 0.20,
            'pattern': 0.15,
            'sr_level': 0.15
        }
        
        confidence = sum(
            abs(signals.get(key, 0)) * weight
            for key, weight in weights.items()
        )
        
        return np.clip(confidence, 0, 1)
    
    def _estimate_reversal_price(self, window_df: pd.DataFrame, direction: str, signals: Dict) -> float:
        """
        Estimate the reversal price level
        
        Args:
            window_df: Price data window
            direction: Direction of reversal
            signals: Signal components
        
        Returns: Estimated reversal price
        """
        current_price = window_df['Close'].iloc[-1]
        volatility = (window_df['High'].iloc[-1] - window_df['Low'].iloc[-1]) / current_price
        
        if direction == 'up':
            # Estimate upside target
            target = current_price * (1 + volatility * 2)
        else:
            # Estimate downside target
            target = current_price * (1 - volatility * 2)
        
        return target
    
    def get_early_signals(self) -> List[Dict]:
        """
        Get all early reversal signals
        
        Returns:
            List of signal dictionaries
        """
        return self.early_signals
    
    def get_latest_prediction(self) -> Optional[PredictedReversal]:
        """
        Get most recent prediction
        
        Returns:
            Latest PredictedReversal or None
        """
        return self.predicted_reversals[-1] if self.predicted_reversals else None


if __name__ == "__main__":
    import pandas as pd
    from data.loader import HFDataLoader
    
    loader = HFDataLoader()
    df = loader.fetch_pair_data('BTCUSDT', timeframe='15m', limit=500)
    
    predictor = PredictiveZigZag(threshold=0.5, predict_bars=3)
    df = predictor.calculate_with_prediction(df)
    
    print(f"Early signals found: {len(predictor.get_early_signals())}")
    latest = predictor.get_latest_prediction()
    if latest:
        print(f"Latest prediction: {latest.direction} with confidence {latest.confidence:.2f}")
