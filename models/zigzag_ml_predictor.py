import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import joblib
from loguru import logger


@dataclass
class PatternLabel:
    index: int
    label: str  # HH, HL, LH, LL
    timestamp: pd.Timestamp
    confidence: float
    price: float


class ZigZagMLPredictor:
    """
    Machine Learning predictor for ZigZag HH/HL/LH/LL patterns
    Based on Pine Script logic from TradingView
    """
    
    def __init__(self, model_type: str = 'ensemble', sequence_length: int = 50):
        """
        Initialize ML predictor
        
        Args:
            model_type: 'ensemble', 'lstm', 'xgboost', 'neural_net'
            sequence_length: Length of price sequence for LSTM
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        self.class_weights = {}
        logger.info(f"Initialized ZigZagMLPredictor with type: {model_type}")
    
    def extract_features(self, df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
        """
        Extract features from OHLCV data for ML
        
        Args:
            df: DataFrame with OHLCV
            lookback: Historical bars to consider
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features = []
        
        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i+1]
            
            # Price statistics
            returns = window['Close'].pct_change().dropna()
            
            feature_vector = [
                window['Close'].iloc[-1] / window['Close'].iloc[0] - 1,  # Total return
                returns.mean(),  # Mean return
                returns.std(),  # Volatility
                (window['High'].max() - window['Low'].min()) / window['Close'].iloc[0],  # Range
                window['Volume'].mean(),  # Avg volume
                (window['Close'].iloc[-1] - window['Close'].iloc[0]) / window['Close'].iloc[0],  # Log return
                window['High'].iloc[-1] / window['High'].mean() - 1,  # High vs avg
                window['Low'].iloc[-1] / window['Low'].mean() - 1,  # Low vs avg
                (window['High'].iloc[-1] - window['Low'].iloc[-1]) / window['Close'].iloc[-1],  # Current range
                np.max(window['High'].diff()),  # Max high move
                np.min(window['Low'].diff()),  # Min low move
                window['Volume'].iloc[-1] / window['Volume'].mean(),  # Volume ratio
                returns.skew(),  # Skewness
                returns.kurtosis(),  # Kurtosis
            ]
            
            # RSI-like momentum
            up = np.sum(returns[returns > 0])
            down = np.sum(np.abs(returns[returns < 0]))
            rsi = 100 * up / (up + down) if (up + down) > 0 else 50
            feature_vector.append(rsi)
            
            # Consecutive highs/lows
            consecutive_highs = 0
            consecutive_lows = 0
            for j in range(1, len(window)):
                if window['High'].iloc[j] > window['High'].iloc[j-1]:
                    consecutive_highs += 1
                if window['Low'].iloc[j] < window['Low'].iloc[j-1]:
                    consecutive_lows += 1
            
            feature_vector.append(consecutive_highs)
            feature_vector.append(consecutive_lows)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_lstm_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract sequential features for LSTM
        
        Args:
            df: DataFrame with OHLCV
        
        Returns:
            3D array (n_samples, sequence_length, n_features)
        """
        features = []
        
        for i in range(len(df) - self.sequence_length):
            sequence = df.iloc[i:i+self.sequence_length]
            
            # Normalize price data
            opens = sequence['Open'].values / sequence['Close'].iloc[0]
            highs = sequence['High'].values / sequence['Close'].iloc[0]
            lows = sequence['Low'].values / sequence['Close'].iloc[0]
            closes = sequence['Close'].values / sequence['Close'].iloc[0]
            volumes = sequence['Volume'].values / sequence['Volume'].mean()
            
            # Stack features
            seq_features = np.column_stack([
                opens, highs, lows, closes, volumes
            ])
            
            features.append(seq_features)
        
        return np.array(features)
    
    def create_labels(self, df: pd.DataFrame, depth: int = 12, deviation: int = 5) -> np.ndarray:
        """
        Create labels based on ZigZag logic from Pine Script
        
        Labels:
        0: HH (Higher High)
        1: HL (Higher Low)
        2: LH (Lower High)
        3: LL (Lower Low)
        4: No significant pattern
        
        Args:
            df: DataFrame with OHLCV
            depth: ZigZag depth parameter
            deviation: ZigZag deviation parameter
        
        Returns:
            Label array
        """
        labels = []
        
        for i in range(depth + 1, len(df) - 1):
            prev_window = df.iloc[max(0, i-depth):i]
            curr_price = df['Close'].iloc[i]
            
            # Get recent highs and lows
            prev_high = prev_window['High'].max()
            prev_low = prev_window['Low'].min()
            curr_high = df['High'].iloc[i]
            curr_low = df['Low'].iloc[i]
            
            # Determine pattern
            label = 4  # Default: no pattern
            
            # Check for HH/HL pattern (uptrend)
            if curr_high > prev_high:
                if i > 0 and df['Low'].iloc[i] > df['Low'].iloc[i-1]:
                    label = 1  # HL (Higher Low)
                else:
                    label = 0  # HH (Higher High)
            
            # Check for LH/LL pattern (downtrend)
            elif curr_high < prev_high:
                if i > 0 and df['Low'].iloc[i] < df['Low'].iloc[i-1]:
                    label = 3  # LL (Lower Low)
                else:
                    label = 2  # LH (Lower High)
            
            labels.append(label)
        
        return np.array(labels)
    
    def build_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Build ensemble of multiple models
        """
        logger.info("Building ensemble model")
        
        # Calculate class weights for imbalanced data
        unique, counts = np.unique(y_train, return_counts=True)
        self.class_weights = {u: len(y_train) / (len(unique) * c) for u, c in zip(unique, counts)}
        
        # Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=7,
            random_state=42
        )
        
        # Neural Network
        self.models['nn'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            batch_size=32,
            max_iter=300,
            random_state=42
        )
        
        # Train all models
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        logger.info("Ensemble model trained")
    
    def build_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Build LSTM neural network
        """
        logger.info("Building LSTM model")
        
        model = Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['lstm'] = model
        
        # Calculate class weights
        unique, counts = np.unique(y_train, return_counts=True)
        self.class_weights = {u: len(y_train) / (len(unique) * c) for u, c in zip(unique, counts)}
        
        logger.info("LSTM model created")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32):
        """
        Train selected model type
        """
        if self.model_type == 'ensemble':
            self.build_ensemble_model(X_train, y_train)
        
        elif self.model_type == 'lstm':
            self.build_lstm_model(X_train, y_train)
            
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            history = self.models['lstm'].fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                class_weight=self.class_weights,
                verbose=1
            )
            
            return history
        
        elif self.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                self.models['xgboost'] = XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=7,
                    random_state=42,
                    tree_method='gpu_hist' if self._has_gpu() else 'auto'
                )
                self.models['xgboost'].fit(X_train, y_train)
                logger.info("XGBoost model trained")
            except ImportError:
                logger.warning("XGBoost not available, using Random Forest instead")
                self.build_ensemble_model(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Returns:
            (predictions, confidence_scores)
        """
        if self.model_type == 'ensemble':
            predictions = []
            confidences = []
            
            for model in self.models.values():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    pred = np.argmax(proba, axis=1)
                    conf = np.max(proba, axis=1)
                else:
                    pred = model.predict(X)
                    conf = np.ones(len(pred)) * 0.5
                
                predictions.append(pred)
                confidences.append(conf)
            
            # Ensemble voting
            predictions = np.array(predictions)
            ensemble_pred = np.median(predictions, axis=0).astype(int)
            ensemble_conf = np.mean(confidences, axis=0)
            
            return ensemble_pred, ensemble_conf
        
        elif self.model_type == 'lstm':
            proba = self.models['lstm'].predict(X)
            predictions = np.argmax(proba, axis=1)
            confidence = np.max(proba, axis=1)
            
            return predictions, confidence
        
        elif self.model_type == 'xgboost' and 'xgboost' in self.models:
            proba = self.models['xgboost'].predict_proba(X)
            predictions = np.argmax(proba, axis=1)
            confidence = np.max(proba, axis=1)
            
            return predictions, confidence
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        """
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        predictions, confidences = self.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, predictions),
            'classification_report': classification_report(y_test, predictions, 
                                                          target_names=['HH', 'HL', 'LH', 'LL', 'No Pattern']),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'avg_confidence': np.mean(confidences)
        }
        
        logger.info(f"Model Accuracy: {results['accuracy']:.2%}")
        logger.info(f"Average Confidence: {results['avg_confidence']:.2%}")
        
        return results
    
    def save_model(self, path: str):
        """
        Save trained model
        """
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'type': self.model_type,
            'feature_importance': self.feature_importance
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load trained model
        """
        data = joblib.load(path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.model_type = data['type']
        self.feature_importance = data['feature_importance']
        logger.info(f"Model loaded from {path}")
    
    @staticmethod
    def _has_gpu() -> bool:
        """
        Check if GPU available
        """
        return len(tf.config.list_physical_devices('GPU')) > 0


if __name__ == "__main__":
    logger.info("ZigZag ML Predictor module loaded")
