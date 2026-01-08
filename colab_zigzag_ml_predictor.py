#!/usr/bin/env python3
"""
ZigZag Label Predictor - ML Model
Predict HH/HL/LH/LL labels ahead of time

Workflow:
1. Extract ZigZag labels from historical data
2. Create features from price/volume data
3. Train ML model to predict next label
4. Evaluate on test set
5. Analyze feature importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("ZigZag Label Predictor - ML Pipeline")
print("="*80)

class ZigZagFixed:
    """ZigZag label generator"""
    
    def __init__(self, depth=3, deviation=2):
        self.depth = depth
        self.deviation = deviation / 100.0
    
    def find_extrema(self, highs, lows):
        n = len(highs)
        extrema = []
        if n < self.depth * 2:
            return extrema
        
        for i in range(self.depth, n - self.depth):
            left_high = np.max(highs[max(0, i - self.depth):i])
            right_high = np.max(highs[i + 1:min(n, i + self.depth + 1)])
            if highs[i] >= left_high and highs[i] >= right_high:
                extrema.append((i, highs[i], 'H'))
            
            left_low = np.min(lows[max(0, i - self.depth):i])
            right_low = np.min(lows[i + 1:min(n, i + self.depth + 1)])
            if lows[i] <= left_low and lows[i] <= right_low:
                extrema.append((i, lows[i], 'L'))
        
        return extrema
    
    def filter_extrema(self, extrema):
        if not extrema:
            return []
        
        extrema = sorted(extrema, key=lambda x: x[0])
        filtered_dict = {}
        for idx, price, ptype in extrema:
            if idx not in filtered_dict:
                filtered_dict[idx] = (idx, price, ptype)
            else:
                existing = filtered_dict[idx]
                if ptype == 'H' and existing[2] == 'H':
                    if price > existing[1]:
                        filtered_dict[idx] = (idx, price, ptype)
                elif ptype == 'L' and existing[2] == 'L':
                    if price < existing[1]:
                        filtered_dict[idx] = (idx, price, ptype)
                elif ptype == 'H':
                    filtered_dict[idx] = (idx, price, ptype)
                else:
                    filtered_dict[idx] = (idx, price, ptype)
        
        extrema = sorted(filtered_dict.values(), key=lambda x: x[0])
        zigzag = []
        for idx, price, ptype in extrema:
            if not zigzag:
                zigzag.append((idx, price, ptype))
                continue
            
            last_idx, last_price, last_type = zigzag[-1]
            if ptype == last_type:
                if ptype == 'H' and price > last_price:
                    zigzag[-1] = (idx, price, ptype)
                elif ptype == 'L' and price < last_price:
                    zigzag[-1] = (idx, price, ptype)
                continue
            
            if ptype == 'H':
                threshold = last_price * (1 + self.deviation)
                if price >= threshold:
                    zigzag.append((idx, price, ptype))
            else:
                threshold = last_price * (1 - self.deviation)
                if price <= threshold:
                    zigzag.append((idx, price, ptype))
        
        return zigzag
    
    def label_zigzag(self, zigzag):
        if len(zigzag) < 2:
            return []
        
        labeled = []
        labeled.append((zigzag[0][0], zigzag[0][2], None))
        
        direction = 1 if zigzag[1][1] > zigzag[0][1] else -1
        labeled.append((zigzag[1][0], zigzag[1][2], None))
        
        for i in range(2, len(zigzag)):
            curr_idx, curr_price, curr_type = zigzag[i]
            prev_idx, prev_price, prev_type = zigzag[i-1]
            prev2_idx, prev2_price, prev2_type = zigzag[i-2]
            
            if curr_type != prev_type:
                direction = 1 if curr_type == 'H' else -1
                last_point = prev2_price
            else:
                last_point = labeled[-1][2]
            
            if last_point is not None:
                if direction > 0:
                    label = 'HH' if curr_price > last_point else 'LH'
                else:
                    label = 'LL' if curr_price < last_point else 'HL'
            else:
                label = None
            
            labeled.append((curr_idx, curr_type, label))
        
        return labeled
    
    def get_labels_array(self, df, zigzag):
        """Create array with labels aligned to bars"""
        labels = np.full(len(df), None, dtype=object)
        for bar_idx, ptype, label in self.label_zigzag(zigzag):
            if 0 <= bar_idx < len(df):
                labels[bar_idx] = label
        return labels


class FeatureExtractor:
    """Extract features for ML model"""
    
    @staticmethod
    def extract_features(df, lookback=20):
        """Extract technical indicators and price features"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['close'] = df['close']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['volume']
        
        # Returns
        features['ret_1'] = df['close'].pct_change(1)
        features['ret_5'] = df['close'].pct_change(5)
        features['ret_10'] = df['close'].pct_change(10)
        
        # Volatility
        features['volatility_5'] = df['close'].pct_change().rolling(5).std()
        features['volatility_10'] = df['close'].pct_change().rolling(10).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        features['sma_5'] = df['close'].rolling(5).mean()
        features['sma_10'] = df['close'].rolling(10).mean()
        features['sma_20'] = df['close'].rolling(20).mean()
        
        # MA ratios - FIX: Handle division by zero
        features['price_to_sma5'] = df['close'] / features['sma_5']
        features['price_to_sma20'] = df['close'] / features['sma_20']
        features['sma5_to_sma20'] = features['sma_5'] / features['sma_20']
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands - FIX: Handle division by zero
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        features['bb_upper'] = sma + 2 * std
        features['bb_lower'] = sma - 2 * std
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        
        # FIX: Avoid division by zero in bb_position
        bb_width = features['bb_width'].copy()
        bb_width[bb_width == 0] = 1e-6  # Replace zero with small value
        features['bb_position'] = (df['close'] - features['bb_lower']) / bb_width
        
        # Directional features
        features['high_low_range'] = (df['high'] - df['low']) / df['close']
        features['close_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volume features
        vol_sma = df['volume'].rolling(20).mean()
        vol_sma[vol_sma == 0] = 1e-6  # Replace zero with small value
        features['volume_sma_ratio'] = df['volume'] / vol_sma
        features['volume_change'] = df['volume'].pct_change()
        
        # FIX: Replace all inf and nan values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Final validation
        if np.any(np.isnan(features.values)):
            features = features.fillna(0)
        
        if np.any(np.isinf(features.values)):
            features = features.replace([np.inf, -np.inf], 0)
        
        return features


class ZigZagPredictor:
    """ML model to predict ZigZag labels"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.label_encoder = {'HH': 0, 'HL': 1, 'LH': 2, 'LL': 3}
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
    
    def prepare_data(self, df, labeled_data, shift=1):
        """Prepare features and labels for training
        
        shift: predict N bars ahead (shift=1 means predict next bar's label)
        """
        features = FeatureExtractor.extract_features(df)
        self.feature_names = features.columns.tolist()
        
        # Shift labels forward (predict future)
        labels = labeled_data.copy()
        labels_shifted = np.roll(labels, shift)
        labels_shifted[:shift] = None  # Can't predict first few
        
        # Filter: only rows with valid labels
        valid_mask = np.array([label in self.label_encoder for label in labels_shifted])
        valid_indices = np.where(valid_mask)[0]
        
        X = features.iloc[valid_indices].values
        y = np.array([self.label_encoder[labels_shifted[i]] for i in valid_indices])
        
        print(f"  Data prepared: {len(X)} samples, {X.shape[1]} features")
        print(f"  Label distribution:")
        for label, code in self.label_encoder.items():
            count = np.sum(y == code)
            print(f"    {label}: {count} ({100*count/len(y):.1f}%)")
        
        return X, y, features
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model (version-compatible)"""
        print("\n  Training XGBoost...")
        
        # Validate input data
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            raise ValueError("X_train contains NaN or Inf values")
        if np.any(np.isnan(X_val)) or np.any(np.isinf(X_val)):
            raise ValueError("X_val contains NaN or Inf values")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model - use simple fit without early stopping for compatibility
        self.model = xgb.XGBClassifier(
            n_estimators=150,  # Reduced from 200
            max_depth=6,       # Slightly reduced to avoid overfitting
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # Simple fit without eval_set (most compatible)
        self.model.fit(X_train_scaled, y_train, verbose=False)
        
        print(f"  Model trained successfully!")
        
        # Evaluate on validation set
        val_score = self.model.score(X_val_scaled, y_val)
        print(f"  Validation accuracy: {val_score:.4f}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n  Test Accuracy: {accuracy:.4f}")
        print(f"  F1 Score (weighted): {f1_weighted:.4f}")
        print(f"\n  Classification Report:")
        labels_list = list(self.label_encoder.keys())
        report = classification_report(y_test, y_pred, target_names=labels_list)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        return cm, y_pred, y_pred_proba
    
    def feature_importance(self):
        """Get feature importance"""
        importance = self.model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        
        print(f"\n  Top 20 Important Features:")
        for i, idx in enumerate(sorted_idx[:20]):
            print(f"    {i+1:2d}. {self.feature_names[idx]:20s} - {importance[idx]:.4f}")
        
        return sorted_idx, importance


# MAIN
print("\n[1/6] Fetching data...")
try:
    df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index(drop=True)
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"  OK - {len(df):,} bars loaded")
except Exception as e:
    print(f"  ERROR: {e}")
    exit(1)

print("\n[2/6] Generating ZigZag labels...")
zz = ZigZagFixed(depth=3, deviation=2)
extrema = zz.find_extrema(df['high'].values, df['low'].values)
zigzag = zz.filter_extrema(extrema)
labeled_data = zz.get_labels_array(df, zigzag)
print(f"  Found {np.sum(labeled_data != None)} labeled points")

print("\n[3/6] Extracting features...")
predictor = ZigZagPredictor()
X, y, features_df = predictor.prepare_data(df, labeled_data, shift=1)

print("\n[4/6] Splitting data (80% train, 10% val, 10% test)...")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)
print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

print("\n[5/6] Training model...")
predictor.train(X_train, y_train, X_val, y_val)

print("\n[6/6] Evaluating...")
cm, y_pred, y_pred_proba = predictor.evaluate(X_test, y_test)
sorted_idx, importance = predictor.feature_importance()

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print("""
Model Performance:
- Predicts HH/HL/LH/LL labels 1 bar ahead
- Training on technical features (RSI, MACD, Bollinger Bands, etc.)
- XGBoost classifier with 150 estimators

Next Steps:
1. Use model.predict() on new data to get probability for each label
2. Build trading strategy based on predicted labels
3. Backtest strategy on historical data
4. Fine-tune parameters based on performance
""")
print("="*80 + "\n")

# Save model
import pickle
with open('zigzag_predictor_model.pkl', 'wb') as f:
    pickle.dump(predictor, f)
print("Model saved to zigzag_predictor_model.pkl")

print("\nDone!")
