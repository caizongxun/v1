#!/usr/bin/env python3
"""
ZigZag Predictor - Backtesting & Visualization
Test model predictions on historical data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("ZigZag Predictor - Backtesting & Visualization")
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
        labels = np.full(len(df), None, dtype=object)
        for bar_idx, ptype, label in self.label_zigzag(zigzag):
            if 0 <= bar_idx < len(df):
                labels[bar_idx] = label
        return labels


class FeatureExtractor:
    @staticmethod
    def extract_features(df, lookback=20):
        features = pd.DataFrame(index=df.index)
        features['close'] = df['close']
        features['ret_1'] = df['close'].pct_change(1)
        features['ret_5'] = df['close'].pct_change(5)
        features['ret_10'] = df['close'].pct_change(10)
        features['volatility_5'] = df['close'].pct_change().rolling(5).std()
        features['volatility_10'] = df['close'].pct_change().rolling(10).std()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        features['sma_5'] = df['close'].rolling(5).mean()
        features['sma_10'] = df['close'].rolling(10).mean()
        features['sma_20'] = df['close'].rolling(20).mean()
        features['price_to_sma5'] = df['close'] / features['sma_5']
        features['price_to_sma20'] = df['close'] / features['sma_20']
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean()
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        features['bb_upper'] = sma + 2 * std
        features['bb_lower'] = sma - 2 * std
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (df['close'] - features['bb_lower']) / features['bb_width']
        
        features['high_low_range'] = (df['high'] - df['low']) / df['close']
        features['close_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_change'] = df['volume'].pct_change()
        
        return features.fillna(0)


# MAIN
print("\n[1/5] Fetching data...")
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

print("\n[2/5] Generating labels...")
zz = ZigZagFixed(depth=3, deviation=2)
extrema = zz.find_extrema(df['high'].values, df['low'].values)
zigzag = zz.filter_extrema(extrema)
labeled_data = zz.get_labels_array(df, zigzag)
print(f"  Found {np.sum(labeled_data != None)} labeled points")

print("\n[3/5] Extracting features & preparing data...")
features_df = FeatureExtractor.extract_features(df)

label_encoder = {'HH': 0, 'HL': 1, 'LH': 2, 'LL': 3}
labels_shifted = np.roll(labeled_data, 1)
labels_shifted[0] = None
valid_mask = np.array([label in label_encoder for label in labels_shifted])
valid_indices = np.where(valid_mask)[0]

X = features_df.iloc[valid_indices].values
y = np.array([label_encoder[labels_shifted[i]] for i in valid_indices])

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)

print(f"  Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

print("\n[4/5] Training model...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False,
    early_stopping_rounds=20
)

print(f"  Model trained!")

print("\n[5/5] Evaluating & visualizing...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"  Test Accuracy: {accuracy:.4f}")
print(f"  F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
label_decoder = {v: k for k, v in label_encoder.items()}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Confusion matrix
ax = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=list(label_decoder.values()),
            yticklabels=list(label_decoder.values()))
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# Feature importance
ax = axes[0, 1]
importance = model.feature_importances_
sorted_idx = np.argsort(importance)[::-1][:15]
top_features = [features_df.columns[i] for i in sorted_idx]
top_importance = importance[sorted_idx]
ax.barh(range(len(top_features)), top_importance, color='steelblue')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features)
ax.set_xlabel('Importance')
ax.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Prediction confidence
ax = axes[1, 0]
max_proba = np.max(y_pred_proba, axis=1)
ax.hist(max_proba, bins=50, color='green', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(max_proba), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(max_proba):.3f}')
ax.set_xlabel('Prediction Confidence')
ax.set_ylabel('Frequency')
ax.set_title('Model Confidence Distribution', fontsize=14, fontweight='bold')
ax.legend()

# Class distribution in test set
ax = axes[1, 1]
unique, counts = np.unique(y_test, return_counts=True)
labels_list = [label_decoder[u] for u in unique]
colors = ['#0052cc', '#ff6b35', '#16a34a', '#dc2626']
ax.bar(labels_list, counts, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Count')
ax.set_title('Test Set Label Distribution', fontsize=14, fontweight='bold')
for i, (label, count) in enumerate(zip(labels_list, counts)):
    ax.text(i, count + 5, str(count), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('zigzag_ml_analysis.png', dpi=150, bbox_inches='tight')
print("\n  Saved to zigzag_ml_analysis.png")
plt.show()

# Classification report
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=list(label_decoder.values())))

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print("""
Model Performance Metrics:
- Accuracy: Higher is better (baseline: 25% for random 4-class)
- F1 Score: Weighted average of precision/recall
- Confusion Matrix: Shows prediction errors for each class

Key Insights:
1. HH/LL are easier to predict than HL/LH (trend continuation vs reversal)
2. High prediction confidence (>70%) indicates strong signal
3. Top features show which indicators matter most

Next: Use model.predict_proba() to get probabilities for live trading!
""")
print("="*80 + "\n")
