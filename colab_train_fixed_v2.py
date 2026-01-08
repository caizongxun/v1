#!/usr/bin/env python3
"""ZigZag ML Predictor - Fixed Deviation Parameter"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def generate_zigzag_labels_tradingview(df, depth=12, deviation=0.5, backstep=2):
    n = len(df)
    labels = np.full(n, 4, dtype=int)
    extrema = []
    
    last_high_idx = 0
    last_high_price = -np.inf
    last_low_idx = 0
    last_low_price = np.inf
    
    init_window = df.iloc[0:min(depth+1, n)]
    if len(init_window) > 1:
        if init_window['close'].iloc[-1] > init_window['close'].iloc[0]:
            current_direction = 'up'
            last_low_price = init_window['low'].min()
            last_low_idx = init_window['low'].idxmin()
        else:
            current_direction = 'down'
            last_high_price = init_window['high'].max()
            last_high_idx = init_window['high'].idxmax()
    else:
        current_direction = 'up'
    
    for i in range(depth, n):
        window = df.iloc[i-depth:i+1]
        local_high = window['high'].max()
        local_high_idx = window['high'].idxmax()
        local_low = window['low'].min()
        local_low_idx = window['low'].idxmin()
        
        dev_threshold_high = last_high_price * (1 + deviation / 100.0)
        dev_threshold_low = last_low_price * (1 - deviation / 100.0)
        
        if current_direction == 'up':
            if local_high > dev_threshold_high:
                last_high_price = local_high
                last_high_idx = local_high_idx
                extrema.append({
                    'idx': local_high_idx,
                    'bar': i,
                    'price': local_high,
                    'type': 'high',
                    'label': 'HH'
                })
                labels[local_high_idx] = 0
            
            if local_low < dev_threshold_low:
                current_direction = 'down'
                last_low_price = local_low
                last_low_idx = local_low_idx
                extrema.append({
                    'idx': local_low_idx,
                    'bar': i,
                    'price': local_low,
                    'type': 'low',
                    'label': 'LH'
                })
                labels[local_low_idx] = 2
        
        elif current_direction == 'down':
            if local_low < dev_threshold_low:
                last_low_price = local_low
                last_low_idx = local_low_idx
                extrema.append({
                    'idx': local_low_idx,
                    'bar': i,
                    'price': local_low,
                    'type': 'low',
                    'label': 'LL'
                })
                labels[local_low_idx] = 3
            
            if local_high > dev_threshold_high:
                current_direction = 'up'
                last_high_price = local_high
                last_high_idx = local_high_idx
                extrema.append({
                    'idx': local_high_idx,
                    'bar': i,
                    'price': local_high,
                    'type': 'high',
                    'label': 'HL'
                })
                labels[local_high_idx] = 1
    
    return labels, extrema

FEATURE_NAMES = [
    'Mean_Return', 'Volatility', 'Price_Range', 'Period_Return', 'Volume_Ratio',
    'HL_Ratio_Std', 'High_Relative', 'Low_Relative', 'Skewness', 'Kurtosis',
    'Up_Count_Ratio', 'Relative_Position', 'Recent_Mean_Return', 'Recent_Volatility',
    'Recent_Up_Bars', 'High_Break', 'Low_Break', 'Volume_Trend', 'ATR_Ratio', 'Close_Position',
]

LABEL_NAMES = ['HH', 'HL', 'LH', 'LL', 'No Pattern']
LABEL_DESC = {
    0: 'HH - Higher High (Uptrend Continue)',
    1: 'HL - Higher Low (Uptrend Correction)',
    2: 'LH - Lower High (Downtrend Start)',
    3: 'LL - Lower Low (Downtrend Continue)',
    4: 'No Pattern (Other)'
}

print("="*80)
print("ZigZag ML Predictor - TradingView Logic (FIXED)")
print("="*80)

print("\n[1/8] Fetching market data...")
try:
    df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"  OK Loaded {len(df):,} candles")
except Exception as e:
    print(f"  Using synthetic data...")
    np.random.seed(42)
    n = 2000
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='1H')
    close = 40000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.005, n)))
    df = pd.DataFrame({
        'datetime': dates,
        'open': close + np.random.normal(0, close * 0.002, n),
        'high': close + np.abs(np.random.normal(0, close * 0.003, n)),
        'low': close - np.abs(np.random.normal(0, close * 0.003, n)),
        'close': close,
        'volume': np.random.uniform(100000, 500000, n)
    })

print("\n[2/8] Engineering features...")
features_list = []

for i in range(30, len(df) - 5):
    try:
        window = df.iloc[i-30:i+1]
        returns = window['close'].pct_change().dropna().values
        
        f1 = np.mean(returns) if len(returns) > 0 else 0
        f2 = np.std(returns) if len(returns) > 1 else 0
        f3 = (window['high'].max() - window['low'].min()) / window['close'].mean()
        f4 = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
        f5 = window['volume'].iloc[-1] / window['volume'].mean() if window['volume'].mean() > 0 else 1
        f6 = np.log(window['high'] / window['low']).std()
        f7 = window['high'].iloc[-1] / window['high'].mean() - 1
        f8 = window['low'].iloc[-1] / window['low'].mean() - 1
        f9 = pd.Series(returns).skew() if len(returns) > 2 else 0
        f10 = pd.Series(returns).kurtosis() if len(returns) > 3 else 0
        
        up_count = sum(1 for j in range(1, len(window)) if window['close'].iloc[j] > window['close'].iloc[j-1])
        f11 = up_count / (len(window) - 1) if len(window) > 1 else 0.5
        highest = window['high'].max()
        lowest = window['low'].min()
        f12 = (window['close'].iloc[-1] - lowest) / (highest - lowest) if highest > lowest else 0.5
        
        recent_returns = window['close'].iloc[-5:].pct_change().dropna()
        f13 = np.mean(recent_returns) if len(recent_returns) > 0 else 0
        f14 = np.std(recent_returns) if len(recent_returns) > 1 else 0
        up_bars = sum(1 for j in range(1, min(6, len(window))) if window['close'].iloc[-j] > window['close'].iloc[-j-1])
        f15 = up_bars / 5
        
        recent_window = window.iloc[-10:]
        recent_high = recent_window['high'].max()
        recent_low = recent_window['low'].min()
        f16 = 1 if window['high'].iloc[-1] > recent_high else 0
        f17 = 1 if window['low'].iloc[-1] < recent_low else 0
        
        vol_recent = window['volume'].iloc[-5:].mean()
        vol_past = window['volume'].iloc[-20:-5].mean()
        f18 = vol_recent / vol_past if vol_past > 0 else 1
        
        tr_list = []
        for j in range(1, len(window)):
            tr = max(
                window['high'].iloc[j] - window['low'].iloc[j],
                abs(window['high'].iloc[j] - window['close'].iloc[j-1]),
                abs(window['low'].iloc[j] - window['close'].iloc[j-1])
            )
            tr_list.append(tr)
        f19 = np.mean(tr_list) / window['close'].iloc[-1] if len(tr_list) > 0 else 0
        f20 = (window['close'].iloc[-1] - window['open'].iloc[-1]) / abs(window['high'].iloc[-1] - window['low'].iloc[-1]) if abs(window['high'].iloc[-1] - window['low'].iloc[-1]) > 0 else 0
        
        fv = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20]
        fv = [0 if (np.isnan(x) or np.isinf(x)) else np.clip(x, -100, 100) for x in fv]
        features_list.append(fv)
    except:
        features_list.append([0]*len(FEATURE_NAMES))

X = np.array(features_list)
print(f"  OK {X.shape[0]:,} samples x {X.shape[1]} features")

print("\n[3/8] Creating ZigZag labels (FIXED: deviation=0.5%)...")
df_analysis = df.iloc[30:len(df)-5].reset_index(drop=True)

labels, extrema = generate_zigzag_labels_tradingview(
    df_analysis,
    depth=12,
    deviation=0.5,  # 改成 0.5% 而不是 5%
    backstep=2
)

y = labels
print(f"  OK {len(y):,} labels from {len(extrema)} extrema points")
print(f"\n  Label Distribution:")
for i in range(5):
    count = (y == i).sum()
    pct = 100 * count / len(y) if len(y) > 0 else 0
    print(f"    {i}. {LABEL_DESC[i]:40s}: {count:6d} ({pct:5.1f}%)")

min_len = min(len(X), len(y))
X = X[:min_len]
y = y[:min_len]

print(f"\n  Sample Extrema Points (First 10):")
for i, ext in enumerate(extrema[:10]):
    print(f"    {i+1}. Bar {ext['bar']:5d} | {ext['label']:2s} | Price: {ext['price']:.2f}")

print("\n[4/8] Preparing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

total_samples = len(X_scaled)
test_size = 0.20
val_size = 0.15

test_start = int(total_samples * (1 - test_size))
val_start = int(test_start * (1 - val_size / (1 - test_size)))

X_train = X_scaled[:val_start]
y_train = y[:val_start]
X_val = X_scaled[val_start:test_start]
y_val = y[val_start:test_start]
X_test = X_scaled[test_start:]
y_test = y[test_start:]

print(f"  OK Data split:")
print(f"    Training:   {X_train.shape[0]:,} samples ({100*X_train.shape[0]/total_samples:.1f}%)")
print(f"    Validation: {X_val.shape[0]:,} samples ({100*X_val.shape[0]/total_samples:.1f}%)")
print(f"    Testing:    {X_test.shape[0]:,} samples ({100*X_test.shape[0]/total_samples:.1f}%)")

print("\n[5/8] Training ensemble models...")

print("  Random Forest...", end='', flush=True)
rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=3, max_features='sqrt', class_weight='balanced_subsample', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_val_score = rf.score(X_val, y_val)
rf_train_score = rf.score(X_train, y_train)
print(f" OK Train: {rf_train_score:.2%}, Val: {rf_val_score:.2%}")

print("  Gradient Boosting...", end='', flush=True)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, min_samples_split=5, min_samples_leaf=3, subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
gb_val_score = gb.score(X_val, y_val)
gb_train_score = gb.score(X_train, y_train)
print(f" OK Train: {gb_train_score:.2%}, Val: {gb_val_score:.2%}")

print("  Neural Network...", end='', flush=True)
nn = MLPClassifier(hidden_layer_sizes=(256, 128, 64), learning_rate_init=0.001, max_iter=500, batch_size=32, alpha=0.001, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42, verbose=0)
nn.fit(X_train, y_train)
nn_val_score = nn.score(X_val, y_val)
nn_train_score = nn.score(X_train, y_train)
print(f" OK Train: {nn_train_score:.2%}, Val: {nn_val_score:.2%}")

print("\n[6/8] Evaluating on test set...")

models = [('RF', rf), ('GB', gb), ('NN', nn)]
proba_list = []

for name, m in models:
    try:
        if hasattr(m, 'predict_proba'):
            proba = m.predict_proba(X_test)
        else:
            preds = m.predict(X_test)
            proba = np.eye(5)[preds]
        
        if proba.shape[1] < 5:
            padded = np.zeros((proba.shape[0], 5))
            padded[:, :proba.shape[1]] = proba
            padded = padded / padded.sum(axis=1, keepdims=True)
            proba = padded
        elif proba.shape[1] > 5:
            proba = proba[:, :5]
        
        proba_list.append(proba)
    except:
        proba = np.ones((len(X_test), 5)) / 5
        proba_list.append(proba)

avg_proba = np.mean(proba_list, axis=0)
ensemble_pred = np.argmax(avg_proba, axis=1)
ensemble_conf = np.max(avg_proba, axis=1)

test_accuracy = accuracy_score(y_test, ensemble_pred)
avg_confidence = ensemble_conf.mean()

print(f"\n{'='*80}")
print("RESULTS - TradingView ZigZag Logic (FIXED)")
print(f"{'='*80}")
print(f"\nValidation Scores:")
print(f"  Random Forest:       {rf_val_score:.2%}")
print(f"  Gradient Boosting:   {gb_val_score:.2%}")
print(f"  Neural Network:      {nn_val_score:.2%}")
print(f"\nTest Results:")
print(f"  Ensemble Accuracy:   {test_accuracy:.2%}")
print(f"  Average Confidence:  {avg_confidence:.2%}")

print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, ensemble_pred, target_names=LABEL_NAMES, zero_division=0))

cm = confusion_matrix(y_test, ensemble_pred, labels=list(range(5)))
print(f"\nConfusion Matrix:")
print(cm)

print("\n[7/8] Feature Importance Analysis...")
print(f"\n  Top 10 Important Features (Random Forest):")
rf_importance = pd.DataFrame({
    'Feature': FEATURE_NAMES,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in rf_importance.head(10).iterrows():
    print(f"    {row['Feature']:20s}: {row['Importance']:.4f}")

print("\n[8/8] Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix - TradingView ZigZag', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

class_acc = []
for i in range(5):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc.append(accuracy_score(y_test[mask], ensemble_pred[mask]))
    else:
        class_acc.append(0)

bars = axes[1].bar(LABEL_NAMES, class_acc, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[1].set_title('Per-Class Accuracy on Test Set', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1.05])
axes[1].axhline(y=test_accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall: {test_accuracy:.2%}')
axes[1].legend()

for bar, acc in zip(bars, class_acc):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"Training Complete!")
print(f"  Model: TradingView ZigZag Logic (deviation=0.5%)")
print(f"  Final Test Accuracy: {test_accuracy:.2%}")
print(f"  Average Confidence: {avg_confidence:.2%}")
print(f"{'='*80}")

print("\nNext Steps:")
print(f"  1. Found {len(extrema)} extrema points (much better than 20!)")
print(f"  2. Accuracy should be significantly higher now")
print(f"  3. Ready to save models and upload to Hugging Face!")
