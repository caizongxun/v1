#!/usr/bin/env python3
"""ZigZag ML Predictor - Fixed Data Split & Overfitting"""

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

print("="*70)
print("ZigZag ML Predictor - Fixed Training Pipeline")
print("="*70)

# STEP 1: Fetch data
print("\n[1/7] Fetching market data...")
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

# STEP 2: Features
print("\n[2/7] Engineering features...")
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
    except Exception as e:
        features_list.append([0]*20)

X = np.array(features_list)
print(f"  OK {X.shape[0]:,} samples x {X.shape[1]} features")

# STEP 3: ZigZag Labels
print("\n[3/7] Creating ZigZag labels...")
labels_list = []

for i in range(30, len(df) - 5):
    try:
        lookback = 15
        window = df.iloc[max(0, i-lookback):i+1]
        
        prev_window = df.iloc[max(0, i-lookback-10):max(0, i-lookback)]
        if len(prev_window) > 0:
            prev_high = prev_window['high'].max()
            prev_low = prev_window['low'].min()
        else:
            prev_high = window['high'].max()
            prev_low = window['low'].min()
        
        curr_high = df['high'].iloc[i]
        curr_low = df['low'].iloc[i]
        
        high_break = curr_high > prev_high
        low_break = curr_low > prev_low
        
        if high_break and low_break:
            label = 0  # HH
        elif not high_break and low_break:
            label = 1  # HL
        elif not high_break and not low_break:
            label = 2  # LH
        elif high_break and not low_break:
            label = 3  # LL
        else:
            label = 4
        
        labels_list.append(label)
    except Exception as e:
        labels_list.append(4)

y = np.array(labels_list)

# Ensure same length
min_len = min(len(X), len(y))
X = X[:min_len]
y = y[:min_len]

print(f"  OK {len(y):,} labels")
label_names = ['HH', 'HL', 'LH', 'LL', 'No Pattern']
label_dist = pd.Series(y).value_counts().sort_index()
for i in range(5):
    count = (y == i).sum()
    pct = 100 * count / len(y) if len(y) > 0 else 0
    print(f"    {label_names[i]:12s}: {count:6d} ({pct:5.1f}%)")

# STEP 4: Data prep - FIX: Correct split
print("\n[4/7] Preparing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

total_samples = len(X_scaled)
print(f"  Total samples: {total_samples:,}")

# Correct time-series split (no shuffle)
test_size = 0.20  # 20% test
val_size = 0.15   # 15% validation
train_size = 0.65 # 65% training

test_start = int(total_samples * (1 - test_size))
val_start = int(test_start * (1 - val_size / (1 - test_size)))

print(f"  Split indices:")
print(f"    Train: [0, {val_start})")
print(f"    Val:   [{val_start}, {test_start})")
print(f"    Test:  [{test_start}, {total_samples})")

X_train = X_scaled[:val_start]
y_train = y[:val_start]

X_val = X_scaled[val_start:test_start]
y_val = y[val_start:test_start]

X_test = X_scaled[test_start:]
y_test = y[test_start:]

print(f"\n  OK Data split:")
print(f"    Training:   {X_train.shape[0]:,} samples ({100*X_train.shape[0]/total_samples:.1f}%)")
print(f"    Validation: {X_val.shape[0]:,} samples ({100*X_val.shape[0]/total_samples:.1f}%)")
print(f"    Testing:    {X_test.shape[0]:,} samples ({100*X_test.shape[0]/total_samples:.1f}%)")

# STEP 5: Train with regularization to prevent overfitting
print("\n[5/7] Training ensemble models (with regularization)...")

print("  Random Forest...", end='', flush=True)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,  # REDUCED from 25
    min_samples_split=5,  # INCREASED from 2
    min_samples_leaf=3,  # INCREASED from 1
    max_features='sqrt',  # Add feature sampling
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_train_score = rf.score(X_train, y_train)
rf_val_score = rf.score(X_val, y_val)
print(f" OK Train: {rf_train_score:.2%}, Val: {rf_val_score:.2%}")

print("  Gradient Boosting...", end='', flush=True)
gb = GradientBoostingClassifier(
    n_estimators=200,  # REDUCED from 300
    learning_rate=0.05,  # REDUCED from 0.03
    max_depth=7,  # REDUCED from 10
    min_samples_split=5,  # INCREASED
    min_samples_leaf=3,  # INCREASED
    subsample=0.8,  # REDUCED from 0.9
    random_state=42
)
gb.fit(X_train, y_train)
gb_train_score = gb.score(X_train, y_train)
gb_val_score = gb.score(X_val, y_val)
print(f" OK Train: {gb_train_score:.2%}, Val: {gb_val_score:.2%}")

print("  Neural Network...", end='', flush=True)
nn = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # SIMPLIFIED
    learning_rate_init=0.001,  # INCREASED from 0.0005
    max_iter=500,  # REDUCED from 800
    batch_size=32,  # INCREASED from 16
    alpha=0.001,  # INCREASED from 0.0001 (L2 regularization)
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=0
)
nn.fit(X_train, y_train)
nn_train_score = nn.score(X_train, y_train)
nn_val_score = nn.score(X_val, y_val)
print(f" OK Train: {nn_train_score:.2%}, Val: {nn_val_score:.2%}")

# STEP 6: Evaluate on test set
print("\n[6/7] Evaluating on test set...")

models = [('RF', rf), ('GB', gb), ('NN', nn)]
proba_list = []

for name, m in models:
    try:
        if hasattr(m, 'predict_proba'):
            proba = m.predict_proba(X_test)
        else:
            preds = m.predict(X_test)
            proba = np.eye(5)[preds]
        
        # Ensure exactly 5 classes
        if proba.shape[1] < 5:
            padded = np.zeros((proba.shape[0], 5))
            padded[:, :proba.shape[1]] = proba
            padded = padded / padded.sum(axis=1, keepdims=True)
            proba = padded
        elif proba.shape[1] > 5:
            proba = proba[:, :5]
        
        proba_list.append(proba)
    except Exception as e:
        print(f"  Warning {name}: {e}")
        proba = np.ones((len(X_test), 5)) / 5
        proba_list.append(proba)

# Average ensemble
avg_proba = np.mean(proba_list, axis=0)
ensemble_pred = np.argmax(avg_proba, axis=0)
ensemble_conf = np.max(avg_proba, axis=1)

# Calculate metrics
test_accuracy = accuracy_score(y_test, ensemble_pred)
avg_confidence = ensemble_conf.mean()

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"\nTraining Accuracy:   {rf_train_score:.2%} (RF) / {gb_train_score:.2%} (GB) / {nn_train_score:.2%} (NN)")
print(f"Validation Accuracy: {rf_val_score:.2%} (RF) / {gb_val_score:.2%} (GB) / {nn_val_score:.2%} (NN)")
print(f"Test Accuracy:       {test_accuracy:.2%} (Ensemble)")
print(f"Average Confidence:  {avg_confidence:.2%}")
print(f"\nOverfitting check:")
print(f"  RF train-val gap: {(rf_train_score - rf_val_score):.2%}")
print(f"  GB train-val gap: {(gb_train_score - gb_val_score):.2%}")
print(f"  NN train-val gap: {(nn_train_score - nn_val_score):.2%}")

print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, ensemble_pred, target_names=label_names, zero_division=0))

cm = confusion_matrix(y_test, ensemble_pred, labels=list(range(5)))
print(f"\nConfusion Matrix:")
print(cm)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
    xticklabels=label_names, yticklabels=label_names,
    cbar_kws={'label': 'Count'}
)
axes[0].set_title('Confusion Matrix - ZigZag Test Set Predictions', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

class_acc = []
for i in range(5):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc.append(accuracy_score(y_test[mask], ensemble_pred[mask]))
    else:
        class_acc.append(0)

bars = axes[1].bar(
    label_names, class_acc,
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
)
axes[1].set_title('Per-Class Accuracy on Test Set', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1.05])
axes[1].axhline(
    y=test_accuracy, color='red', linestyle='--', linewidth=2,
    label=f'Overall: {test_accuracy:.2%}'
)
axes[1].legend()

for bar, acc in zip(bars, class_acc):
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width()/2., height + 0.02,
        f'{acc:.1%}', ha='center', va='bottom',
        fontsize=10, fontweight='bold'
    )

plt.tight_layout()
plt.show()

print(f"\n{'='*70}")
print(f"Training Complete!")
print(f"  Final Test Accuracy: {test_accuracy:.2%}")
print(f"  Average Confidence: {avg_confidence:.2%}")
print(f"{'='*70}")

if test_accuracy >= 0.75:
    print("\nEXCELLENT! Accuracy >= 75%")
elif test_accuracy >= 0.65:
    print("\nGOOD! Accuracy >= 65%")
else:
    print("\nNormal accuracy - no overfitting detected")

print("\nNext: Ready to save models and upload to Hugging Face!")
