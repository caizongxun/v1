#!/usr/bin/env python3
"""ZigZag ML Predictor - Pure Inline Training for Google Colab"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ZigZag ML Predictor - Inline Training Pipeline")
print("="*70)

# STEP 1: Fetch Market Data
print("\n[1/6] Fetching market data...")

try:
    print("  Downloading BTC-USD 1h data (2 years)...")
    df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df.columns = [c.lower().strip() for c in df.columns]
    
    print(f"  OK Loaded {len(df):,} candles")
    
except Exception as e:
    print(f"  WARNING: Data fetch failed: {e}")
    print("  Creating synthetic data for demo...")
    
    np.random.seed(42)
    n_samples = 2000
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq='1H')
    
    returns = np.random.normal(0.0001, 0.005, n_samples)
    close = 40000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': close + np.random.normal(0, close * 0.002, n_samples),
        'high': close + np.abs(np.random.normal(0, close * 0.003, n_samples)),
        'low': close - np.abs(np.random.normal(0, close * 0.003, n_samples)),
        'close': close,
        'volume': np.random.uniform(100000, 500000, n_samples)
    })
    
    print(f"  OK Generated {len(df):,} synthetic candles")

# STEP 2: Feature Engineering
print("\n[2/6] Engineering features...")

features_list = []

for i in range(20, len(df) - 1):
    try:
        window = df.iloc[i-20:i+1]
        returns = window['close'].pct_change().dropna().values
        
        mean_return = np.mean(returns) if len(returns) > 0 else 0
        volatility = np.std(returns) if len(returns) > 1 else 0
        price_range = (window['high'].max() - window['low'].min()) / window['close'].mean()
        period_return = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
        volume_ratio = window['volume'].iloc[-1] / window['volume'].mean()
        hl_ratio = np.log(window['high'] / window['low']).std()
        skewness = pd.Series(returns).skew() if len(returns) > 2 else 0
        kurtosis = pd.Series(returns).kurtosis() if len(returns) > 3 else 0
        consecutive_up = sum(1 for j in range(1, len(window)) if window['close'].iloc[j] > window['close'].iloc[j-1])
        highest = window['high'].max()
        lowest = window['low'].min()
        relative_pos = (window['close'].iloc[-1] - lowest) / (highest - lowest) if highest > lowest else 0.5
        
        feature_vector = [mean_return, volatility, price_range, period_return, volume_ratio, hl_ratio, skewness, kurtosis, consecutive_up, relative_pos]
        feature_vector = [0 if (np.isnan(x) or np.isinf(x)) else x for x in feature_vector]
        features_list.append(feature_vector)
    except:
        features_list.append([0] * 10)

X = np.array(features_list)
print(f"  OK Extracted {X.shape[0]:,} samples with {X.shape[1]} features")

# STEP 3: Label Creation (ZigZag Logic)
print("\n[3/6] Creating ZigZag labels...")

labels_list = []
for i in range(20, len(df) - 1):
    try:
        prev_high = df['high'].iloc[max(0, i-12):i].max()
        prev_low = df['low'].iloc[max(0, i-12):i].min()
        curr_high = df['high'].iloc[i]
        curr_low = df['low'].iloc[i]
        
        if i > 0:
            prev_bar_high = df['high'].iloc[i-1]
            prev_bar_low = df['low'].iloc[i-1]
            
            if curr_high > prev_high and curr_low > prev_bar_low:
                label = 0  # HH
            elif curr_high < prev_high and curr_low > prev_bar_low:
                label = 1  # HL
            elif curr_high < prev_high and curr_low < prev_bar_low:
                label = 2  # LH
            elif curr_high > prev_high and curr_low < prev_bar_low:
                label = 3  # LL
            else:
                label = 4
        else:
            label = 4
        labels_list.append(label)
    except:
        labels_list.append(4)

y = np.array(labels_list)
label_names = ['HH', 'HL', 'LH', 'LL', 'No Pattern']
label_dist = pd.Series(y).value_counts().sort_index()
print(f"  OK Created {len(y):,} labels")
for i, count in label_dist.items():
    pct = 100 * count / len(y)
    print(f"    {label_names[i]:12s}: {count:5d} ({pct:5.1f}%)")

# STEP 4: Data Preparation
print("\n[4/6] Preparing data...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

test_size = 0.2
split_idx = int(len(X_scaled) * (1 - test_size))

X_train_full = X_scaled[:split_idx]
y_train_full = y[:split_idx]
X_test = X_scaled[split_idx:]
y_test = y[split_idx:]

val_size = 0.2
val_idx = int(len(X_train_full) * (1 - val_size))

X_train = X_train_full[:val_idx]
y_train = y_train_full[:val_idx]
X_val = X_train_full[val_idx:]
y_val = y_train_full[val_idx:]

print(f"  OK Data split complete")
print(f"    Training:   {X_train.shape[0]:,} samples")
print(f"    Validation: {X_val.shape[0]:,} samples")
print(f"    Testing:    {X_test.shape[0]:,} samples")

# STEP 5: Model Training
print("\n[5/6] Training ensemble models...")

models = {}

print("  Training Random Forest...", end='', flush=True)
rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=3, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
models['RF'] = rf
rf_score = rf.score(X_val, y_val)
print(f" OK {rf_score:.2%}")

print("  Training Gradient Boosting...", end='', flush=True)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
models['GB'] = gb
gb_score = gb.score(X_val, y_val)
print(f" OK {gb_score:.2%}")

print("  Training Neural Network...", end='', flush=True)
nn = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), learning_rate_init=0.001, max_iter=500, batch_size=32, random_state=42, verbose=0)
nn.fit(X_train, y_train)
models['NN'] = nn
nn_score = nn.score(X_val, y_val)
print(f" OK {nn_score:.2%}")

# STEP 6: Evaluation
print("\n[6/6] Evaluating results...")

proba_list = []
for m in models.values():
    if hasattr(m, 'predict_proba'):
        proba = m.predict_proba(X_test)
    else:
        proba = np.eye(5)[m.predict(X_test)]
    proba_list.append(proba)

avg_proba = np.mean(proba_list, axis=0)
ensemble_pred = np.argmax(avg_proba, axis=0)

test_accuracy = accuracy_score(y_test, ensemble_pred)
avg_confidence = np.max(avg_proba, axis=1).mean()

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"\nTest Accuracy: {test_accuracy:.2%}")
print(f"Average Confidence: {avg_confidence:.2%}")

print(f"\nClassification Report:")
print(classification_report(y_test, ensemble_pred, target_names=label_names, zero_division=0))

cm = confusion_matrix(y_test, ensemble_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=label_names, yticklabels=label_names, cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix - ZigZag Predictions', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

class_acc = []
for i in range(5):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc.append(accuracy_score(y_test[mask], ensemble_pred[mask]))
    else:
        class_acc.append(0)

bars = axes[1].bar(label_names, class_acc, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[1].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_ylim([0, 1.05])
axes[1].axhline(y=test_accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall: {test_accuracy:.2%}')
axes[1].legend(fontsize=11)

for bar, acc in zip(bars, class_acc):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n{'='*70}")
print(f"Training Complete!")
print(f"Final Test Accuracy: {test_accuracy:.2%}")
print(f"Average Confidence: {avg_confidence:.2%}")
print(f"{'='*70}")

print("\nNext Steps:")
print("  1. Check the confusion matrix and per-class accuracy above")
print("  2. If accuracy < 80%, try adjusting hyperparameters")
print("  3. Ready to upload to Hugging Face when satisfied")
