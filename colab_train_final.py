#!/usr/bin/env python3
"""ZigZag ML Predictor - Fixed Inline Training for Google Colab"""

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
print("ZigZag ML Predictor - Training Pipeline")
print("="*70)

# STEP 1: Fetch data
print("\n[1/6] Fetching market data...")
try:
    df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"  OK Loaded {len(df):,} candles")
except Exception as e:
    print(f"  Using synthetic data ({e})...")
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
print("\n[2/6] Engineering features...")
features_list = []

for i in range(20, len(df) - 1):
    try:
        window = df.iloc[i-20:i+1]
        returns = window['close'].pct_change().dropna().values
        
        mean_ret = np.mean(returns) if len(returns) > 0 else 0
        vol = np.std(returns) if len(returns) > 1 else 0
        price_range = (window['high'].max() - window['low'].min()) / window['close'].mean()
        period_ret = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
        vol_ratio = window['volume'].iloc[-1] / window['volume'].mean()
        hl_ratio = np.log(window['high'] / window['low']).std()
        skew = pd.Series(returns).skew() if len(returns) > 2 else 0
        kurt = pd.Series(returns).kurtosis() if len(returns) > 3 else 0
        cons_up = sum(1 for j in range(1, len(window)) if window['close'].iloc[j] > window['close'].iloc[j-1])
        rel_pos = (window['close'].iloc[-1] - window['low'].min()) / (window['high'].max() - window['low'].min()) if window['high'].max() > window['low'].min() else 0.5
        
        fv = [mean_ret, vol, price_range, period_ret, vol_ratio, hl_ratio, skew, kurt, cons_up, rel_pos]
        fv = [0 if (np.isnan(x) or np.isinf(x)) else x for x in fv]
        features_list.append(fv)
    except Exception as e:
        features_list.append([0]*10)

X = np.array(features_list)
print(f"  OK {X.shape[0]:,} features extracted")

# STEP 3: Labels - IMPORTANT: Match the same index range as features
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
                label = 4  # No Pattern
        else:
            label = 4
        
        labels_list.append(label)
    except Exception as e:
        labels_list.append(4)

y = np.array(labels_list)

# ENSURE SAME LENGTH
min_len = min(len(X), len(y))
X = X[:min_len]
y = y[:min_len]

print(f"  OK {len(y):,} labels created")
print(f"  Total samples: {len(X):,}")

label_names = ['HH', 'HL', 'LH', 'LL', 'No Pattern']
label_dist = pd.Series(y).value_counts().sort_index()
for i, count in label_dist.items():
    pct = 100 * count / len(y)
    print(f"    {label_names[i]:12s}: {count:6d} ({pct:5.1f}%)")

# STEP 4: Data prep
print("\n[4/6] Preparing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time-series split (no shuffle)
test_size = 0.2
train_val_size = int(len(X_scaled) * (1 - test_size))

X_train_val = X_scaled[:train_val_size]
y_train_val = y[:train_val_size]
X_test = X_scaled[train_val_size:]
y_test = y[train_val_size:]

# Validation split
val_size = 0.2
train_size = int(len(X_train_val) * (1 - val_size))

X_train = X_train_val[:train_size]
y_train = y_train_val[:train_size]
X_val = X_train_val[train_size:]
y_val = y_train_val[train_size:]

print(f"  OK Data split:")
print(f"    Training:   {X_train.shape[0]:,} samples")
print(f"    Validation: {X_val.shape[0]:,} samples")
print(f"    Testing:    {X_test.shape[0]:,} samples")

# STEP 5: Train
print("\n[5/6] Training ensemble models...")

print("  Random Forest...", end='', flush=True)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_score = rf.score(X_val, y_val)
print(f" OK {rf_score:.2%}")

print("  Gradient Boosting...", end='', flush=True)
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
gb_score = gb.score(X_val, y_val)
print(f" OK {gb_score:.2%}")

print("  Neural Network...", end='', flush=True)
nn = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    learning_rate_init=0.001,
    max_iter=500,
    batch_size=32,
    random_state=42,
    verbose=0
)
nn.fit(X_train, y_train)
nn_score = nn.score(X_val, y_val)
print(f" OK {nn_score:.2%}")

# STEP 6: Evaluate
print("\n[6/6] Evaluating results...")

models = {'RF': rf, 'GB': gb, 'NN': nn}
proba_list = []

for m in models.values():
    if hasattr(m, 'predict_proba'):
        proba = m.predict_proba(X_test)
    else:
        # For models without predict_proba, create artificial probabilities
        preds = m.predict(X_test)
        proba = np.eye(5)[preds]
    proba_list.append(proba)

# Average ensemble predictions
avg_proba = np.mean(proba_list, axis=0)
ensemble_pred = np.argmax(avg_proba, axis=0)
ensemble_conf = np.max(avg_proba, axis=1)

test_accuracy = accuracy_score(y_test, ensemble_pred)
avg_confidence = ensemble_conf.mean()

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"\n✓ Test Accuracy: {test_accuracy:.2%}")
print(f"✓ Average Confidence: {avg_confidence:.2%}")

print(f"\nClassification Report:")
print(classification_report(y_test, ensemble_pred, target_names=label_names, zero_division=0))

cm = confusion_matrix(y_test, ensemble_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
    xticklabels=label_names, yticklabels=label_names,
    cbar_kws={'label': 'Count'}
)
axes[0].set_title('Confusion Matrix - ZigZag Predictions', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Per-Class Accuracy
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
axes[1].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_ylim([0, 1.05])
axes[1].axhline(
    y=test_accuracy, color='red', linestyle='--', linewidth=2,
    label=f'Overall: {test_accuracy:.2%}'
)
axes[1].legend(fontsize=11)

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
print(f"✓ Training Complete!")
print(f"  Final Test Accuracy: {test_accuracy:.2%}")
print(f"  Average Confidence: {avg_confidence:.2%}")
print(f"{'='*70}")

if test_accuracy >= 0.80:
    print("\n🎉 EXCELLENT! Accuracy >= 80%")
elif test_accuracy >= 0.70:
    print("\n👍 GOOD! Accuracy >= 70%")
else:
    print("\n⚠️  Consider improving: Try more epochs or different hyperparameters")

print("\nNext Steps:")
print("  1. Review confusion matrix above")
print("  2. If satisfied with accuracy, ready to upload to Hugging Face")
print("  3. For higher accuracy, try adjusting hyperparameters")
