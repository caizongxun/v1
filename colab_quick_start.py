#!/usr/bin/env python3
# ZigZag ML Predictor - Quick Start for Google Colab
# 直接複制整個 cell 的內容來 Colab 執行

import os
import sys

# ============================================================================
# SETUP: Install dependencies
# ============================================================================
print("[1/4] Installing dependencies...")
os.system('pip install -q pandas numpy scikit-learn xgboost yfinance matplotlib seaborn plotly joblib')
print("✓ Dependencies installed\n")

# ============================================================================
# FETCH: Download training script from GitHub
# ============================================================================
print("[2/4] Downloading training script from GitHub...")
os.system('wget -q https://raw.githubusercontent.com/caizongxun/v1/main/notebooks/train_zigzag_colab_fixed.py')

if not os.path.exists('train_zigzag_colab_fixed.py'):
    print("✗ Failed to download training script")
    print("Trying alternative method...")
    
    # Fallback: Create training script inline
    print("Creating training script inline...")
    
    training_code = '''
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
import joblib
warnings.filterwarnings('ignore')

print("="*60)
print("ZigZag ML Predictor - Colab Training Pipeline")
print("="*60)

# Fetch data
print("\\n[Step 1] Fetching market data...")
try:
    df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    print(f"✓ Loaded {len(df)} candles")
except:
    print("⚠ Using dummy data")
    np.random.seed(42)
    n = 2000
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='1H')
    close = 40000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({
        'datetime': dates,
        'open': close + np.random.randn(n) * 50,
        'high': close + np.abs(np.random.randn(n) * 50),
        'low': close - np.abs(np.random.randn(n) * 50),
        'close': close,
        'volume': np.random.uniform(100000, 500000, n)
    })

# Feature extraction
print("\\n[Step 2] Extracting features...")
features = []
for i in range(20, len(df) - 1):
    window = df.iloc[i-20:i+1]
    returns = window['close'].pct_change().dropna()
    
    f_vec = [
        returns.mean() if len(returns) > 0 else 0,
        returns.std() if len(returns) > 0 else 0,
        (window['high'].max() - window['low'].min()) / window['close'].mean(),
        (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0],
        window['volume'].iloc[-1] / window['volume'].mean() if window['volume'].mean() > 0 else 1,
        returns.skew() if len(returns) > 1 else 0,
        returns.kurtosis() if len(returns) > 1 else 0,
    ]
    f_vec = [0 if (np.isnan(x) or np.isinf(x)) else x for x in f_vec]
    features.append(f_vec)

X = np.array(features)
print(f"✓ Extracted {X.shape[0]} samples, {X.shape[1]} features")

# Labels (ZigZag-like)
print("\\n[Step 3] Creating labels...")
labels = []
for i in range(20, len(df) - 1):
    prev_high = df['high'].iloc[max(0, i-12):i].max()
    prev_low = df['low'].iloc[max(0, i-12):i].min()
    curr_high = df['high'].iloc[i]
    curr_low = df['low'].iloc[i]
    
    if i > 0:
        prev_price_high = df['high'].iloc[i-1]
        prev_price_low = df['low'].iloc[i-1]
        
        if curr_high > prev_high and curr_low > prev_price_low:
            label = 0  # HH
        elif curr_high < prev_high and curr_low > prev_price_low:
            label = 1  # HL
        elif curr_high < prev_high and curr_low < prev_price_low:
            label = 2  # LH
        elif curr_high > prev_high and curr_low < prev_price_low:
            label = 3  # LL
        else:
            label = 4  # No Pattern
    else:
        label = 4
    labels.append(label)

y = np.array(labels)
print(f"✓ Labels distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

# Data preparation
print("\\n[Step 4] Preparing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=False
)

print(f"✓ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# Training
print("\\n[Step 5] Training ensemble models...")
models = {
    'RF': RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', 
                                 random_state=42, n_jobs=-1),
    'GB': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=8, 
                                     random_state=42),
    'NN': MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=500, 
                        learning_rate_init=0.001, random_state=42),
}

trained_models = {}
for name, model in models.items():
    print(f"  Training {name}...", end='', flush=True)
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)
    trained_models[name] = model
    print(f" ✓ {val_score:.2%}")

# Prediction
print("\\n[Step 6] Making predictions...")
predictions_list = []
proba_list = []

for model in trained_models.values():
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
    else:
        proba = np.eye(5)[model.predict(X_test)]
    proba_list.append(proba)

avg_proba = np.mean(proba_list, axis=0)
ensemble_pred = np.argmax(avg_proba, axis=0)
ensemble_conf = np.max(avg_proba, axis=0)

# Results
print("\\n" + "="*60)
print("RESULTS")
print("="*60)

test_accuracy = accuracy_score(y_test, ensemble_pred)
print(f"\\n✓ Ensemble Test Accuracy: {test_accuracy:.2%}")
print(f"✓ Average Confidence: {ensemble_conf.mean():.2%}")

print("\\nClassification Report:")
print(classification_report(y_test, ensemble_pred, 
                          target_names=['HH', 'HL', 'LH', 'LL', 'No Pattern'],
                          zero_division=0))

cm = confusion_matrix(y_test, ensemble_pred)
print("\\nConfusion Matrix:")
print(cm)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['HH', 'HL', 'LH', 'LL', 'No Pattern'],
            yticklabels=['HH', 'HL', 'LH', 'LL', 'No Pattern'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

class_acc = []
for i in range(5):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc.append(accuracy_score(y_test[mask], ensemble_pred[mask]))
    else:
        class_acc.append(0)

axes[1].bar(['HH', 'HL', 'LH', 'LL', 'No Pattern'], class_acc, color='steelblue')
axes[1].set_title('Per-Class Accuracy')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1])
for i, v in enumerate(class_acc):
    axes[1].text(i, v + 0.02, f'{v:.1%}', ha='center')

plt.tight_layout()
plt.savefig('zigzag_results.png', dpi=100, bbox_inches='tight')
print("\\n✓ Results saved to zigzag_results.png")
plt.show()

# Save models
joblib.dump(scaler, 'zigzag_scaler.pkl')
joblib.dump(trained_models, 'zigzag_models.pkl')
print("✓ Models saved (zigzag_scaler.pkl, zigzag_models.pkl)")

print("\\n" + "="*60)
print(f"✓ Training complete! Accuracy: {test_accuracy:.2%}")
print("="*60)
'''
    
    with open('train_zigzag_colab_fixed.py', 'w') as f:
        f.write(training_code)
    print("✓ Training script created inline")
else:
    print("✓ Training script downloaded\n")

# ============================================================================
# EXECUTE: Run the training
# ============================================================================
print("[3/4] Running training...\n")
exec(open('train_zigzag_colab_fixed.py').read())

print("\n[4/4] Complete!")
print("\nGenerated files:")
print("  - zigzag_results.png (confusion matrix + per-class accuracy)")
print("  - zigzag_scaler.pkl (feature scaler)")
print("  - zigzag_models.pkl (trained models)")
print("\nYou can now download these files or upload to Hugging Face!")
