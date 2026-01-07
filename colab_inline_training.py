# ZigZag ML Predictor - Pure Inline Training for Google Colab
# 完全自包含，不依賴文件系統
# 直接在 Colab cell 中執行即可

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

# ============================================================================
# STEP 1: Fetch Market Data
# ============================================================================
print("\n[1/6] Fetching market data...")

try:
    print("  Downloading BTC-USD 1h data (2 years)...")
    df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df.columns = [c.lower().strip() for c in df.columns]
    
    print(f"  ✓ Successfully loaded {len(df):,} candles")
    print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"    Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
except Exception as e:
    print(f"  ⚠ Data fetch failed: {e}")
    print("  Creating synthetic data for demo...")
    
    np.random.seed(42)
    n_samples = 2000
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq='1H')
    
    # Generate realistic synthetic OHLCV
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
    
    print(f"  ✓ Generated {len(df):,} synthetic candles")

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\n[2/6] Engineering features...")

features_list = []
errors = 0

for i in range(20, len(df) - 1):
    try:
        window = df.iloc[i-20:i+1]
        
        # Price returns
        returns = window['close'].pct_change().dropna().values
        
        # Feature 1: Mean return
        mean_return = np.mean(returns) if len(returns) > 0 else 0
        
        # Feature 2: Volatility
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Feature 3: Price range
        price_range = (window['high'].max() - window['low'].min()) / window['close'].mean()
        
        # Feature 4: Return on period
        period_return = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
        
        # Feature 5: Volume ratio
        volume_ratio = window['volume'].iloc[-1] / window['volume'].mean()
        
        # Feature 6: High-Low volatility (Parkinson)
        hl_ratio = np.log(window['high'] / window['low']).std()
        
        # Feature 7: Skewness
        skewness = pd.Series(returns).skew() if len(returns) > 2 else 0
        
        # Feature 8: Kurtosis
        kurtosis = pd.Series(returns).kurtosis() if len(returns) > 3 else 0
        
        # Feature 9: Consecutive up closes
        consecutive_up = sum(1 for j in range(1, len(window)) if window['close'].iloc[j] > window['close'].iloc[j-1])
        
        # Feature 10: Relative position (High-Low position)
        highest = window['high'].max()
        lowest = window['low'].min()
        relative_pos = (window['close'].iloc[-1] - lowest) / (highest - lowest) if highest > lowest else 0.5
        
        # Combine all features
        feature_vector = [
            mean_return,
            volatility,
            price_range,
            period_return,
            volume_ratio,
            hl_ratio,
            skewness,
            kurtosis,
            consecutive_up,
            relative_pos
        ]
        
        # Replace NaN/Inf with 0
        feature_vector = [0 if (np.isnan(x) or np.isinf(x)) else x for x in feature_vector]
        features_list.append(feature_vector)
        
    except Exception as e:
        # Append default feature vector on error
        features_list.append([0] * 10)
        errors += 1

X = np.array(features_list)
print(f"  ✓ Extracted {X.shape[0]:,} samples with {X.shape[1]} features")
if errors > 0:
    print(f"    (Skipped {errors} samples with errors)")

# ============================================================================
# STEP 3: Label Creation (ZigZag-like Logic)
# ============================================================================
print("\n[3/6] Creating ZigZag labels...")

labels_list = []
depth = 12

for i in range(20, len(df) - 1):
    try:
        # Get previous pivot highs/lows
        prev_high = df['high'].iloc[max(0, i-depth):i].max()
        prev_low = df['low'].iloc[max(0, i-depth):i].min()
        
        # Current bar
        curr_high = df['high'].iloc[i]
        curr_low = df['low'].iloc[i]
        
        # Previous bar
        if i > 0:
            prev_bar_high = df['high'].iloc[i-1]
            prev_bar_low = df['low'].iloc[i-1]
            
            # ZigZag logic (simplified from Pine)
            if curr_high > prev_high and curr_low > prev_bar_low:
                label = 0  # HH (Higher High)
            elif curr_high < prev_high and curr_low > prev_bar_low:
                label = 1  # HL (Higher Low)
            elif curr_high < prev_high and curr_low < prev_bar_low:
                label = 2  # LH (Lower High)
            elif curr_high > prev_high and curr_low < prev_bar_low:
                label = 3  # LL (Lower Low)
            else:
                label = 4  # No significant pattern
        else:
            label = 4
        
        labels_list.append(label)
    except:
        labels_list.append(4)

y = np.array(labels_list)

label_names = ['HH', 'HL', 'LH', 'LL', 'No Pattern']
label_dist = pd.Series(y).value_counts().sort_index()
print(f"  ✓ Created {len(y):,} labels")
print(f"    Distribution:")
for i, count in label_dist.items():
    pct = 100 * count / len(y)
    print(f"      {label_names[i]:12s}: {count:5d} ({pct:5.1f}%)")

# ============================================================================
# STEP 4: Data Preparation
# ============================================================================
print("\n[4/6] Preparing data...")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time-series split (no shuffle to preserve temporal order)
test_size = 0.2
split_idx = int(len(X_scaled) * (1 - test_size))

X_train_full = X_scaled[:split_idx]
y_train_full = y[:split_idx]
X_test = X_scaled[split_idx:]
y_test = y[split_idx:]

# Further split training into train/val
val_size = 0.2
val_idx = int(len(X_train_full) * (1 - val_size))

X_train = X_train_full[:val_idx]
y_train = y_train_full[:val_idx]
X_val = X_train_full[val_idx:]
y_val = y_train_full[val_idx:]

print(f"  ✓ Data split complete")
print(f"    Training:   {X_train.shape[0]:,} samples")
print(f"    Validation: {X_val.shape[0]:,} samples")
print(f"    Testing:    {X_test.shape[0]:,} samples")

# ============================================================================
# STEP 5: Model Training
# ============================================================================
print("\n[5/6] Training ensemble models...")

models = {}

# Random Forest
print("  Training Random Forest...", end='', flush=True)
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_val_score = rf_model.score(X_val, y_val)
models['RF'] = rf_model
print(f" ✓ Val Accuracy: {rf_val_score:.2%}")

# Gradient Boosting
print("  Training Gradient Boosting...", end='', flush=True)
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_val_score = gb_model.score(X_val, y_val)
models['GB'] = gb_model
print(f" ✓ Val Accuracy: {gb_val_score:.2%}")

# Neural Network
print("  Training Neural Network...", end='', flush=True)
nn_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    learning_rate_init=0.001,
    max_iter=500,
    batch_size=32,
    random_state=42,
    verbose=0
)
nn_model.fit(X_train, y_train)
nn_val_score = nn_model.score(X_val, y_val)
models['NN'] = nn_model
print(f" ✓ Val Accuracy: {nn_val_score:.2%}")

# ============================================================================
# STEP 6: Evaluation & Visualization
# ============================================================================
print("\n[6/6] Evaluating and visualizing results...")

# Make ensemble predictions
proba_list = []
for model in models.values():
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
    else:
        # Fallback for models without predict_proba
        preds = model.predict(X_test)
        proba = np.eye(5)[preds]
    proba_list.append(proba)

# Average probabilities
avg_proba = np.mean(proba_list, axis=0)
ensemble_pred = np.argmax(avg_proba, axis=0)
ensemble_conf = np.max(avg_proba, axis=0)

# Calculate metrics
test_accuracy = accuracy_score(y_test, ensemble_pred)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"\n✓ Ensemble Test Accuracy: {test_accuracy:.2%}")
print(f"✓ Average Confidence: {ensemble_conf.mean():.2%}")

print(f"\nClassification Report:")
print(classification_report(
    y_test, ensemble_pred,
    target_names=label_names,
    zero_division=0
))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, ensemble_pred)
print(cm)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix Heatmap
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
    xticklabels=label_names,
    yticklabels=label_names,
    cbar_kws={'label': 'Count'}
)
axes[0].set_title('Confusion Matrix - ZigZag Predictions', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Per-Class Accuracy
class_accuracies = []
for i in range(5):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test[mask], ensemble_pred[mask])
        class_accuracies.append(class_acc)
    else:
        class_accuracies.append(0)

bars = axes[1].bar(label_names, class_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[1].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_ylim([0, 1.05])
axes[1].axhline(y=test_accuracy, color='red', linestyle='--', label=f'Overall: {test_accuracy:.2%}')
axes[1].legend()

# Add value labels on bars
for bar, acc in zip(bars, class_accuracies):
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width()/2., height + 0.02,
        f'{acc:.1%}', ha='center', va='bottom', fontsize=10
    )

plt.tight_layout()
plt.show()

print(f"\n{'='*70}")
print(f"✓ Training Complete!")
print(f"  Final Test Accuracy: {test_accuracy:.2%}")
print(f"  Average Confidence: {ensemble_conf.mean():.2%}")
print(f"{'='*70}")

print(f"\nNext Steps:")
print(f"  1. Check the confusion matrix and per-class accuracy above")
print(f"  2. If accuracy < 80%, try adjusting hyperparameters")
print(f"  3. Ready to upload to Hugging Face when satisfied")
