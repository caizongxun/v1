# Google Colab Training Script for ZigZag ML Predictor
# 用於在Google Colab上訓練ZigZag HH/HL/LH/LL預測模型
# 目標準確率: 80-90%

# !pip install pandas numpy scikit-learn tensorflow xgboost yfinance loguru matplotlib seaborn plotly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PART 1: DATA PREPARATION
# =============================================================================

def fetch_market_data(ticker: str = 'BTC-USD', period: str = '2y', interval: str = '1h') -> pd.DataFrame:
    print(f"Fetching {ticker} data...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df.reset_index()
    df.columns = [col.lower() for col in df.columns]
    
    # Calculate technical indicators
    df['Returns'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    df = df.dropna()
    print(f"Loaded {len(df)} candles")
    return df


def create_advanced_features(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """
    Extract advanced features for 80-90% accuracy
    包含波动率、动量、支撑阻力等特征
    """
    features = []
    
    for i in range(lookback, len(df) - 1):
        window = df.iloc[i-lookback:i+1]
        
        # 1. Price Statistics
        returns = window['Close'].pct_change().dropna()
        high_low_ratio = window['High'] / window['Low']
        
        # 2. Volatility Measures
        volatility = returns.std()
        parkinson_volatility = np.sqrt(np.mean(np.log(window['High'] / window['Low'])**2) / (4 * np.log(2)))
        
        # 3. Trend Indicators
        sma_fast = window['Close'].rolling(3).mean().iloc[-1]
        sma_slow = window['Close'].rolling(10).mean().iloc[-1]
        trend = sma_fast / sma_slow - 1
        
        # 4. Momentum Indicators
        roc = (window['Close'].iloc[-1] - window['Close'].iloc[0]) / window['Close'].iloc[0]
        
        # 5. Support/Resistance (Recent highs and lows)
        recent_high = window['High'].iloc[-5:].max()
        recent_low = window['Low'].iloc[-5:].min()
        price_to_resistance = window['Close'].iloc[-1] / recent_high - 1
        price_to_support = window['Close'].iloc[-1] / recent_low - 1
        
        # 6. Volume Analysis
        avg_volume = window['Volume'].mean()
        volume_ratio = window['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
        
        # 7. Pattern Detection
        consecutive_higher_closes = 0
        consecutive_higher_highs = 0
        for j in range(1, len(window)):
            if window['Close'].iloc[j] > window['Close'].iloc[j-1]:
                consecutive_higher_closes += 1
            if window['High'].iloc[j] > window['High'].iloc[j-1]:
                consecutive_higher_highs += 1
        
        # 8. Volatility Regimes
        vol_ratio = volatility / window['Close'].std() if window['Close'].std() > 0 else 1
        
        # 9. ATR-like measure
        tr1 = window['High'].iloc[-1] - window['Low'].iloc[-1]
        tr2 = abs(window['High'].iloc[-1] - window['Close'].iloc[-2]) if len(window) > 1 else 0
        tr3 = abs(window['Low'].iloc[-1] - window['Close'].iloc[-2]) if len(window) > 1 else 0
        atr = max(tr1, tr2, tr3)
        
        # 10. Mean Reversion Indicators
        highest = window['High'].max()
        lowest = window['Low'].min()
        position = (window['Close'].iloc[-1] - lowest) / (highest - lowest) if highest > lowest else 0.5
        
        feature_vector = [
            returns.mean(),
            volatility,
            parkinson_volatility,
            trend,
            roc,
            price_to_resistance,
            price_to_support,
            volume_ratio,
            consecutive_higher_closes,
            consecutive_higher_highs,
            vol_ratio,
            atr / window['Close'].iloc[-1],
            position,
            returns.skew(),
            returns.kurtosis(),
            np.std(high_low_ratio),
        ]
        
        features.append(feature_vector)
    
    return np.array(features)


def prepare_data(df: pd.DataFrame, test_size: float = 0.2):
    print("Preparing data...")
    
    # Extract advanced features
    X = create_advanced_features(df, lookback=20)
    
    # Create labels based on ZigZag logic
    labels = []
    for i in range(20, len(df) - 1):
        prev_high = df['High'].iloc[max(0, i-12):i].max()
        prev_low = df['Low'].iloc[max(0, i-12):i].min()
        curr_high = df['High'].iloc[i]
        curr_low = df['Low'].iloc[i]
        
        label = 4  # Default
        
        if i > 0:
            prev_price_high = df['High'].iloc[i-1]
            prev_price_low = df['Low'].iloc[i-1]
            
            if curr_high > prev_high and curr_low > prev_price_low:
                label = 0  # HH
            elif curr_high < prev_high and curr_low > prev_price_low:
                label = 1  # HL
            elif curr_high < prev_high and curr_low < prev_price_low:
                label = 2  # LH
            elif curr_high > prev_high and curr_low < prev_price_low:
                label = 3  # LL
        
        labels.append(label)
    
    y = np.array(labels)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels distribution:\n{pd.Series(y).value_counts().sort_index()}")
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split (time series - no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, shuffle=False
    )
    
    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )
    
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def train_ensemble_models(X_train, X_val, y_train, y_val):
    """Train multiple models and ensemble them"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    
    print("\nTraining ensemble models...")
    
    models = {
        'RF': RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=3, 
                                    class_weight='balanced', random_state=42, n_jobs=-1),
        'GB': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=8,
                                        random_state=42),
        'NN': MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=500,
                           learning_rate_init=0.001, random_state=42),
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"  Training {name}...", end='')
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        trained_models[name] = model
        print(f" Val Accuracy: {val_score:.2%}")
    
    return trained_models


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*60)
    print("ZigZag ML Predictor - Colab Training Pipeline")
    print("="*60)
    
    # Step 1: Fetch data
    df = fetch_market_data(ticker='BTC-USD', period='2y', interval='1h')
    
    # Step 2: Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(df)
    
    # Step 3: Train ensemble
    models = train_ensemble_models(X_train, X_val, y_train, y_val)
    
    # Step 4: Ensemble prediction
    print("\nMaking ensemble predictions...")
    predictions_list = []
    proba_list = []
    
    for model in models.values():
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
        else:
            # For SVM, convert predictions to probabilities
            proba = np.eye(5)[model.predict(X_test)]
        proba_list.append(proba)
        predictions_list.append(model.predict(X_test))
    
    # Average predictions
    avg_proba = np.mean(proba_list, axis=0)
    ensemble_pred = np.argmax(avg_proba, axis=0)
    ensemble_conf = np.max(avg_proba, axis=0)
    
    # Step 5: Evaluate
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    test_accuracy = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble Test Accuracy: {test_accuracy:.2%}")
    print(f"Average Confidence: {ensemble_conf.mean():.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_pred, 
                              target_names=['HH', 'HL', 'LH', 'LL', 'No Pattern']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, ensemble_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['HH', 'HL', 'LH', 'LL', 'No Pattern'],
                yticklabels=['HH', 'HL', 'LH', 'LL', 'No Pattern'])
    axes[0].set_title('Confusion Matrix')
    
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
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)
    
    return models, scaler, ensemble_pred, y_test


if __name__ == "__main__":
    models, scaler, predictions, true_labels = main()
    print("\n準備好上傳到Hugging Face或用於實時交易!")
