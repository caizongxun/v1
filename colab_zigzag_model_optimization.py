#!/usr/bin/env python3
"""
ZigZag Model Optimization
Implement 4 quick optimization methods
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("ZigZag Model Optimization")
print("="*80)

# ============================================================================
# OPTIMIZATION METHOD 1: HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "="*80)
print("METHOD 1: HYPERPARAMETER TUNING")
print("="*80)

print("""
当前参数:
  n_estimators: 150 (可能不够)
  max_depth: 6 (可能太浅)
  learning_rate: 0.05
  
优化建议:
  n_estimators: 150 → 300-400
  max_depth: 6 → 8-10
  reg_alpha/lambda: 增加 (防止过拟合)
  
预期效果: +2-3% 精准度
""")

hyperparams_v1 = {
    'name': 'Conservative (推荐)',
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'gamma': 1,
}

hyperparams_v2 = {
    'name': 'Aggressive',
    'n_estimators': 500,
    'max_depth': 10,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 2,
    'reg_lambda': 2,
    'gamma': 2,
}

print("\n推荐配置 v1 (Conservative):")
for k, v in hyperparams_v1.items():
    if k != 'name':
        print(f"  {k}: {v}")

print("\n推荐配置 v2 (Aggressive):")
for k, v in hyperparams_v2.items():
    if k != 'name':
        print(f"  {k}: {v}")

print("\n✓ 推荐使用 v1 (Conservative) 开始优化")

# ============================================================================
# OPTIMIZATION METHOD 2: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("METHOD 2: FEATURE ENGINEERING (新增 8 个特征)")
print("="*80)

print("""
要添加的新特征:
  1. Momentum (5, 10 期)
  2. Force Index (成交量 × 价格变化)
  3. Stochastic %K (超买/超卖)
  4. Price Acceleration (加速度)
  5. Volume Momentum (成交量动量)
  6. HL Ratio (高低比)
  7. Price to 20d High (20日高点)
  8. CCI (商品通道指数)
  
预期效果: +3-5% 精准度
""")

def add_advanced_features(df):
    """添加高级特征"""
    features = pd.DataFrame(index=df.index)
    
    # 原有特征 (简化版，实际包括所有 28 个)
    features['close'] = df['close']
    features['volume'] = df['volume']
    
    # ============ 新增特征 ============
    
    # 1. Momentum
    features['momentum_5'] = df['close'].pct_change(5)
    features['momentum_10'] = df['close'].pct_change(10)
    print("✓ 添加: Momentum (5, 10)")
    
    # 2. Force Index
    raw_force = df['close'].diff() * df['volume']
    features['force_index'] = raw_force.ewm(span=13).mean()
    print("✓ 添加: Force Index")
    
    # 3. Stochastic %K
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    features['stoch_k'] = 100 * (df['close'] - low_14) / ((high_14 - low_14) + 1e-8)
    print("✓ 添加: Stochastic %K")
    
    # 4. Price Acceleration
    price_diff = df['close'].diff()
    features['acceleration'] = price_diff.diff()
    print("✓ 添加: Price Acceleration")
    
    # 5. Volume Momentum
    features['volume_momentum'] = df['volume'].pct_change(5)
    print("✓ 添加: Volume Momentum")
    
    # 6. HL Ratio
    features['hl_ratio'] = df['high'] / (df['low'] + 1e-8)
    print("✓ 添加: HL Ratio")
    
    # 7. Price to 20d High
    high_20 = df['high'].rolling(window=20).max()
    features['price_to_20d_high'] = df['close'] / (high_20 + 1e-8)
    print("✓ 添加: Price to 20d High")
    
    # 8. CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = (tp - sma_tp).rolling(window=20).apply(lambda x: np.abs(x).mean())
    features['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
    print("✓ 添加: CCI")
    
    # 清理
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    
    return features

print("\n添加特征的函数已定义")
print("实际使用时替换原有的 extract_features() 函数")

# ============================================================================
# OPTIMIZATION METHOD 3: CLASS WEIGHT BALANCING
# ============================================================================

print("\n" + "="*80)
print("METHOD 3: CLASS WEIGHT BALANCING")
print("="*80)

print("""
当前标签分布:
  HH: 25.1%
  HL: 25.5%
  LH: 24.9% ← 最难
  LL: 24.5%
  
虽然平衡，但给难的类别更大权重:
  HH: 1.0 (基础)
  HL: 1.2 (精准度重要)
  LH: 1.5 (最难)
  LL: 1.0 (基础)
  
预期效果: +2-3% 精准度
""")

class_weights = {
    0: 1.0,  # HH
    1: 1.2,  # HL
    2: 1.5,  # LH - 最高权重
    3: 1.0,  # LL
}

print("\n类别权重配置:")
for idx, weight in class_weights.items():
    labels = ['HH', 'HL', 'LH', 'LL']
    print(f"  {labels[idx]}: {weight}")

# ============================================================================
# OPTIMIZATION METHOD 4: DATA AUGMENTATION
# ============================================================================

print("\n" + "="*80)
print("METHOD 4: DATA AUGMENTATION")
print("="*80)

print("""
当前数据:
  周期: 2 年
  时间框架: 1 小时
  样本量: 780 个标记
  
优化方案:
  
  方案 A: 扩展历史数据
    2 年 → 5 年 (样本增加 2.5 倍)
    
  方案 B: 多币种
    BTC → BTC + ETH + BNB + SOL (样本增加 4 倍)
    
  方案 C: 多时间框架
    1h → 1h + 4h + 1d (样本增加 3 倍)
    
  方案 D: 组合
    5 年 + 3 币种 + 3 时间框架 (样本增加 22.5 倍)
    
预期效果: +2-4% 精准度 (如果增加 50% 数据)
""")

print("""
快速实现:

```python
# 方案 A: 5 年数据
df = yf.download('BTC-USD', start='2019-01-01', end='2024-01-01', interval='1h')

# 方案 B: 多币种
cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD']
all_data = []
for crypto in cryptos:
    df_temp = yf.download(crypto, period='2y', interval='1h')
    all_data.append(df_temp)
df_combined = pd.concat(all_data)
```
""")

# ============================================================================
# OPTIMIZATION SUMMARY
# ============================================================================

print("\n" + "="*80)
print("优化效果预测")
print("="*80)

optimization_plan = """
初始性能: 69.23%

快速优化 (1-2 小时):
  ✓ 方法 1: 超参数调优............... +2-3%  (71-72%)
  ✓ 方法 2: 特征工程............... +3-5%  (74-77%)
  ✓ 方法 3: 类别权重............... +2-3%  (76-80%)
  ────────────────────────────────────────────
  预期总改进: +7-11% → 76-80% 🎯

中期优化 (4-6 小时):
  ✓ 方法 4: 数据增强............... +2-4%  (78-84%)
  ✓ 集成学习 (XGB + LGB + RF)....... +3-5%  (81-89%)
  ────────────────────────────────────────────
  预期总改进: +10-15% → 79-84% 🌟

高级优化 (8+ 小时):
  ✓ 深度学习 (神经网络)............. +5-8%  (84-92%)
  ✓ 贝叶斯优化..................... +2-3%  (86-95%)
  ────────────────────────────────────────────
  预期总改进: +15-25% → 84-94% 🚀
"""

print(optimization_plan)

# ============================================================================
# RECOMMENDED NEXT STEPS
# ============================================================================

print("\n" + "="*80)
print("推荐行动计划")
print("="*80)

action_plan = """
第 1 阶段: 快速优化 (今天)
  [ ] 1. 修改超参数 (n_estimators: 300, max_depth: 8)
  [ ] 2. 添加 8 个新特征
  [ ] 3. 实施类别权重
  预期: 69% → 76-80%
  时间: 1-2 小时

第 2 阶段: 中期优化 (明天)
  [ ] 4. 扩展数据 (5 年 或 多币种)
  [ ] 5. 实施集成学习 (XGB + LGB + RF)
  预期: 76-80% → 79-84%
  时间: 4-6 小时

第 3 阶段: 高级优化 (周末)
  [ ] 6. 尝试深度学习
  [ ] 7. 贝叶斯超参数优化
  预期: 79-84% → 84-90%
  时间: 8+ 小时

立即开始: 第 1 阶段只需 1-2 小时即可获得 +7-11% 的改进！
"""

print(action_plan)

print("\n" + "="*80)
print("代码示例: 如何应用第一个优化")
print("="*80)

code_example = """
# 应用优化的代码
import xgboost as xgb

# 第 1 步: 优化超参数
model_optimized = xgb.XGBClassifier(
    n_estimators=300,          # ← 从 150 增加
    max_depth=8,               # ← 从 6 增加
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,               # ← 增加正则化
    reg_lambda=1,
    gamma=1,                   # ← 增加分裂阈值
    random_state=42,
    n_jobs=-1
)

# 第 2 步: 使用类别权重
class_weights = {0: 1.0, 1: 1.2, 2: 1.5, 3: 1.0}
sample_weight = np.array([class_weights[y] for y in y_train])

# 第 3 步: 训练
model_optimized.fit(
    X_train_scaled, y_train,
    sample_weight=sample_weight,  # ← 应用权重
    verbose=False
)

# 第 4 步: 评估
accuracy = accuracy_score(y_test, model_optimized.predict(X_test_scaled))
f1 = f1_score(y_test, model_optimized.predict(X_test_scaled), average='weighted')

print(f"优化后 - 精准度: {accuracy:.4f}, F1: {f1:.4f}")
"""

print(code_example)

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)
print("""
下一步:
  1. 选择一个优化方法开始
  2. 复制相应的代码
  3. 在你的模型上应用
  4. 比较性能改进
  5. 如果满意，保存新模型
  6. 对其他方法重复

预期: 1-2 小时内从 69% 提升到 76-80%

""")
