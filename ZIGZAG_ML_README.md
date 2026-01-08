# ZigZag Label Predictor - Complete ML Pipeline

## Overview

这个项目使用机器学习模型来预测 ZigZag 图表模式中的 **HH (Higher High)、HL (Higher Low)、LH (Lower High)、LL (Lower Low)** 标记，以解决 ZigZag 的滞后性问题。

**核心问题**：ZigZag 是一个滞后指标 - 你只有在价格已经形成之后才知道是 HH 还是 HL。

**解决方案**：训练 ML 模型在标记形成 **之前** 预测它。

---

## 项目结构

```
├── colab_zigzag_fixed_lh_hl.py      # ZigZag 标签生成 (核心算法)
├── colab_zigzag_ml_predictor.py     # ML 模型训练
├── colab_zigzag_backtest.py         # 模型评估和可视化
└── ZIGZAG_ML_README.md              # 本文件
```

---

## 快速开始

### 步骤 1：在 Colab 中运行模型训练

```python
import urllib.request
url = "https://raw.githubusercontent.com/caizongxun/v1/main/colab_zigzag_ml_predictor.py"
code = urllib.request.urlopen(url).read().decode('utf-8')
exec(code)
```

这会：
- 从 Yahoo Finance 下载 2 年 BTC-USD 1 小时数据
- 生成 ZigZag 标签（HH/HL/LH/LL）
- 提取 30+ 个技术指标特征
- 训练 XGBoost 分类模型
- 保存模型到 `zigzag_predictor_model.pkl`

### 步骤 2：评估模型

```python
import urllib.request
url = "https://raw.githubusercontent.com/caizongxun/v1/main/colab_zigzag_backtest.py"
code = urllib.request.urlopen(url).read().decode('utf-8')
exec(code)
```

这会生成：
- 混淆矩阵（Confusion Matrix）
- 特征重要性排名
- 预测置信度分布
- 分类报告（Precision/Recall/F1）

---

## 数据流程

```
1. 原始 K 线数据
   ↓
2. ZigZag 算法 (depth=3, deviation=2%)
   ↓
3. 标签生成 (HH/HL/LH/LL)
   ↓
4. 特征提取 (RSI, MACD, Bollinger Bands, 等)
   ↓
5. 数据分割 (80% 训练, 10% 验证, 10% 测试)
   ↓
6. XGBoost 模型训练
   ↓
7. 模型评估
   ↓
8. 生成预测概率
```

---

## 特征说明

### 价格特征
- `close`, `high`, `low` - 原始价格
- `ret_1`, `ret_5`, `ret_10` - 1/5/10 根 K 线收益率
- `volatility_5`, `volatility_10` - 5/10 根 K 线波动率

### 动量指标
- `rsi` - 相对强弱指数 (14 周期)
- `macd`, `macd_signal`, `macd_hist` - MACD 指标

### 趋势指标
- `sma_5`, `sma_10`, `sma_20` - 简单移动平均线
- `price_to_sma5`, `price_to_sma20` - 价格相对 MA 的位置

### 波动率指标
- `atr` - 平均真实波幅 (14 周期)
- `bb_upper`, `bb_lower`, `bb_position` - 布林带
- `high_low_range` - K 线真实波幅

### 成交量特征
- `volume_sma_ratio` - 成交量与 MA 的比率
- `volume_change` - 成交量变化率

---

## 模型配置

```python
xgb.XGBClassifier(
    n_estimators=200,      # 决策树数量
    max_depth=7,           # 树深度（防止过拟合）
    learning_rate=0.05,    # 学习率
    subsample=0.8,         # 行采样率
    colsample_bytree=0.8,  # 列采样率
    random_state=42,
    n_jobs=-1,             # 使用所有 CPU 核心
)
```

### 参数调整

| 参数 | 效果 | 建议值 |
|------|------|--------|
| `n_estimators` | 更多树 = 更好但更慢 | 200-500 |
| `max_depth` | 更深 = 更复杂但易过拟合 | 5-10 |
| `learning_rate` | 更小 = 更稳定但更慢 | 0.01-0.1 |
| `subsample` | 小于 1 = 随机森林效果 | 0.7-0.9 |

---

## 模型输出解释

### 1. 混淆矩阵

```
           Predicted
         HH  HL  LH  LL
Actual HH [ ][ ][ ][ ]  ← 对角线高 = 模型准确
HL [ ][ ][ ][ ]
LH [ ][ ][ ][ ]
LL [ ][ ][ ][ ]
```

- **对角线**：正确预测
- **非对角线**：错误预测

### 2. 分类报告

```
        Precision  Recall  F1-Score
HH      0.85       0.80    0.82
HL      0.72       0.75    0.73
LH      0.70       0.72    0.71
LL      0.88       0.85    0.86
```

- **Precision**：预测正确的比例
- **Recall**：实际正确的被找到的比例
- **F1-Score**：Precision 和 Recall 的调和平均

### 3. 特征重要性

前 5 个最重要的特征通常是：
1. RSI - 超买/超卖信号
2. MACD - 趋势强度
3. Bollinger Bands Position - 价格位置
4. ATR - 波动率
5. Moving Average Ratios - 趋势确认

---

## 预测方式

### 获取概率预测

```python
import pickle
from sklearn.preprocessing import StandardScaler

# 加载模型
with open('zigzag_predictor_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 新数据特征 (shape: [n_samples, n_features])
X_new = extract_features(new_data)  # 你的特征提取函数

# 获取预测概率
proba = model.model.predict_proba(X_new)
# 返回 shape: [n_samples, 4]
# 列的顺序: [HH, HL, LH, LL]

# 转换概率为标记
max_proba = np.max(proba, axis=1)
predicted_label_idx = np.argmax(proba, axis=1)
label_map = {0: 'HH', 1: 'HL', 2: 'LH', 3: 'LL'}
predicted_labels = [label_map[i] for i in predicted_label_idx]
confidence = max_proba
```

### 实时交易信号

```python
# 只在高置信度时交易
CONFIDENCE_THRESHOLD = 0.7

for i, (label, conf) in enumerate(zip(predicted_labels, confidence)):
    if conf >= CONFIDENCE_THRESHOLD:
        # 强信号 - 可以交易
        signal = label  # 'HH', 'HL', 'LH', 'LL'
        print(f"Bar {i}: {signal} (confidence: {conf:.2%})")
    else:
        # 弱信号 - 等待
        print(f"Bar {i}: 信号不确定 (confidence: {conf:.2%})")
```

---

## 性能基准

基于 2 年 BTC-USD 1 小时数据的预期性能：

| 指标 | 值 |
|------|----|
| 总样本数 | ~15,000+ |
| 测试集准确率 | 65-75% |
| F1 Score (加权) | 0.65-0.72 |
| 基准准确率 | 25% (4 类随机) |
| 改进 | +150-200% |

### 标记分布

```
HH (Higher High)  : ~30%  ← 上升趋势继续
HL (Higher Low)   : ~20%  ← 上升趋势开始
LH (Lower High)   : ~20%  ← 下降趋势开始
LL (Lower Low)    : ~30%  ← 下降趋势继续
```

HH 和 LL 更容易预测（趋势延续），HL 和 LH 更难（趋势反转）。

---

## 优化建议

### 1. 超参数调优

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

### 2. 特征工程

添加更多特征：
- 微观结构特征（买卖挂单差异）
- 市场情绪指标（恐惧贪婪指数）
- 跨市场相关性（期货、期权）
- 链上数据（转账数、鲸鱼活动）

### 3. 集成学习

组合多个模型：
```python
from sklearn.ensemble import StackingClassifier

base_models = [
    xgb.XGBClassifier(),
    LGBMClassifier(),
    RandomForestClassifier()
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()
)
```

### 4. 时间序列验证

避免数据泄露：
```python
# 不要用时间混乱的 train_test_split
# 用时间序列分割
split_idx = int(0.8 * len(data))
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]
```

---

## 交易策略示例

### 策略 1：纯预测信号

```python
def strategy_pure_prediction(model, features, data):
    predictions = model.predict(features)  # 0-3
    signals = {0: 'BUY', 1: 'SELL', 2: 'SELL', 3: 'BUY'}
    return [signals[p] for p in predictions]
```

### 策略 2：置信度加权

```python
def strategy_confidence_weighted(model, features, data, threshold=0.6):
    proba = model.predict_proba(features)
    predictions = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    
    signals = []
    for pred, conf in zip(predictions, confidence):
        if conf < threshold:
            signals.append('HOLD')
        elif pred in [0, 3]:  # HH, LL
            signals.append('FOLLOW_TREND')
        else:  # HL, LH
            signals.append('PREPARE_REVERSAL')
    
    return signals
```

### 策略 3：转折点检测

```python
def strategy_reversal_detection(model, features, data):
    predictions = np.argmax(model.predict_proba(features), axis=1)
    signals = []
    
    for i in range(1, len(predictions)):
        prev_label = predictions[i-1]
        curr_label = predictions[i]
        
        # 检测 LL -> HH (潜在底部) 或 HH -> LL (潜在顶部)
        if prev_label == 3 and curr_label == 0:  # LL -> HH
            signals.append('STRONG_BUY')
        elif prev_label == 0 and curr_label == 3:  # HH -> LL
            signals.append('STRONG_SELL')
        else:
            signals.append('HOLD')
    
    return signals
```

---

## 常见问题

### Q1: 为什么准确率只有 70%？
A: 4 类分类本身比 2 类困难。70% 对 25% 基准的改进是 +180%。而且市场中确实有随机性。

### Q2: 如何处理不平衡的标签？
A: 使用 `class_weight='balanced'` 或过采样/欠采样少数类。

### Q3: 模型会过拟合吗？
A: 使用早停法、验证集监控、增加正则化。如果 train/test 性能差异 >10% 可能过拟合。

### Q4: 如何改进模型？
A: 1) 增加特征 2) 调参 3) 更多数据 4) 集成学习 5) 特征选择

### Q5: 能用于其他币种/资产吗？
A: 可以！ZigZag 是通用的。重新训练不同资产的数据即可。

---

## 参考资源

1. **ZigZag 原始指标**: https://www.tradingview.com/script/sZrwXphZ-ZigZag-++/
2. **XGBoost 文档**: https://xgboost.readthedocs.io/
3. **技术分析**: https://en.wikipedia.org/wiki/Technical_analysis
4. **时间序列 ML**: https://www.coursera.org/learn/time-series-forecasting

---

## 许可证

MIT

---

## 联系方式

如有问题或建议，欢迎提交 Issue 或 PR！
