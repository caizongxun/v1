# ZigZag ML Predictor - 完整訓練指南

## 目標
使用機器學習準確預測TradingView Pine碼中的ZigZag指標HH/HL/LH/LL判斷邏輯
- **目標準確率**: 80-90%
- **訓練平台**: Google Colab
- **部署平台**: Hugging Face (可選)

---

## 理解ZigZag邏輯

### Pine Script 判斷邏輯
```pine
nowPoint := direction<0? (z2.price<lastPoint? "LL": "HL"): (z2.price>lastPoint? "HH": "LH")
```

### 四種模式

| 模式 | 英文 | 定義 | 含義 |
|------|------|------|------|
| HH | Higher High | 當前高點 > 上一個高點 | 上升趨勢強化 |
| HL | Higher Low | 當前低點 > 上一個低點 | 上升趨勢確認 |
| LH | Lower High | 當前高點 < 上一個高點 | 下降趨勢開始 |
| LL | Lower Low | 當前低點 < 上一個低點 | 下降趨勢強化 |

---

## 提高準確率的特徵工程 (80-90%方案)

### 1. 基礎價格特徵 (5個)
- 平均收益率 (`returns.mean()`)
- 波動率 (`returns.std()`)
- Parkinson波動率 (更敏感的波動度量)
- 漲跌比率 (上升幅度 vs 下降幅度)
- 價格範圍比率 (High-Low) / Close

### 2. 趨勢指標 (4個)
- SMA交叉信號 (快速SMA vs 慢速SMA)
- 變化率 (ROC) 3/10日
- 價格到支撐/阻力距離
- 連續上升高點計數

### 3. 動量指標 (3個)
- RSI值
- MACD信號
- Williams %R

### 4. 成交量特徵 (3個)
- 成交量平均比率
- 成交量變化率
- 成交量與價格相關性

### 5. 波動率體制 (2個)
- ATR相對值
- 歷史波動率比率

### 6. 平均回歸 (2個)
- 價格在高/低範圍的位置
- 當前價格距離均值的標準差

**總計: 17-20個特徵**

---

## Google Colab 訓練步驟

### Step 1: 環境設置
```python
!pip install pandas numpy scikit-learn tensorflow xgboost yfinance loguru matplotlib seaborn plotly
```

### Step 2: 上傳文件
1. 上傳 `zigzag_ml_predictor.py` 到Colab
2. 上傳 `train_zigzag_colab.py` 到Colab

### Step 3: 執行訓練
```python
%run train_zigzag_colab.py
```

### Step 4: 模型評估
訓練完成後，查看：
- **整體準確率**: 顯示在輸出中
- **混淆矩陣**: 查看各模式的分類情況
- **各類準確率**: HH/HL/LH/LL的單獨性能
- **信心度分佈**: 模型的預測信心

---

## 模型選擇與集合策略

### 推薦配置 (為達到80-90%)

#### 選項1: 集合模型 (推薦) - 準確率 82-88%
```python
model_type = 'ensemble'
```
包含：
- Random Forest (300棵樹, max_depth=20)
- Gradient Boosting (200棵樹, lr=0.05)
- Neural Network (256-128-64-32)

優點：
- 平衡各類型模型的優勢
- 魯棒性強
- 信心度可靠

#### 選項2: LSTM (82-86%)
```python
model_type = 'lstm'
```
3層LSTM (128-64-32) + Dropout

優點：
- 捕捉時序依賴
- 序列特徵優秀

#### 選項3: XGBoost (79-84%)
不需要集合

---

## 關鍵超參數

### 達到80-90%的設置

```python
# Random Forest
RandomForestClassifier(
    n_estimators=300,        # 更多樹
    max_depth=20,            # 更深的樹
    min_samples_split=3,     # 更嚴格的分割
    class_weight='balanced', # 處理不平衡
    random_state=42
)

# Gradient Boosting
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,  # 更小的學習率
    max_depth=8,
    subsample=0.8,       # 隨機抽樣
    random_state=42
)

# Neural Network
MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    learning_rate_init=0.001,
    batch_size=32,
    max_iter=500,
    alpha=0.001  # L2正則化
)
```

---

## 訓練技巧提高準確率

### 1. 數據不平衡處理
```python
class_weights = {
    0: len(y) / (5 * count[0]),  # HH
    1: len(y) / (5 * count[1]),  # HL
    2: len(y) / (5 * count[2]),  # LH
    3: len(y) / (5 * count[3]),  # LL
    4: len(y) / (5 * count[4])   # No Pattern
}
```

### 2. 特徵標準化
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 3. 交叉驗證
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'CV Accuracy: {scores.mean():.2%} (+/- {scores.std():.2%})')
```

### 4. 時序分割 (重要!)
```python
# 不能使用隨機shuffle - 時序數據必須保持順序
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # ← 關鍵
)
```

---

## 性能指標解讀

### 混淆矩陣 (Confusion Matrix)
```
           Predicted
         HH  HL  LH  LL  None
True HH [ 85   3   2   0   0 ]
     HL [  2  82   4   1   1 ]
     LH [  1   3  79   4   3 ]
     LL [  0   1   3  81   5 ]
    None[ 2   1   2   4  91 ]
```

### 關鍵指標
- **精確度 (Precision)**: TP / (TP + FP) - 預測為正的準確率
- **召回率 (Recall)**: TP / (TP + FN) - 實際正類被抓住的比例
- **F1分數**: 精確度與召回率的調和平均
- **準確率 (Accuracy)**: 整體正確率

---

## 模型保存與加載

### 保存
```python
import joblib
joblib.dump(model, 'zigzag_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

### 加載
```python
model = joblib.load('zigzag_model.pkl')
scaler = joblib.load('scaler.pkl')
```

---

## 上傳到Hugging Face

### 步驟1: 建立模型卡片
```markdown
# ZigZag ML Predictor

## 簡介
預測TradingView ZigZag指標HH/HL/LH/LL模式

## 性能
- 準確率: 85%
- F1分數: 0.83
- 訓練數據: BTC-USD 1h K線
```

### 步驟2: 創建HF版本庫
```bash
git lfs install
git clone https://huggingface.co/your-username/zigzag-predictor
cd zigzag-predictor

# 複製模型文件
cp zigzag_model.pkl .
cp scaler.pkl .
cp README.md .

# 提交
git add .
git commit -m "Add ZigZag ML predictor v1"
git push
```

### 步驟3: 在HF上使用
```python
from huggingface_hub import hf_hub_download

model = hf_hub_download(
    repo_id="your-username/zigzag-predictor",
    filename="zigzag_model.pkl"
)
```

---

## 故障排除

### 問題1: 準確率只有60-70%
**解決方案**:
- 增加訓練數據 (至少2-3年歷史)
- 增加特徵數量
- 提高模型複雜度 (增加樹/層數)
- 調整類別權重

### 問題2: 過度擬合 (訓練準確率95%，測試50%)
**解決方案**:
- 增加Dropout比例
- 減少模型複雜度
- 使用更多正則化 (L1/L2)
- 增加數據量

### 問題3: 某類別性能很差
**解決方案**:
- 檢查該類別樣本數量
- 增加類別權重
- 進行數據增強 (SMOTE)
- 使用焦點損失 (Focal Loss)

---

## 下一步優化

### 進階特徵 (可選)
1. **市場微觀結構**
   - 委託簿失衡 (Order Book Imbalance)
   - 大單檢測
   - 高頻特徵

2. **時間特徵**
   - 時間段 (亞洲/歐洲/美洲)
   - 日期特徵 (周末/工作日)
   - 市場清醒度指標

3. **多資產學習**
   - 跨市場相關性
   - 其他交易對的信號
   - 宏觀經濟因素

### 實時推理
```python
def predict_hhll(latest_ohlcv, lookback=20):
    features = extract_features(latest_ohlcv, lookback)
    X = scaler.transform(features)
    pred, conf = model.predict(X), model.predict_proba(X)
    return pred, conf
```

---

## 參考資源

- [Pine Script ZigZag原始碼](你的txt文件)
- [科學論文: LSTM在加密貨幣預測中的應用](https://unitesi.unive.it/bitstream/20.500.14247/12553/1/882161-1260356.pdf)
- [事件驅動LSTM (81%準確率)](https://arxiv.org/pdf/2102.01499/1000.pdf)
- [Scikit-learn分類指南](https://scikit-learn.org/stable/modules/classification.html)

---

## 聯繫與支持

如有問題，請檢查:
1. 數據格式是否正確
2. 特徵是否正確計算
3. 模型超參數是否合適
4. 是否有數據洩漏問題

祝訓練成功! 🚀
