# ZigZag K棒標記規範

## 1. 標記定義

共有 4 種標記，編碼如下：

- 0: HH (High to High) - 前一個高點後，出現更高的高點
- 1: HL (High to Low) - 前一個高點後，出現更低的低點
- 2: LH (Low to High) - 前一個低點後，出現更高的高點
- 3: LL (Low to Low) - 前一個低點後，出現更低的低點

## 2. 輸入資料

### 資料格式

DataFrame 必須包含以下列（按此順序）：
- `open`: 開盤價
- `high`: 最高價
- `low`: 最低價
- `close`: 收盤價
- `volume`: 成交量

### 資料類型

所有價格列（open, high, low, close）必須是 float64 或 float32。

### 資料品質

- 不存在 NaN 值
- 不存在無窮大值
- 所有 high >= low
- 所有 open 和 close 在 [low, high] 範圍內

## 3. 標記演算法

### 演算法參數

```
threshold = 0.02  (2% 變化幅度)
lookback_window = 20  (向前掃描的最大 K 棒數)
```

### 演算法步驟

#### 步驟 1: 初始化

```
輸入: K 棒時間序列 DataFrame
輸出: 長度相同的標記陣列，每個元素是 0, 1, 2, 3

labels = []
i = 0
```

#### 步驟 2: 逐根 K 棒處理

```
while i < len(close) - 1:
    current_high = high[i]
    current_low = low[i]
    next_high = current_high
    next_low = current_low
    found_reversal = False
    
    # 進入掃描迴圈
    for k in range(i + 1, min(i + lookback_window, len(close))):
        # 更新追蹤的高低點
        if high[k] > next_high:
            next_high = high[k]
        if low[k] < next_low:
            next_low = low[k]
        
        # 計算變化幅度
        if current_low > 0:
            change_up = (next_high - current_low) / current_low
        else:
            change_up = 0
        
        if current_high > 0:
            change_down = (current_high - next_low) / current_high
        else:
            change_down = 0
        
        # 檢查向上反轉
        if change_up > threshold:
            found_reversal = True
            if next_low < current_low:
                labels.append(2)  # LH
            else:
                labels.append(0)  # HH
            i = k
            break
        
        # 檢查向下反轉
        if change_down > threshold:
            found_reversal = True
            if next_high > current_high:
                labels.append(0)  # HH
            else:
                labels.append(3)  # LL
            i = k
            break
    
    # 未找到反轉時的處理
    if not found_reversal:
        if i + 2 < len(close):
            h1 = high[i]
            h2 = high[i + 1]
            h3 = high[i + 2]
            l1 = low[i]
            l2 = low[i + 1]
            l3 = low[i + 2]
            
            # 檢查中間 K 棒是否為高點
            if (h2 >= h1) and (h2 >= h3):
                if (l2 >= l1) and (l2 >= l3):
                    labels.append(0)  # HH
                else:
                    labels.append(1)  # HL
            # 檢查中間 K 棒是否為低點
            elif (l2 <= l1) and (l2 <= l3):
                if (h2 <= h1) and (h2 <= h3):
                    labels.append(3)  # LL
                else:
                    labels.append(2)  # LH
            # 都不是則隨機
            else:
                labels.append(random_int(0, 3))
            i += 1
        else:
            i += 1
```

#### 步驟 3: 補齊剩餘標記

```
while len(labels) < len(close):
    labels.append(random_int(0, 3))

return labels[:len(close)]
```

## 4. 邏輯細節

### 反轉判斷

向上反轉的判斷條件：

```
計算自 current_low 到 next_high 的漲幅
change_up = (next_high - current_low) / current_low

如果 change_up > 0.02 (2%)：
  這被認為是向上反轉
  
  進一步判斷類型：
    如果 next_low < current_low:
      標記為 LH (低點反轉向高)
    否則：
      標記為 HH (高點持續)
```

向下反轉的判斷條件：

```
計算自 current_high 到 next_low 的跌幅
change_down = (current_high - next_low) / current_high

如果 change_down > 0.02 (2%)：
  這被認為是向下反轉
  
  進一步判斷類型：
    如果 next_high > current_high:
      標記為 HH (高點反轉向高)
    否則：
      標記為 LL (低點持續)
```

### 掃描窗口

掃描窗口的含義：

```
從當前 K 棒 i 開始
向前掃描最多 20 根後續 K 棒 (i+1 到 i+20)

在這 20 根 K 棒內，追蹤：
- 最高的高點 (next_high)
- 最低的低點 (next_low)

只要發現高於或低於閾值的變化，立即：
1. 記錄標記
2. 跳到該 K 棒位置
3. 開始下一輪掃描

如果掃描完 20 根仍無反轉，則：
1. 檢查當前 K 棒及其前後鄰近 K 棒的相對高低點
2. 根據局部結構判斷標記
3. 移動到下一根 K 棒
```

### 邊界情況

1. 當 current_low 或 current_high 為 0 或接近 0：
   - 設置 change_up 或 change_down 為 0
   - 避免除以零錯誤

2. 當接近序列結尾（i + 2 >= len(close)）：
   - 跳過局部高低點檢查
   - 直接移動到下一根 K 棒

3. 當標記數少於 K 棒數：
   - 用隨機 0-3 值填補
   - 確保輸出長度等於輸入長度

## 5. 範例

### 範例 1: 清晰的向上反轉

```
K棒序列：
bar[i]:     high=100, low=95
bar[i+1]:   high=101, low=96
bar[i+2]:   high=105, low=100
bar[i+3]:   high=108, low=102

處理 bar[i]：
current_high = 100
current_low = 95

掃描 bar[i+1]：
next_high = 101, next_low = 96
change_up = (101 - 95) / 95 = 0.0632 (6.32%)
 change_up > 0.02，發現向上反轉
next_low (96) >= current_low (95)
標記為 0 (HH)

跳到 i = i+1，继续处理 bar[i+1]
```

### 範例 2: 明確的向下反轉

```
K棒序列：
bar[i]:     high=100, low=95
bar[i+1]:   high=99, low=94
bar[i+2]:   high=98, low=90
bar[i+3]:   high=97, low=88

處理 bar[i]：
current_high = 100
current_low = 95

掃描 bar[i+2]：
next_high = 99, next_low = 90
change_down = (100 - 90) / 100 = 0.10 (10%)
 change_down > 0.02，發現向下反轉
next_high (99) < current_high (100)
標記為 3 (LL)

跳到 i = i+2，継续处理 bar[i+2]
```

### 範例 3: 無反轉，使用局部結構

```
K棒序列：
bar[i]:     high=100, low=95
bar[i+1]:   high=100.5, low=95.5
bar[i+2]:   high=100.2, low=95.3

處理 bar[i]：
掃描 20 根 K 棒，無反轉

檢查局部結構：
h1 = 100, h2 = 100.5, h3 = 100.2
l1 = 95, l2 = 95.5, l3 = 95.3

check: h2 >= h1 and h2 >= h3 → 100.5 >= 100 and 100.5 >= 100.2 → true
check: l2 >= l1 and l2 >= l3 → 95.5 >= 95 and 95.5 >= 95.3 → true

標記為 0 (HH)

移動到 i = i+1
```

## 6. 實現檢查清單

實現此演算法時，必須確保：

- [ ] 輸入資料已驗證（無 NaN、無無窮大）
- [ ] high >= low（所有 K 棒）
- [ ] threshold 設定為 0.02
- [ ] lookback_window 設定為 20
- [ ] 迴圈正確跟蹤 next_high 和 next_low
- [ ] 除以零已處理（current_low > 0, current_high > 0）
- [ ] 邊界情況已處理（i + 2 >= len）
- [ ] 輸出標記陣列長度等於輸入 K 棒數
- [ ] 所有標記值在 [0, 1, 2, 3] 範圍內
- [ ] 標記按順序對應 K 棒（labels[i] 對應 close[i]）

## 7. 性質

此演算法的性質：

1. 決定性：給定相同輸入，始終產生相同輸出（除了隨機填補部分）
2. 貪心：一旦發現反轉立即標記，不回溯
3. 前向掃描：只看未來 20 根 K 棒，不考慮歷史
4. 跳躍式處理：標記後直接跳到反轉點，不連續處理每根 K 棒
5. 局部兼容：在無反轉時，用局部高低點結構判斷

## 8. 驗證方法

驗證實現是否正確：

1. 準備 10-20 根簡單 K 棒序列（人工設計）
2. 手工計算預期標記
3. 運行實現，比對輸出
4. 檢查邊界情況（序列末尾、掃描窗口邊界）
5. 確認輸出長度 == 輸入長度
6. 確認所有標記值有效
