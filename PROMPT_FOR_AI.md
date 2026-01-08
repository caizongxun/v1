# 提示詞：實現 ZigZag K棒標記演算法

## 任務概述

你的任務是實現一個 ZigZag K棒標記演算法。此演算法接收OHLCV時間序列資料，輸出每根K棒的標記類別。

## 輸入規格

必須接收一個 Pandas DataFrame，包含以下列（必須有這些列名）：
- open: 開盤價（float64或float32）
- high: 最高價（float64或float32）
- low: 最低價（float64或float32）
- close: 收盤價（float64或float32）
- volume: 成交量（float64或float32）

資料品質要求：
- 沒有NaN值
- 沒有無窮大值
- 所有K棒滿足: high >= low
- 所有K棒滿足: open和close都在[low, high]範圍內

## 輸出規格

必須輸出一個長度等於輸入K棒數的陣列，其中每個元素是整數0、1、2或3。

標記含義：
- 0: HH (High to High) - 前一個高點後，出現更高的高點
- 1: HL (High to Low) - 前一個高點後，出現更低的低點
- 2: LH (Low to High) - 前一個低點後，出現更高的高點
- 3: LL (Low to Low) - 前一個低點後，出現更低的低點

## 演算法參數

固定參數（不可改變）：
- threshold: 0.02 (反轉變化幅度閾值)
- lookback_window: 20 (向前掃描的最大K棒數)

## 演算法步驟

### 步驟1：初始化
建立一個空陣列labels用於儲存標記結果。
設定索引i = 0。

### 步驟2：主迴圈

執行以下迴圈直到i >= len(close) - 1：

#### 2.1：提取當前K棒資訊
```
current_high = high[i]
current_low = low[i]
next_high = current_high
next_low = current_low
found_reversal = False
```

#### 2.2：掃描迴圈

從k = i + 1到 min(i + lookback_window, len(close) - 1)執行迴圈：

2.2.1：更新追蹤的高低點
```
if high[k] > next_high:
    next_high = high[k]
if low[k] < next_low:
    next_low = low[k]
```

2.2.2：計算變化幅度
```
if current_low > 0:
    change_up = (next_high - current_low) / current_low
else:
    change_up = 0

if current_high > 0:
    change_down = (current_high - next_low) / current_high
else:
    change_down = 0
```

2.2.3：檢查向上反轉
```
if change_up > threshold:
    found_reversal = True
    if next_low < current_low:
        labels.append(2)  # LH
    else:
        labels.append(0)  # HH
    i = k
    break (跳出掃描迴圈)
```

2.2.4：檢查向下反轉
```
if change_down > threshold:
    found_reversal = True
    if next_high > current_high:
        labels.append(0)  # HH
    else:
        labels.append(3)  # LL
    i = k
    break (跳出掃描迴圈)
```

#### 2.3：掃描結束後的處理

如果found_reversal為False（未發現反轉）：

如果 i + 2 < len(close)：

  提取三根K棒的高低點：
  ```
  h1 = high[i]
  h2 = high[i + 1]
  h3 = high[i + 2]
  l1 = low[i]
  l2 = low[i + 1]
  l3 = low[i + 2]
  ```

  檢查中間K棒是否為高點：
  ```
  if (h2 >= h1) and (h2 >= h3):
      if (l2 >= l1) and (l2 >= l3):
          labels.append(0)  # HH
      else:
          labels.append(1)  # HL
  ```

  檢查中間K棒是否為低點：
  ```
  elif (l2 <= l1) and (l2 <= l3):
      if (h2 <= h1) and (h2 <= h3):
          labels.append(3)  # LL
      else:
          labels.append(2)  # LH
  ```

  都不符合時：
  ```
  else:
      labels.append(random_integer_in_range(0, 3))
  ```

  移動到下一根K棒：
  ```
  i = i + 1
  ```

如果 i + 2 >= len(close)：

  直接移動到下一根K棒：
  ```
  i = i + 1
  ```

### 步驟3：補齊剩餘標記

如果標記數少於K棒總數（這種情況不應該發生，但作為保險）：

```
while len(labels) < len(close):
    labels.append(random_integer_in_range(0, 3))
```

### 步驟4：返回結果

返回長度為len(close)的陣列，只取前len(close)個標記。

## 重要細節

### 掃描窗口的定義

掃描窗口是指從當前K棒i開始，向前最多看20根後續K棒(i+1到i+20)。

在掃描過程中，追蹤這20根K棒中出現的最高高點和最低低點。

### 反轉判斷邏輯

向上反轉：
- 計算公式：change_up = (next_high - current_low) / current_low
- 判斷：如果change_up > 0.02，表示向上反轉
- 分類：
  - 如果next_low < current_low，標記為LH (2)
  - 否則標記為HH (0)

向下反轉：
- 計算公式：change_down = (current_high - next_low) / current_high
- 判斷：如果change_down > 0.02，表示向下反轉
- 分類：
  - 如果next_high > current_high，標記為HH (0)
  - 否則標記為LL (3)

### 邊界情況

1. 當current_low接近0時：
   - 設定change_up = 0以避免除零錯誤

2. 當current_high接近0時：
   - 設定change_down = 0以避免除零錯誤

3. 當接近序列結尾(i + 2 >= len(close))時：
   - 跳過局部高低點檢查
   - 直接移動到下一根K棒

4. 當掃描完20根K棒仍無反轉時：
   - 使用局部高低點結構判斷
   - 如果都不符合則隨機標記

## 驗證清單

實現完成後，必須驗證以下項目：

1. 輸入資料驗證
   - 檢查DataFrame是否包含所有必需列
   - 檢查列名是否為小寫(open, high, low, close, volume)
   - 驗證沒有NaN或無窮大值
   - 驗證high >= low對所有K棒成立

2. 演算法邏輯
   - 驗證threshold設定為0.02
   - 驗證lookback_window設定為20
   - 驗證掃描迴圈正確追蹤next_high和next_low
   - 驗證除以零已正確處理

3. 輸出驗證
   - 驗證輸出陣列長度等於輸入K棒數
   - 驗證所有輸出值都在[0, 1, 2, 3]範圍內
   - 驗證沒有NaN或無窮大值
   - 驗證標記與K棒按順序對應

4. 邊界情況
   - 驗證序列末尾的處理
   - 驗證掃描窗口邊界的處理
   - 驗證局部高低點檢查只在i + 2 < len(close)時執行

## 實現建議

使用Python和NumPy實現以提高效率。

建議函數簽名：
```python
def find_zigzag_labels(df):
    """
    輸入: DataFrame with columns [open, high, low, close, volume]
    輸出: array of length len(df), values in [0, 1, 2, 3]
    """
    # 實現演算法
    return labels
```

## 測試範例

提供至少3組測試資料以驗證實現：

測試1：明確的向上反轉
```
bar[i]:     high=100, low=95
bar[i+1]:   high=101, low=96
bar[i+2]:   high=105, low=100
bar[i+3]:   high=108, low=102

預期結果: bar[i]標記為0(HH)或2(LH)
```

測試2：明確的向下反轉
```
bar[i]:     high=100, low=95
bar[i+1]:   high=99, low=94
bar[i+2]:   high=98, low=90
bar[i+3]:   high=97, low=88

預期結果: bar[i]標記為1(HL)或3(LL)
```

測試3：小幅波動，無明確反轉
```
bar[i]:     high=100, low=95
bar[i+1]:   high=100.5, low=95.5
bar[i+2]:   high=100.2, low=95.3

預期結果: bar[i]標記為0(HH)
```

## 性質檢查

實現應具備以下性質：

1. 決定性：給定相同輸入，始終產生相同輸出(除了隨機填充部分)
2. 貪心：一旦發現反轉立即標記，不回溯
3. 前向：只掃描未來20根K棒，不考慮歷史
4. 跳躍式：標記後跳至反轉點，不逐根處理
5. 局部相容：在無反轉時使用局部結構判斷

## 注意事項

1. 不要修改輸入DataFrame
2. 不要改變threshold或lookback_window值
3. 確保輸出陣列不包含無效值
4. 確保輸出順序與輸入K棒順序一致
5. 隨機填充部分應使用安全的隨機數生成器
