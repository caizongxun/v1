# ZigZag 准确标记 - Colab 指南

## 概述

本文档提供了两个高精度的 ZigZag 实现，完全遵循 TradingView Pine Script 的逻辑。

### 核心改进

**问题**: 原始实现的 HH/HL/LH/LL 标记精度低

**解决方案**: 
- 完全重写标记逻辑，遵循 Pine Script 的精确定义
- 每个 pivot 点与前一个不同方向的 pivot 比较（而不是前一个 pivot）
- 准确追踪 direction 变化

---

## Pine Script 逻辑解析

### 关键变量

```pine
[direction, z1, z2] = ZigZag.zigzag(low, high, Depth, Deviation, Backstep)
```

- **direction**: 当前 ZigZag 方向 (>0 = 上升趋势, <0 = 下降趋势)
- **z1**: 前一个 pivot 点
- **z2**: 当前 pivot 点
- **lastPoint**: 上一次方向改变时的 pivot 价格

### 标记规则

```pine
if direction < 0:  // 下降趋势
    if z2.price < lastPoint:
        label = "LL"  // Lower Low
    else:
        label = "HL"  // Higher Low
else:  // 上升趋势 (direction > 0)
    if z2.price > lastPoint:
        label = "HH"  // Higher High
    else:
        label = "LH"  // Lower High
```

### 关键洞察

**重要**: 标记时比较的是：
- **当前 pivot 价格** vs **前一个不同方向 pivot 的价格**

而不是：
- 当前 vs 前一个 pivot
- 当前 vs 前两个 pivot
- 当前 vs rolling high/low

---

## 文件说明

### 1. `zigzag_pinescript_accurate.py`

**用途**: 完整的独立 Python 脚本

**特性**:
- 遵循 Pine Script 的精确逻辑
- 包含完整的数据获取、处理、标记、可视化
- 详细的控制台输出
- 自动生成 CSV 报告
- 生成高质量的 PNG 图表

**运行方式**:

```bash
# 本地执行
python3 zigzag_pinescript_accurate.py

# 输出文件
# - zigzag_visualization.png  (高分辨率图表)
# - zigzag_pivots.csv        (所有 pivot 数据)
```

### 2. `colab_zigzag_visualize.py`

**用途**: Google Colab 优化版本

**特性**:
- 自动依赖安装
- Colab 原生支持
- 增强的错误处理
- 优化的输出格式
- Colab 友好的可视化

---

## 在 Google Colab 中执行

### 方式一: 直接运行脚本

在 Colab 新 cell 中执行：

```python
import urllib.request

print("Downloading ZigZag visualization script...")
url = "https://raw.githubusercontent.com/caizongxun/v1/main/colab_zigzag_visualize.py"
code = urllib.request.urlopen(url).read().decode('utf-8')

print("Executing...\n")
exec(code)
```

**优点**:
- 一行代码执行
- 自动处理依赖
- 完整的中文输出

### 方式二: 克隆仓库

```bash
!git clone https://github.com/caizongxun/v1.git
%cd v1
!python3 colab_zigzag_visualize.py
```

### 方式三: 仅导入核心类

```python
import urllib.request
import sys

code_url = "https://raw.githubusercontent.com/caizongxun/v1/main/colab_zigzag_visualize.py"
code_raw = urllib.request.urlopen(code_url).read().decode('utf-8')

# 提取 ZigZagPineScript 类定义
exec(code_raw.split('# MAIN PIPELINE')[0])

# 现在你可以使用 ZigZagPineScript 类
import yfinance as yf
import pandas as pd

# 自定义代码...
zz = ZigZagPineScript(depth=12, deviation=5)
```

---

## 输出解释

### 标记分布

```
Label distribution:
   HH (Higher High):    123  ( 5.12%)
   HL (Higher Low):      98  ( 4.08%)
   LH (Lower High):     156  ( 6.49%)
   LL (Lower Low):      201  ( 8.37%)
   START (Unlabeled):     2  ( 0.08%)
   TOTAL:               580
```

**说明**:
- 如果标记分布接近均匀 (各占 20-25%)，说明标记正确
- 如果某个标记占比过高 (>30%)，检查参数设置
- START 点通常只有 1-2 个

### 最近 20 个 Pivot 点

```
#   Bar    Label    Type  Price        Direction
1  12543      HH     H   65432.50        UP
2  12567      LH     H   64123.25        DOWN
3  12589      LL     L   63456.75        DOWN
4  12612      HL     L   63789.50        UP
```

**说明**:
- 按时间顺序显示最近的 20 个 pivot
- Type: H = 高点, L = 低点
- Direction: UP = 上升趋势, DOWN = 下降趋势

### 可视化图表

**颜色代码**:
- 蓝色 (HH): 更高的高点 - 上升趋势强劲
- 橙色 (HL): 更高的低点 - 下降趋势中的支撑
- 绿色 (LH): 更低的高点 - 下降趋势中的阻力
- 红色 (LL): 更低的低点 - 下降趋势强劲

**线条**:
- 连接 pivot 点的线条颜色对应下一个点的标记
- 线条宽度表示 ZigZag 的强度

---

## 参数调整

### Depth (深度)

**当前**: 12

**调整指南**:
- **增大 (15-20)**: 找到更大的趋势，信号减少
- **减小 (8-10)**: 找到更多细微波动，信号增加

### Deviation (偏差)

**当前**: 5%

**调整指南**:
- **增大 (7-10%)**: 筛选掉小的假信号
- **减小 (1-3%)**: 捕捉更多小波动

### 示例

```python
# 捕捉更多信号
zz = ZigZagPineScript(depth=10, deviation=2)

# 保守参数 - 只保留主趋势
zz = ZigZagPineScript(depth=15, deviation=8)
```

---

## 准确性验证

### 对比 TradingView

1. 在 TradingView 上打开同一个币对的同一个时间框架
2. 添加 Dev Lucem 的 ZigZag++ 指标
3. 比较标记点的位置和标签
4. 应该完全一致

### 标签正确性检查

```python
# 导出数据进行人工验证
for p in labeled_pivots[-10:]:
    print(f"Bar {p['idx']}: {p['label']} at {p['price']}")
```

---

## Colab 最佳实践

### 1. 处理大数据集

```python
# 对于超过 2 年的数据，考虑分段处理
for year in ['2023-01-01', '2024-01-01', '2025-01-01']:
    df = yf.download('BTC-USD', start=year, period='1y', interval='1h')
    # 处理
```

### 2. 保存结果

```python
# 保存到 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 复制文件
import shutil
shutil.copy('zigzag_pivots.csv', '/content/drive/My Drive/')
shutil.copy('zigzag_analysis.png', '/content/drive/My Drive/')
```

### 3. 并行处理多个币对

```python
coins = ['BTC-USD', 'ETH-USD', 'XRP-USD']

for coin in coins:
    print(f"\nProcessing {coin}...")
    df = yf.download(coin, period='2y', interval='1h', progress=False)
    # ... 处理逻辑
```

---

## 常见问题

### Q: 为什么标记和 TradingView 不完全一致？

**A**: 检查以下几点：
1. 数据来源是否相同 (yfinance vs TradingView)
2. 参数是否相同 (depth, deviation)
3. 时间框架是否相同 (1h vs 4h 等)

### Q: 如何在 Colab 中保存高分辨率图表？

**A**:
```python
plt.savefig('chart.png', dpi=300, bbox_inches='tight')
```

### Q: 如何导出为 CSV 格式便于后续分析？

**A**:
```python
pivot_df = pd.DataFrame(labeled_pivots)
pivot_df.to_csv('analysis.csv', index=False)
```

---

## 技术架构

### 类: ZigZagPineScript

#### 方法: find_pivots(highs, lows)

功能: 识别 pivot 点

返回: [(index, price, type), ...]

#### 方法: label_pivots_accurate(pivots)

功能: 为 pivot 标记 HH/HL/LH/LL

返回: [{'idx', 'price', 'type', 'label', 'direction'}, ...]

### 函数: visualize_zigzag_accurate()

功能: 生成出版级别的图表

返回: (fig, ax) matplotlib 对象

---

## 下一步

### 1. 训练 ML 模型

使用这些准确的标记来训练机器学习模型预测 HH/HL/LH/LL

### 2. 实时信号生成

将此算法集成到实时交易系统中

### 3. 多时间框架分析

同时分析 15m, 1h, 4h 时间框架的 ZigZag 模式

### 4. 特征工程

基于 ZigZag 标记生成高级交易特征

---

## 参考资源

- Pine Script 原始代码: `Xin-Zeng-Wen-Zi-Wen-Jian-Fu-Zhi-4.txt`
- 开发者: Dev Lucem (TradingView)
- 算法基础: MT4 ZigZag 指标

---

## 许可证

MIT License

## 最后更新

2026-01-08
