#!/usr/bin/env python3
"""ZigZag Label Visualizer v2 - Corrected Logic"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def generate_zigzag_labels_tradingview_v2(df, depth=12, deviation=5, backstep=2):
    """
    改進的 ZigZag 邏輯 - 只標記趨勢轉折點
    
    真正的 TradingView ZigZag 邏輯:
    1. 找到局部極值點
    2. 如果偏離前一個極值點超過 deviation %，才記錄為新的轉折點
    3. 在上升趨勢中，只記錄「高點突破」為 HH
    4. 在上升趨勢中，低點突破時才轉向，記錄為 LH
    """
    
    n = len(df)
    labels = np.full(n, 4, dtype=int)  # 4 = No Pattern
    extrema = []
    
    # 找所有局部高低點
    local_highs = []
    local_lows = []
    
    for i in range(depth, n - depth):
        window = df.iloc[i-depth:i+depth+1]
        
        # 檢查是否是局部高點
        if df['high'].iloc[i] == window['high'].max():
            local_highs.append((i, df['high'].iloc[i]))
        
        # 檢查是否是局部低點
        if df['low'].iloc[i] == window['low'].min():
            local_lows.append((i, df['low'].iloc[i]))
    
    # 合併高低點並排序
    all_extrema = sorted(local_highs + local_lows, key=lambda x: x[0])
    
    if len(all_extrema) < 2:
        return labels, []
    
    # 決定起始方向
    if all_extrema[1][1] > all_extrema[0][1]:  # 如果第二個點比第一個高
        is_high = False  # 第一個是低點
    else:
        is_high = True   # 第一個是高點
    
    # 只保留有意義的轉折點
    filtered_extrema = [all_extrema[0]]
    
    for i in range(1, len(all_extrema)):
        curr_idx, curr_price = all_extrema[i]
        last_idx, last_price = filtered_extrema[-1]
        
        # 計算百分比變化
        pct_change = abs(curr_price - last_price) / last_price * 100
        
        if pct_change >= deviation:
            filtered_extrema.append(all_extrema[i])
    
    # 現在生成標籤
    direction = 'up' if is_high else 'down'  # 根據第一個點的類型
    
    for i in range(len(filtered_extrema)):
        curr_idx, curr_price = filtered_extrema[i]
        
        if i == 0:
            # 第一個點不標記
            continue
        
        prev_idx, prev_price = filtered_extrema[i - 1]
        
        if direction == 'up':
            # 上升趨勢中
            if curr_price > prev_price:
                # 新高點
                label = 0  # HH
                extrema.append({
                    'idx': curr_idx,
                    'price': curr_price,
                    'label': 'HH'
                })
                labels[curr_idx] = 0
            else:
                # 低點突破 → 轉向下降
                label = 2  # LH
                extrema.append({
                    'idx': curr_idx,
                    'price': curr_price,
                    'label': 'LH'
                })
                labels[curr_idx] = 2
                direction = 'down'
        
        else:  # direction == 'down'
            # 下降趨勢中
            if curr_price < prev_price:
                # 新低點
                label = 3  # LL
                extrema.append({
                    'idx': curr_idx,
                    'price': curr_price,
                    'label': 'LL'
                })
                labels[curr_idx] = 3
            else:
                # 高點突破 → 轉向上升
                label = 1  # HL
                extrema.append({
                    'idx': curr_idx,
                    'price': curr_price,
                    'label': 'HL'
                })
                labels[curr_idx] = 1
                direction = 'up'
    
    return labels, extrema

def visualize_zigzag(df, labels, extrema, title="ZigZag Pattern Visualization", figsize=(20, 8)):
    """
    可視化 ZigZag 標籤
    
    顏色編碼:
    - 藍色: HH (Higher High) - 上升趨勢中的新高
    - 橙色: HL (Higher Low) - 上升轉向時的低點
    - 綠色: LH (Lower High) - 下降趨勢開始時的高點
    - 紅色: LL (Lower Low) - 下降趨勢中的新低
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    df_plot = df.reset_index(drop=True)
    
    # 畫K線
    ax.plot(df_plot.index, df_plot['high'], color='lightgray', linewidth=0.5, alpha=0.3)
    ax.plot(df_plot.index, df_plot['low'], color='lightgray', linewidth=0.5, alpha=0.3)
    ax.plot(df_plot.index, df_plot['close'], color='black', linewidth=1.5, alpha=0.8)
    
    # 顏色映射
    label_colors = {
        'HH': '#1f77b4',  # 藍色
        'HL': '#ff7f0e',  # 橙色
        'LH': '#2ca02c',  # 綠色
        'LL': '#d62728',  # 紅色
    }
    
    # 繪製極值點
    for ext in extrema:
        idx = ext['idx']
        price = ext['price']
        label = ext['label']
        color = label_colors[label]
        
        # 繪製圓圈和方框
        ax.scatter(idx, price, color=color, s=300, marker='o', edgecolors='black', linewidth=2.5, zorder=5)
        ax.scatter(idx, price, color=color, s=800, marker='s', alpha=0.3, edgecolors='none', zorder=4)
        
        # 添加標籤文字
        ax.text(idx, price * 1.015, label, ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    # 繪製ZigZag連線
    if len(extrema) > 1:
        sorted_extrema = sorted(extrema, key=lambda x: x['idx'])
        for i in range(len(sorted_extrema) - 1):
            x1, y1 = sorted_extrema[i]['idx'], sorted_extrema[i]['price']
            x2, y2 = sorted_extrema[i+1]['idx'], sorted_extrema[i+1]['price']
            
            # 根據類型選擇線的顏色
            next_label = sorted_extrema[i+1]['label']
            line_color = label_colors[next_label]
            
            ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=2, alpha=0.6, zorder=3)
    
    # 設置圖表
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Bar Index', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 圖例
    legend_elements = [
        mpatches.Patch(color='#1f77b4', label='HH - Higher High (上升趨勢新高)'),
        mpatches.Patch(color='#ff7f0e', label='HL - Higher Low (上升轉向低點)'),
        mpatches.Patch(color='#2ca02c', label='LH - Lower High (下降轉向高點)'),
        mpatches.Patch(color='#d62728', label='LL - Lower Low (下降趨勢新低)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    return fig, ax

def print_label_statistics(labels, extrema):
    """
    打印標籤統計
    """
    
    label_names = ['HH', 'HL', 'LH', 'LL', 'No Pattern']
    
    print("\n" + "="*70)
    print("LABEL STATISTICS")
    print("="*70)
    
    print(f"\nTotal bars: {len(labels):,}")
    print(f"Total extrema points: {len(extrema):,}")
    
    print(f"\nLabel Distribution:")
    for i in range(5):
        count = (labels == i).sum()
        pct = 100 * count / len(labels)
        print(f"  {i}. {label_names[i]:12s}: {count:6d} ({pct:5.2f}%)")
    
    # 極值點分布
    print(f"\nExtrema Points by Type:")
    hh_count = sum(1 for e in extrema if e['label'] == 'HH')
    hl_count = sum(1 for e in extrema if e['label'] == 'HL')
    lh_count = sum(1 for e in extrema if e['label'] == 'LH')
    ll_count = sum(1 for e in extrema if e['label'] == 'LL')
    
    print(f"  HH (Higher High):     {hh_count:4d}")
    print(f"  HL (Higher Low):      {hl_count:4d}")
    print(f"  LH (Lower High):      {lh_count:4d}")
    print(f"  LL (Lower Low):       {ll_count:4d}")
    print(f"  ───────────────────")
    print(f"  Total Extrema:        {len(extrema):4d}")
    
    print(f"\nExtreme Point Details:")
    for i, ext in enumerate(extrema[:30]):
        print(f"  {i+1:2d}. Bar {ext['idx']:5d} | {ext['label']:2s} | Price: {ext['price']:10.2f}")
    
    if len(extrema) > 30:
        print(f"  ... and {len(extrema) - 30} more")

def compare_deviations(df, deviations=[1, 2, 3, 5, 7, 10]):
    """
    比較不同的 deviation 參數
    """
    
    print("\n" + "="*70)
    print("COMPARING DIFFERENT DEVIATION PARAMETERS")
    print("="*70)
    
    results = []
    
    for dev in deviations:
        labels, extrema = generate_zigzag_labels_tradingview_v2(df, depth=12, deviation=dev, backstep=2)
        
        hh = sum(1 for e in extrema if e['label'] == 'HH')
        hl = sum(1 for e in extrema if e['label'] == 'HL')
        lh = sum(1 for e in extrema if e['label'] == 'LH')
        ll = sum(1 for e in extrema if e['label'] == 'LL')
        total_extrema = len(extrema)
        total_labeled = (labels != 4).sum()
        
        results.append({
            'deviation': dev,
            'extrema': total_extrema,
            'labeled': total_labeled,
            'hh': hh,
            'hl': hl,
            'lh': lh,
            'll': ll,
            'pct': 100 * total_labeled / len(labels)
        })
    
    print(f"\n{'Dev':>6s} {'Extrema':>10s} {'Labeled':>10s} {'%':>8s} {'HH':>4s} {'HL':>4s} {'LH':>4s} {'LL':>4s}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['deviation']:5d}% {r['extrema']:10d} {r['labeled']:10d} {r['pct']:7.2f}% {r['hh']:4d} {r['hl']:4d} {r['lh']:4d} {r['ll']:4d}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

print("="*70)
print("ZigZag Label Visualizer v2 - Corrected Logic")
print("="*70)

print("\n[1/5] Fetching market data...")
try:
    df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"  OK Loaded {len(df):,} candles")
except Exception as e:
    print(f"  Error: {e}")
    print(f"  Using synthetic data...")
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

print(f"  Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")

print("\n[2/5] Comparing different deviation parameters...")
comparison_results = compare_deviations(df, deviations=[1, 2, 3, 5, 7, 10])

# 選擇 5% 作為主要參數
deviation_choice = 5
print(f"\n[3/5] Generating labels with deviation={deviation_choice}%...")
labels, extrema = generate_zigzag_labels_tradingview_v2(df, depth=12, deviation=deviation_choice, backstep=2)
print(f"  OK Found {len(extrema)} extrema points")

print(f"\n[4/5] Printing statistics...")
print_label_statistics(labels, extrema)

print(f"\n[5/5] Creating visualization...")
fig, ax = visualize_zigzag(
    df,
    labels,
    extrema,
    title=f"ZigZag Pattern Visualization (Deviation={deviation_choice}%) - TradingView Logic",
    figsize=(22, 10)
)
plt.show()

print("\n" + "="*70)
print("Visualization complete!")
print("="*70)

# 保存標籤到 CSV
df_output = df.copy()
df_output['zigzag_label'] = labels
df_output['label_name'] = df_output['zigzag_label'].map({
    0: 'HH',
    1: 'HL',
    2: 'LH',
    3: 'LL',
    4: 'No Pattern'
})

print(f"\nSaving labels to 'zigzag_labels_v2.csv'...")
df_output.to_csv('zigzag_labels_v2.csv', index=False)
print(f"  OK Saved")

print(f"\n下一步:")
print(f"  1. 如果信號數量不對，調整 deviation_choice (試試 2, 3, 5, 7)")
print(f"  2. 確認標記的位置符合趨勢轉折")
print(f"  3. 使用 'zigzag_labels_v2.csv' 進行模型訓練")
