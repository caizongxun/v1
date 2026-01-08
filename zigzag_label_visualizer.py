#!/usr/bin/env python3
"""ZigZag Label Visualizer - Standalone Tool"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def generate_zigzag_labels_tradingview(df, depth=12, deviation=0.5, backstep=2):
    """
    生成 ZigZag 標籤，基於 TradingView 官方邏輯
    
    返回:
    - labels: [0, 1, 2, 3, 4, ...] 每根K線的標籤
    - extrema: [(bar_idx, price, label_type), ...]
    """
    
    n = len(df)
    labels = np.full(n, 4, dtype=int)  # 4 = No Pattern
    extrema = []
    
    last_high_idx = 0
    last_high_price = -np.inf
    last_low_idx = 0
    last_low_price = np.inf
    
    init_window = df.iloc[0:min(depth+1, n)]
    if len(init_window) > 1:
        if init_window['close'].iloc[-1] > init_window['close'].iloc[0]:
            current_direction = 'up'
            last_low_price = init_window['low'].min()
            last_low_idx = init_window['low'].idxmin()
        else:
            current_direction = 'down'
            last_high_price = init_window['high'].max()
            last_high_idx = init_window['high'].idxmax()
    else:
        current_direction = 'up'
    
    for i in range(depth, n):
        window = df.iloc[i-depth:i+1]
        local_high = window['high'].max()
        local_high_idx = window['high'].idxmax()
        local_low = window['low'].min()
        local_low_idx = window['low'].idxmin()
        
        dev_threshold_high = last_high_price * (1 + deviation / 100.0)
        dev_threshold_low = last_low_price * (1 - deviation / 100.0)
        
        if current_direction == 'up':
            if local_high > dev_threshold_high:
                last_high_price = local_high
                last_high_idx = local_high_idx
                extrema.append({
                    'idx': local_high_idx,
                    'bar': i,
                    'price': local_high,
                    'type': 'high',
                    'label': 'HH'
                })
                labels[local_high_idx] = 0  # HH
            
            if local_low < dev_threshold_low:
                current_direction = 'down'
                last_low_price = local_low
                last_low_idx = local_low_idx
                extrema.append({
                    'idx': local_low_idx,
                    'bar': i,
                    'price': local_low,
                    'type': 'low',
                    'label': 'LH'
                })
                labels[local_low_idx] = 2  # LH
        
        elif current_direction == 'down':
            if local_low < dev_threshold_low:
                last_low_price = local_low
                last_low_idx = local_low_idx
                extrema.append({
                    'idx': local_low_idx,
                    'bar': i,
                    'price': local_low,
                    'type': 'low',
                    'label': 'LL'
                })
                labels[local_low_idx] = 3  # LL
            
            if local_high > dev_threshold_high:
                current_direction = 'up'
                last_high_price = local_high
                last_high_idx = local_high_idx
                extrema.append({
                    'idx': local_high_idx,
                    'bar': i,
                    'price': local_high,
                    'type': 'high',
                    'label': 'HL'
                })
                labels[local_high_idx] = 1  # HL
    
    return labels, extrema

def visualize_zigzag(df, labels, extrema, title="ZigZag Pattern Visualization", figsize=(20, 8)):
    """
    可視化 ZigZag 標籤
    
    顏色編碼:
    - 藍色圓圈: HH (Higher High)
    - 橙色圓圈: HL (Higher Low)
    - 綠色圓圈: LH (Lower High)
    - 紅色圓圈: LL (Lower Low)
    - 灰色線: ZigZag 連線
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 繪製蠟燭圖
    df_plot = df.reset_index(drop=True)
    
    # 高低線
    ax.plot(df_plot.index, df_plot['high'], color='lightgray', linewidth=0.5, alpha=0.5, label='High')
    ax.plot(df_plot.index, df_plot['low'], color='lightgray', linewidth=0.5, alpha=0.5, label='Low')
    
    # 收盤線
    ax.plot(df_plot.index, df_plot['close'], color='black', linewidth=1, alpha=0.7, label='Close')
    
    # 顏色映射
    label_colors = {
        0: ('blue', 'HH'),      # Higher High
        1: ('orange', 'HL'),    # Higher Low
        2: ('green', 'LH'),     # Lower High
        3: ('red', 'LL'),       # Lower Low
        4: ('gray', 'No Pattern')
    }
    
    # 繪製極值點
    for ext in extrema:
        idx = ext['idx']
        price = ext['price']
        label = ext['label']
        
        color = label_colors[{'HH': 0, 'HL': 1, 'LH': 2, 'LL': 3}[label]][0]
        
        # 繪製圓圈
        ax.scatter(idx, price, color=color, s=200, marker='o', edgecolors='black', linewidth=2, zorder=5)
        
        # 添加標籤文字
        ax.text(idx, price * 1.01, label, ha='center', fontsize=8, fontweight='bold')
    
    # 繪製 ZigZag 連線
    if len(extrema) > 1:
        sorted_extrema = sorted(extrema, key=lambda x: x['idx'])
        for i in range(len(sorted_extrema) - 1):
            x1, y1 = sorted_extrema[i]['idx'], sorted_extrema[i]['price']
            x2, y2 = sorted_extrema[i+1]['idx'], sorted_extrema[i+1]['price']
            ax.plot([x1, x2], [y1, y2], color='gray', linewidth=1.5, linestyle='--', alpha=0.6, zorder=2)
    
    # 設置圖表
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Bar Index', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 圖例
    legend_elements = [
        mpatches.Patch(color='blue', label='HH (Higher High)'),
        mpatches.Patch(color='orange', label='HL (Higher Low)'),
        mpatches.Patch(color='green', label='LH (Lower High)'),
        mpatches.Patch(color='red', label='LL (Lower Low)'),
        mpatches.Patch(color='gray', label='No Pattern')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    return fig, ax

def print_label_statistics(labels, extrema):
    """
    打印標籤統計
    """
    
    label_names = ['HH', 'HL', 'LH', 'LL', 'No Pattern']
    
    print("\n" + "="*60)
    print("LABEL STATISTICS")
    print("="*60)
    
    print(f"\nTotal bars: {len(labels):,}")
    print(f"Total extrema points: {len(extrema):,}")
    
    print(f"\nLabel Distribution:")
    for i in range(5):
        count = (labels == i).sum()
        pct = 100 * count / len(labels)
        print(f"  {i}. {label_names[i]:12s}: {count:6d} ({pct:5.1f}%)")
    
    # 極值點分佈
    print(f"\nExtrema Points by Type:")
    hh_count = sum(1 for e in extrema if e['label'] == 'HH')
    hl_count = sum(1 for e in extrema if e['label'] == 'HL')
    lh_count = sum(1 for e in extrema if e['label'] == 'LH')
    ll_count = sum(1 for e in extrema if e['label'] == 'LL')
    
    print(f"  HH: {hh_count:4d}")
    print(f"  HL: {hl_count:4d}")
    print(f"  LH: {lh_count:4d}")
    print(f"  LL: {ll_count:4d}")
    
    print(f"\nExtreme Point Details (First 20):")
    for i, ext in enumerate(extrema[:20]):
        print(f"  {i+1:2d}. Bar {ext['bar']:5d} | {ext['label']:2s} | Price: {ext['price']:10.2f}")
    
    if len(extrema) > 20:
        print(f"  ... and {len(extrema) - 20} more")

def compare_deviations(df, deviations=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0]):
    """
    用不同的 deviation 參數生成標籤，並比較
    """
    
    print("\n" + "="*60)
    print("COMPARING DIFFERENT DEVIATION PARAMETERS")
    print("="*60)
    
    results = []
    
    for dev in deviations:
        labels, extrema = generate_zigzag_labels_tradingview(df, depth=12, deviation=dev, backstep=2)
        
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
    
    print(f"\n{'Dev':>6s} {'Extrema':>10s} {'Labeled':>10s} {'%':>6s} {'HH':>4s} {'HL':>4s} {'LH':>4s} {'LL':>4s}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['deviation']:6.2f}% {r['extrema']:10d} {r['labeled']:10d} {r['pct']:5.1f}% {r['hh']:4d} {r['hl']:4d} {r['lh']:4d} {r['ll']:4d}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

print("="*60)
print("ZigZag Label Visualizer")
print("="*60)

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
comparison_results = compare_deviations(df, deviations=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0])

# 選擇 0.5% 作為主要參數
deviation_choice = 0.5
print(f"\n[3/5] Generating labels with deviation={deviation_choice}%...")
labels, extrema = generate_zigzag_labels_tradingview(df, depth=12, deviation=deviation_choice, backstep=2)
print(f"  OK Found {len(extrema)} extrema points")

print(f"\n[4/5] Printing statistics...")
print_label_statistics(labels, extrema)

print(f"\n[5/5] Creating visualization...")
fig, ax = visualize_zigzag(
    df,
    labels,
    extrema,
    title=f"ZigZag Pattern Visualization (Deviation={deviation_choice}%)",
    figsize=(20, 8)
)
plt.show()

print("\n" + "="*60)
print("Visualization complete!")
print("="*60)

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

print(f"\nSaving labels to 'zigzag_labels.csv'...")
df_output.to_csv('zigzag_labels.csv', index=False)
print(f"  OK Saved")

print(f"\nYou can now:")
print(f"  1. Adjust deviation parameter and re-run")
print(f"  2. Check 'zigzag_labels.csv' for all labels")
print(f"  3. Use these labels for model training")
