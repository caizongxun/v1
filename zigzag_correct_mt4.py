#!/usr/bin/env python3
"""Correct MT4-style ZigZag Implementation (Like TradingView)"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def generate_zigzag_mt4(df, depth=12, deviation=5, backstep=2):
    """
    MT4 風格的 ZigZag - 基於外羅 lastPoint 的比較
    
    关锫:
    - depth: 局部極值窗口 (12)
    - deviation: 百分比 (5%)
    - backstep: 後需越過的根數 (2)
    
    核心針: 比較是跟前一個極值點 (lastPoint) 比較，不是找新的相對高低
    """
    
    n = len(df)
    labels = np.full(n, 4, dtype=int)  # 4 = No Pattern
    extrema = []
    
    # Step 1: 找所有局部極值 (high/low)
    # 標記: 是方向 (+1 = high, -1 = low) 和價格
    zz = []  # (index, price, direction)
    
    # 找所有局部最高點
    for i in range(depth, n - depth):
        window = df.iloc[i-depth:i+depth+1]
        if df['high'].iloc[i] == window['high'].max():
            zz.append((i, df['high'].iloc[i], 1))  # 1 = high
    
    # 找所有局部最低點
    for i in range(depth, n - depth):
        window = df.iloc[i-depth:i+depth+1]
        if df['low'].iloc[i] == window['low'].min():
            zz.append((i, df['low'].iloc[i], -1))  # -1 = low
    
    # 按 index 排序
    zz.sort(key=lambda x: x[0])
    
    if len(zz) < 2:
        return labels, []
    
    # Step 2: 過濾 - 仇親 backstep u75db項
    # 不讓相鄤的 high/low 佊立出現
    filtered = [zz[0]]
    
    for i in range(1, len(zz)):
        curr_idx, curr_price, curr_dir = zz[i]
        
        # 移除之前的 backstep 個點中較弱的
        while len(filtered) >= 2:
            prev_idx, prev_price, prev_dir = filtered[-1]
            prev_prev_idx, prev_prev_price, prev_prev_dir = filtered[-2]
            
            # 如果最後三個點中間點比两邊這較弱、則移除
            if prev_dir == prev_prev_dir:
                if prev_dir == 1:  # 都是 high
                    if prev_price < prev_prev_price:
                        filtered.pop()
                    else:
                        break
                else:  # 都是 low
                    if prev_price > prev_prev_price:
                        filtered.pop()
                    else:
                        break
            else:
                break
        
        # 檢查是否需要新增這個點
        last_idx, last_price, last_dir = filtered[-1]
        
        # 計算偏離率
        pct_change = abs(curr_price - last_price) / last_price * 100
        
        # 如果方向相同 且偏離不足，筆 (e.g. 兩個都是 high 但偏離 < 5%)
        if curr_dir == last_dir:
            if pct_change >= deviation:
                # 相同方向下偏離足夠，拿較擅的這個
                if last_dir == 1 and curr_price > last_price:  # high: 取較高
                    filtered.pop()
                    filtered.append((curr_idx, curr_price, curr_dir))
                elif last_dir == -1 and curr_price < last_price:  # low: 取較低
                    filtered.pop()
                    filtered.append((curr_idx, curr_price, curr_dir))
                else:
                    filtered.append((curr_idx, curr_price, curr_dir))
        else:
            # 方向不同，直接新增
            filtered.append((curr_idx, curr_price, curr_dir))
    
    # Step 3: 根據趨勢生成標籤
    # 根據足前一個極值點來比較
    if len(filtered) < 2:
        return labels, []
    
    # 確定方向
    if filtered[1][2] == 1:  # second point is high -> going up
        direction = 1
    else:
        direction = -1
    
    for i in range(len(filtered)):
        curr_idx, curr_price, curr_dir = filtered[i]
        
        if i == 0:
            continue  # 第一個點不標記
        
        prev_idx, prev_price, prev_dir = filtered[i-1]
        
        # 根據方向和價格比較判斷標籤
        if direction == 1:  # 上升趨勢
            if curr_price > prev_price:
                label = 0  # HH - Higher High
            else:
                label = 2  # LH - Lower High (下降轉折)
                direction = -1  # 轉向下降
        else:  # 下降趨勢
            if curr_price < prev_price:
                label = 3  # LL - Lower Low
            else:
                label = 1  # HL - Higher Low (上升轉折)
                direction = 1  # 轉向上升
        
        extrema.append({
            'idx': curr_idx,
            'price': curr_price,
            'label': ['HH', 'HL', 'LH', 'LL'][label],
            'type': 'high' if curr_dir == 1 else 'low'
        })
        labels[curr_idx] = label
    
    return labels, extrema

def visualize_zigzag(df, labels, extrema, title="ZigZag Pattern", num_bars=100):
    """
    可視化 ZigZag 標籤
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    
    df_plot = df.reset_index(drop=True)
    
    # 只顯示最近 num_bars
    start_idx = max(0, len(df_plot) - num_bars)
    end_idx = len(df_plot)
    df_segment = df_plot.iloc[start_idx:end_idx].copy()
    df_segment = df_segment.reset_index(drop=True)
    
    # 調整 extrema 索引
    extrema_segment = []
    for ext in extrema:
        if start_idx <= ext['idx'] < end_idx:
            extrema_segment.append({
                'idx': ext['idx'] - start_idx,
                'price': ext['price'],
                'label': ext['label']
            })
    
    # 繪製 K 線
    ax.plot(df_segment.index, df_segment['high'], color='lightgray', linewidth=0.5, alpha=0.3)
    ax.plot(df_segment.index, df_segment['low'], color='lightgray', linewidth=0.5, alpha=0.3)
    ax.plot(df_segment.index, df_segment['close'], color='black', linewidth=1.5, alpha=0.8)
    
    # 顏色映射
    label_colors = {
        'HH': '#1f77b4',  # 藍色
        'HL': '#ff7f0e',  # 橙色
        'LH': '#2ca02c',  # 綠色
        'LL': '#d62728',  # 紅色
    }
    
    # 繪製極值點
    for ext in extrema_segment:
        idx = ext['idx']
        price = ext['price']
        label = ext['label']
        color = label_colors[label]
        
        ax.scatter(idx, price, color=color, s=300, marker='o', edgecolors='black', linewidth=2.5, zorder=5)
        ax.scatter(idx, price, color=color, s=800, marker='s', alpha=0.3, edgecolors='none', zorder=4)
        ax.text(idx, price * 1.015, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 繪製 ZigZag 連線
    if len(extrema_segment) > 1:
        sorted_extrema = sorted(extrema_segment, key=lambda x: x['idx'])
        for i in range(len(sorted_extrema) - 1):
            x1, y1 = sorted_extrema[i]['idx'], sorted_extrema[i]['price']
            x2, y2 = sorted_extrema[i+1]['idx'], sorted_extrema[i+1]['price']
            next_label = sorted_extrema[i+1]['label']
            line_color = label_colors[next_label]
            ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=2, alpha=0.6, zorder=3)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Bar Index', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    legend_elements = [
        mpatches.Patch(color='#1f77b4', label='HH - 更高的高'),
        mpatches.Patch(color='#ff7f0e', label='HL - 更高的低'),
        mpatches.Patch(color='#2ca02c', label='LH - 更低的高'),
        mpatches.Patch(color='#d62728', label='LL - 更低的低'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    return fig, ax

def compare_deviations(df, deviations=[1, 2, 3, 5, 7, 10]):
    print("\n" + "="*80)
    print("比較不同的 deviation 參數")
    print("="*80)
    
    results = []
    print(f"\n{'Deviation':>10s} {'Extrema':>10s} {'HH':>5s} {'HL':>5s} {'LH':>5s} {'LL':>5s} {'% of bars':>10s}")
    print("-" * 80)
    
    for dev in deviations:
        labels, extrema = generate_zigzag_mt4(df, depth=12, deviation=dev, backstep=2)
        
        hh = sum(1 for e in extrema if e['label'] == 'HH')
        hl = sum(1 for e in extrema if e['label'] == 'HL')
        lh = sum(1 for e in extrema if e['label'] == 'LH')
        ll = sum(1 for e in extrema if e['label'] == 'LL')
        total = len(extrema)
        pct = 100 * total / len(labels)
        
        results.append({
            'dev': dev,
            'total': total,
            'hh': hh,
            'hl': hl,
            'lh': lh,
            'll': ll
        })
        
        print(f"{dev:10.1f}% {total:10d} {hh:5d} {hl:5d} {lh:5d} {ll:5d} {pct:9.2f}%")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

print("="*80)
print("MT4 風格 ZigZag - 正確的前一個極值點比較邏輯")
print("="*80)

print("\n[1/4] 獲取數據...")
try:
    df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"  OK 加載 {len(df):,} 根 K 線")
except Exception as e:
    print(f"  錯誤: {e}")

print(f"  時間: {df['datetime'].iloc[0]} 至 {df['datetime'].iloc[-1]}")
print(f"  價格: {df['close'].min():.2f} - {df['close'].max():.2f}")

print("\n[2/4] 比較不同的 deviation 參數...")
comparison = compare_deviations(df, deviations=[1, 2, 3, 5, 7, 10])

# 選擇 5% deviation (粗似上來的參數)
deviation_choice = 5
print(f"\n[3/4] 用 deviation={deviation_choice}% 生成標籤...")
labels, extrema = generate_zigzag_mt4(df, depth=12, deviation=deviation_choice, backstep=2)
print(f"  OK 找到 {len(extrema)} 個極值點")

print(f"\n  標籤分佈:")
hh = sum(1 for e in extrema if e['label'] == 'HH')
hl = sum(1 for e in extrema if e['label'] == 'HL')
lh = sum(1 for e in extrema if e['label'] == 'LH')
ll = sum(1 for e in extrema if e['label'] == 'LL')
print(f"    HH: {hh:4d}")
print(f"    HL: {hl:4d}")
print(f"    LH: {lh:4d}")
print(f"    LL: {ll:4d}")
print(f"    合計: {len(extrema):4d}")

print(f"\n  最近的極值點 (Last 40):")
for i, ext in enumerate(extrema[-40:]):
    print(f"    {i+1:2d}. Bar {ext['idx']:5d} | {ext['label']:2s} | Price: {ext['price']:10.2f}")

print(f"\n[4/4] 創建可視化 (只顯示最近 100 根)...")
fig, ax = visualize_zigzag(
    df,
    labels,
    extrema,
    title=f"MT4 ZigZag (deviation={deviation_choice}%) - 最近 100 根 K 線",
    num_bars=100
)
plt.show()

print("\n" + "="*80)
print("完成!")
print("="*80)

# 保存
df_output = df.copy()
df_output['zigzag_label'] = labels
df_output['label_name'] = df_output['zigzag_label'].map({
    0: 'HH', 1: 'HL', 2: 'LH', 3: 'LL', 4: 'No Pattern'
})
df_output.to_csv('zigzag_labels_mt4.csv', index=False)
print(f"\n標籤已保存到 'zigzag_labels_mt4.csv'")
