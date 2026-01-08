#!/usr/bin/env python3
"""ZigZag Implementation - Reddit/StackOverflow Verified Version"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def zigzag(s, ut, dt):
    """
    StackOverflow 版本的 ZigZag
    
    s: DataFrame 包含 H (High) 和 L (Low) 列
    ut: 上升臨界值 (e.g., 1.05 表示 5% 上升)
    dt: 下降臨界值 (e.g., 0.95 表示 5% 下降)
    
    返回: DataFrame 包含 ZigZag 點的索引和價格
    """
    
    tr = None
    ld = None
    lp = None
    zzd = []
    zzp = []
    
    for ix, ch, cl in zip(s.index, s['high'], s['low']):
        # No initial trend
        if tr is None:
            if ch / lp > ut:
                tr = 1
            elif cl / lp < dt:
                tr = -1
        # Trend is up
        elif tr == 1:
            # New H
            if ch > lp:
                ld, lp = ix, ch
            # Reversal
            elif cl / lp < dt:
                zzd.append(ld)
                zzp.append(lp)
                tr, ld, lp = -1, ix, cl
        # Trend is down
        else:
            # New L
            if cl < lp:
                ld, lp = ix, cl
            # Reversal
            elif ch / lp > ut:
                zzd.append(ld)
                zzp.append(lp)
                tr, ld, lp = 1, ix, ch
        
        # Initialize if needed
        if tr is None and lp is None:
            lp = cl if cl > 0 else 1
    
    # Extrapolate the current trend
    if zzd:
        if zzd[-1] != s.index[-1]:
            zzd.append(s.index[-1])
            if tr is None:
                zzp.append(s['close'].iloc[-1])
            elif tr == 1:
                zzp.append(s['high'].iloc[-1])
            else:
                zzp.append(s['low'].iloc[-1])
    
    return pd.DataFrame({'index': zzd, 'price': zzp})

def generate_zigzag_labels(df, ut, dt):
    """
    生成 ZigZag 標籤
    
    基於 Reddit 和 StackOverflow 的經過驗證的實現
    """
    
    # 先找到 ZigZag 點
    zz = zigzag(df, ut, dt)
    
    if zz.empty or len(zz) < 2:
        return np.full(len(df), 4, dtype=int), []
    
    # 轉換為 DataFrame 的 index
    df_reset = df.reset_index(drop=True)
    
    # 找到 ZigZag 點在 reset index 中的位置
    zz_positions = []
    for orig_idx in zz['index']:
        if isinstance(orig_idx, int):
            zz_positions.append(orig_idx)
        else:
            # 如果是 datetime，需要找到對應的位置
            pos = df_reset[df_reset.index == orig_idx].index
            if len(pos) > 0:
                zz_positions.append(pos[0])
    
    labels = np.full(len(df), 4, dtype=int)  # 4 = No Pattern
    extrema = []
    
    # 根據 ZigZag 點生成標籤
    # 確定初始方向
    if len(zz_positions) < 2:
        return labels, extrema
    
    prices = zz['price'].values
    
    # 判斷方向：如果第二個點高於第一個點，則上升
    if prices[1] > prices[0]:
        direction = 1  # 上升
    else:
        direction = -1  # 下降
    
    for i in range(len(zz_positions)):
        pos = zz_positions[i]
        price = prices[i]
        
        if i == 0:
            continue  # 第一個點不標記
        
        prev_price = prices[i-1]
        
        if direction == 1:  # 上升趨勢
            if price > prev_price:
                label = 0  # HH
                label_str = 'HH'
            else:
                label = 2  # LH
                label_str = 'LH'
                direction = -1  # 轉向下降
        else:  # 下降趨勢
            if price < prev_price:
                label = 3  # LL
                label_str = 'LL'
            else:
                label = 1  # HL
                label_str = 'HL'
                direction = 1  # 轉向上升
        
        labels[pos] = label
        extrema.append({
            'idx': pos,
            'price': price,
            'label': label_str
        })
    
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

# ============================================================================
# MAIN
# ============================================================================

print("="*80)
print("ZigZag - Reddit/StackOverflow Verified Implementation")
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

print(f"  時間範圍: {df['datetime'].iloc[0]} 至 {df['datetime'].iloc[-1]}")
print(f"  價格範圍: {df['close'].min():.2f} - {df['close'].max():.2f}")

print("\n[2/4] 比較不同的 percent_change 參數...")

print(f"\n{'percent_change':>15s} {'extrema_count':>15s} {'description':>30s}")
print("-" * 80)

for pct in [1, 2, 3, 5, 7, 10]:
    ut = 1 + pct/100
    dt = 1 - pct/100
    labels, extrema = generate_zigzag_labels(df, ut, dt)
    print(f"{pct:14d}% {len(extrema):15d}")

# 選擇 5% 作為主要參數
pct_choice = 5
print(f"\n[3/4] 用 percent_change={pct_choice}% 生成標籤...")

ut = 1 + pct_choice/100
dt = 1 - pct_choice/100
labels, extrema = generate_zigzag_labels(df, ut, dt)

print(f"  OK 找到 {len(extrema)} 個極值點")

print(f"\n  標籤分布:")
hh = sum(1 for e in extrema if e['label'] == 'HH')
hl = sum(1 for e in extrema if e['label'] == 'HL')
lh = sum(1 for e in extrema if e['label'] == 'LH')
ll = sum(1 for e in extrema if e['label'] == 'LL')
print(f"    HH: {hh:4d}")
print(f"    HL: {hl:4d}")
print(f"    LH: {lh:4d}")
print(f"    LL: {ll:4d}")
print(f"    合計: {len(extrema):4d}")

print(f"\n  最近的極值點 (Last 30):")
for i, ext in enumerate(extrema[-30:]):
    print(f"    {i+1:2d}. Bar {ext['idx']:5d} | {ext['label']:2s} | Price: {ext['price']:10.2f}")

print(f"\n[4/4] 創建可視化 (只顯示最近 100 根 K 線)...")
fig, ax = visualize_zigzag(
    df,
    labels,
    extrema,
    title=f"ZigZag Pattern (percent_change={pct_choice}%) - 最近 100 根 K 線",
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
df_output.to_csv('zigzag_labels_reddit.csv', index=False)
print(f"\n標籤已保存到 'zigzag_labels_reddit.csv'")

print(f"\n下一步:")
print(f"  如果信號數量不對，調整 pct_choice 參數")
print(f"  試試: 1%, 2%, 3%, 5%, 7%, 10%")
