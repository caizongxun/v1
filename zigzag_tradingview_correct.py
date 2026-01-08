#!/usr/bin/env python3
"""Correct TradingView ZigZag Implementation"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def generate_zigzag_correct(df, depth=12, pct=5):
    """
    正確的 TradingView ZigZag 邏輯
    
    depth: 局部極值點的窗口寬度 (12 根 K 線)
    pct: 極值點必須超過前一個的百分比 (不是絕對 5%，而是相對偏離)
    
    核心邏輯:
    1. 找出所有局部高/低點 (在 depth 窗口內的最高/最低)
    2. 只保留偏離前一個極值點 >= pct% 的點
    3. 根據趨勢方向標記為 HH/HL/LH/LL
    """
    
    n = len(df)
    labels = np.full(n, 4, dtype=int)  # 4 = No Pattern
    extrema = []
    
    # Step 1: 找所有局部極值點
    local_maxima_idx = []
    local_minima_idx = []
    
    for i in range(depth, n - depth):
        # 檢查是否是局部最高點
        window_high = df['high'].iloc[i-depth:i+depth+1].max()
        if df['high'].iloc[i] == window_high:
            local_maxima_idx.append(i)
        
        # 檢查是否是局部最低點
        window_low = df['low'].iloc[i-depth:i+depth+1].min()
        if df['low'].iloc[i] == window_low:
            local_minima_idx.append(i)
    
    # Step 2: 合併並排序
    all_extrema = []
    for idx in local_maxima_idx:
        all_extrema.append((idx, df['high'].iloc[idx], 'max'))
    for idx in local_minima_idx:
        all_extrema.append((idx, df['low'].iloc[idx], 'min'))
    
    all_extrema.sort(key=lambda x: x[0])
    
    if len(all_extrema) < 2:
        return labels, []
    
    # Step 3: 過濾 - 只保留偏離前一個 >= pct% 的點
    filtered = [all_extrema[0]]
    
    for i in range(1, len(all_extrema)):
        curr_idx, curr_price, curr_type = all_extrema[i]
        last_idx, last_price, last_type = filtered[-1]
        
        # 計算百分比變化
        pct_change = abs(curr_price - last_price) / last_price * 100
        
        if pct_change >= pct:
            filtered.append(all_extrema[i])
    
    # Step 4: 根據趨勢生成標籤
    # 確定起始趨勢方向
    if len(filtered) > 1:
        if filtered[1][1] > filtered[0][1]:
            current_direction = 'up'   # 第二點更高，所以上升趨勢
        else:
            current_direction = 'down' # 第二點更低，所以下降趨勢
    else:
        current_direction = 'up'
    
    for i in range(len(filtered)):
        curr_idx, curr_price, curr_type = filtered[i]
        
        if i == 0:
            # 第一個點不標記
            continue
        
        prev_idx, prev_price, prev_type = filtered[i-1]
        
        if current_direction == 'up':
            # 上升趨勢中
            if curr_price > prev_price:
                # 新高點
                label = 0  # HH
                extrema.append({
                    'idx': curr_idx,
                    'price': curr_price,
                    'label': 'HH',
                    'type': curr_type
                })
                labels[curr_idx] = 0
            else:
                # 低點突破 -> 轉向下降
                label = 2  # LH (Lower High - 下降起始)
                extrema.append({
                    'idx': curr_idx,
                    'price': curr_price,
                    'label': 'LH',
                    'type': curr_type
                })
                labels[curr_idx] = 2
                current_direction = 'down'
        
        else:  # current_direction == 'down'
            # 下降趨勢中
            if curr_price < prev_price:
                # 新低點
                label = 3  # LL
                extrema.append({
                    'idx': curr_idx,
                    'price': curr_price,
                    'label': 'LL',
                    'type': curr_type
                })
                labels[curr_idx] = 3
            else:
                # 高點突破 -> 轉向上升
                label = 1  # HL (Higher Low - 上升起始)
                extrema.append({
                    'idx': curr_idx,
                    'price': curr_price,
                    'label': 'HL',
                    'type': curr_type
                })
                labels[curr_idx] = 1
                current_direction = 'up'
    
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

def compare_pct_values(df, pct_values=[0.5, 1, 1.5, 2, 3, 5]):
    """
    比較不同的 pct 參數
    """
    print("\n" + "="*80)
    print("比較不同的 pct 參數")
    print("="*80)
    
    results = []
    for pct in pct_values:
        labels, extrema = generate_zigzag_correct(df, depth=12, pct=pct)
        
        hh = sum(1 for e in extrema if e['label'] == 'HH')
        hl = sum(1 for e in extrema if e['label'] == 'HL')
        lh = sum(1 for e in extrema if e['label'] == 'LH')
        ll = sum(1 for e in extrema if e['label'] == 'LL')
        total = len(extrema)
        pct_labeled = 100 * total / len(labels)
        
        results.append({
            'pct': pct,
            'total': total,
            'hh': hh,
            'hl': hl,
            'lh': lh,
            'll': ll,
            'pct_labeled': pct_labeled
        })
        print(f"{pct:5.1f}% -> {total:4d} 個極值點 (HH:{hh:3d} HL:{hl:3d} LH:{lh:3d} LL:{ll:3d}) - {pct_labeled:.2f}% of bars")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

print("="*80)
print("正確的 TradingView ZigZag 實現")
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
    print(f"  使用合成數據...")
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

print(f"  時間: {df['datetime'].iloc[0]} 至 {df['datetime'].iloc[-1]}")
print(f"  價格: {df['close'].min():.2f} - {df['close'].max():.2f}")

print("\n[2/4] 比較不同的 pct 參數...")
comparison = compare_pct_values(df, pct_values=[0.5, 0.75, 1, 1.5, 2, 3])

# 選擇較小的 pct 來得到更多信號
pct_choice = 1.0  # 1% 應該會有很多信號
print(f"\n[3/4] 用 pct={pct_choice}% 生成標籤...")
labels, extrema = generate_zigzag_correct(df, depth=12, pct=pct_choice)
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

print(f"\n  最近的極值點 (Last 30):")
for i, ext in enumerate(extrema[-30:]):
    print(f"    {i+1:2d}. Bar {ext['idx']:5d} | {ext['label']:2s} | Price: {ext['price']:10.2f}")

print(f"\n[4/4] 創建可視化 (只顯示最近 100 根)...")
fig, ax = visualize_zigzag(
    df,
    labels,
    extrema,
    title=f"ZigZag 模式 (pct={pct_choice}%) - 最近 100 根 K 線",
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
df_output.to_csv('zigzag_labels_correct.csv', index=False)
print(f"\n標籤已保存到 'zigzag_labels_correct.csv'")

print(f"\n下一步:")
print(f"  如果信號數量還是不對，試試其他 pct 值:")
for r in comparison:
    print(f"    pct={r['pct']:.2f}% -> {r['total']} 個極值點")
