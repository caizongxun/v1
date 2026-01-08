#!/usr/bin/env python3
"""
Colab ZigZag - Exact Pine Script Logic
One-line execution in Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("ZigZag - Exact Pine Script Implementation (Colab)")
print("="*80)

class ZigZagExact:
    def __init__(self, depth=12, deviation=5, backstep=2):
        self.depth = depth
        self.deviation = deviation / 100.0
        self.backstep = backstep
    
    def find_extrema(self, highs, lows):
        n = len(highs)
        extrema = []
        if n < self.depth * 2:
            return extrema
        
        for i in range(self.depth, n - self.depth):
            left_high = np.max(highs[max(0, i - self.depth):i])
            right_high = np.max(highs[i + 1:min(n, i + self.depth + 1)])
            if highs[i] >= left_high and highs[i] >= right_high:
                extrema.append((i, highs[i], 'H'))
            
            left_low = np.min(lows[max(0, i - self.depth):i])
            right_low = np.min(lows[i + 1:min(n, i + self.depth + 1)])
            if lows[i] <= left_low and lows[i] <= right_low:
                extrema.append((i, lows[i], 'L'))
        
        return extrema
    
    def filter_extrema(self, extrema):
        if not extrema:
            return []
        
        extrema = sorted(extrema, key=lambda x: x[0])
        
        filtered_dict = {}
        for idx, price, ptype in extrema:
            if idx not in filtered_dict:
                filtered_dict[idx] = (idx, price, ptype)
            else:
                existing = filtered_dict[idx]
                if ptype == 'H' and existing[2] == 'H':
                    if price > existing[1]:
                        filtered_dict[idx] = (idx, price, ptype)
                elif ptype == 'L' and existing[2] == 'L':
                    if price < existing[1]:
                        filtered_dict[idx] = (idx, price, ptype)
                elif ptype == 'H':
                    filtered_dict[idx] = (idx, price, ptype)
                else:
                    filtered_dict[idx] = (idx, price, ptype)
        
        extrema = sorted(filtered_dict.values(), key=lambda x: x[0])
        
        zigzag = []
        for idx, price, ptype in extrema:
            if not zigzag:
                zigzag.append((idx, price, ptype))
                continue
            
            last_idx, last_price, last_type = zigzag[-1]
            
            if ptype == last_type:
                if ptype == 'H' and price > last_price:
                    zigzag[-1] = (idx, price, ptype)
                elif ptype == 'L' and price < last_price:
                    zigzag[-1] = (idx, price, ptype)
                continue
            
            if ptype == 'H':
                threshold = last_price * (1 + self.deviation)
                if price >= threshold:
                    zigzag.append((idx, price, ptype))
            else:
                threshold = last_price * (1 - self.deviation)
                if price <= threshold:
                    zigzag.append((idx, price, ptype))
        
        return zigzag
    
    def label_zigzag(self, zigzag):
        if len(zigzag) < 2:
            return []
        
        labeled = []
        
        labeled.append({
            'idx': zigzag[0][0],
            'price': zigzag[0][1],
            'type': zigzag[0][2],
            'label': None,
            'direction': None,
            'lastPoint': None
        })
        
        if zigzag[1][1] > zigzag[0][1]:
            direction = 1
        else:
            direction = -1
        
        labeled.append({
            'idx': zigzag[1][0],
            'price': zigzag[1][1],
            'type': zigzag[1][2],
            'label': None,
            'direction': direction,
            'lastPoint': None
        })
        
        for i in range(2, len(zigzag)):
            curr_idx, curr_price, curr_type = zigzag[i]
            prev_idx, prev_price, prev_type = zigzag[i-1]
            
            new_direction = direction
            last_point = None
            
            if curr_type != prev_type:
                if curr_type == 'H':
                    new_direction = 1
                else:
                    new_direction = -1
                
                last_point = prev_price
                direction = new_direction
            else:
                last_point = labeled[-1]['lastPoint']
            
            if last_point is not None:
                if direction > 0:
                    label = 'HH' if curr_price > last_point else 'LH'
                else:
                    label = 'LL' if curr_price < last_point else 'HL'
            else:
                label = None
            
            labeled.append({
                'idx': curr_idx,
                'price': curr_price,
                'type': curr_type,
                'label': label,
                'direction': direction,
                'lastPoint': last_point
            })
        
        return labeled


def plot_zigzag(df, labeled_zigzag, title="ZigZag Analysis", num_bars=150):
    fig, ax = plt.subplots(figsize=(22, 11))
    
    start_idx = max(0, len(df) - num_bars)
    end_idx = len(df)
    df_plot = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    zz_plot = []
    for zz in labeled_zigzag:
        if start_idx <= zz['idx'] < end_idx:
            zz_copy = zz.copy()
            zz_copy['idx'] = zz['idx'] - start_idx
            zz_plot.append(zz_copy)
    
    ax.plot(df_plot.index, df_plot['high'], color='#e0e0e0', linewidth=0.5, alpha=0.4)
    ax.plot(df_plot.index, df_plot['low'], color='#e0e0e0', linewidth=0.5, alpha=0.4)
    ax.plot(df_plot.index, df_plot['close'], color='#333333', linewidth=1.2, alpha=0.8)
    
    colors = {
        'HH': '#0052cc',
        'HL': '#ff6b35',
        'LH': '#16a34a',
        'LL': '#dc2626',
        None: '#999999'
    }
    
    for zz in zz_plot:
        idx = zz['idx']
        price = zz['price']
        label = zz['label']
        
        color = colors.get(label, '#999999')
        
        if label:
            ax.scatter(idx, price, color=color, s=400, marker='o', 
                      edgecolors='black', linewidth=2.5, zorder=5, alpha=0.95)
            ax.text(idx, price * 1.008, label, ha='center', va='bottom',
                   fontsize=12, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            alpha=0.85, edgecolor=color, linewidth=1.5))
        else:
            ax.scatter(idx, price, color='gray', s=150, marker='s',
                      edgecolors='black', linewidth=1.8, zorder=5, alpha=0.5)
    
    if len(zz_plot) > 1:
        zz_sorted = sorted(zz_plot, key=lambda x: x['idx'])
        for i in range(len(zz_sorted) - 1):
            x1, y1 = zz_sorted[i]['idx'], zz_sorted[i]['price']
            x2, y2 = zz_sorted[i+1]['idx'], zz_sorted[i+1]['price']
            next_label = zz_sorted[i+1]['label']
            line_color = colors.get(next_label, '#999999')
            ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=2.8, alpha=0.7, zorder=3)
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Bar Index (from right)', fontsize=13, fontweight='600')
    ax.set_ylabel('Price (USDT)', fontsize=13, fontweight='600')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_facecolor('#f9f9f9')
    
    legend_elements = [
        mpatches.Patch(facecolor='#0052cc', edgecolor='black', label='HH - Higher High'),
        mpatches.Patch(facecolor='#ff6b35', edgecolor='black', label='HL - Higher Low'),
        mpatches.Patch(facecolor='#16a34a', edgecolor='black', label='LH - Lower High'),
        mpatches.Patch(facecolor='#dc2626', edgecolor='black', label='LL - Lower Low'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
             framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    
    plt.tight_layout()
    return fig, ax


# MAIN
print("\n[1/5] Fetching data...")
try:
    df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"  OK - {len(df):,} bars loaded")
except Exception as e:
    print(f"  ERROR: {e}")
    exit(1)

print("\n[2/5] Finding extrema...")
zz = ZigZagExact(depth=12, deviation=5, backstep=2)
extrema = zz.find_extrema(df['high'].values, df['low'].values)
print(f"  OK - Found {len(extrema)} extrema")

print("\n[3/5] Filtering extrema...")
zigzag = zz.filter_extrema(extrema)
print(f"  OK - {len(zigzag)} zigzag points")

print("\n[4/5] Labeling (HH/HL/LH/LL)...")
labeled = zz.label_zigzag(zigzag)
print(f"  OK - Labeled {len(labeled)} points")

hh = sum(1 for l in labeled if l['label'] == 'HH')
hl = sum(1 for l in labeled if l['label'] == 'HL')
lh = sum(1 for l in labeled if l['label'] == 'LH')
ll = sum(1 for l in labeled if l['label'] == 'LL')
unlabeled = sum(1 for l in labeled if l['label'] is None)
total = len(labeled)

print(f"\n  Distribution:")
print(f"    HH: {hh:5d} ({100*hh/total:5.2f}%)")
print(f"    HL: {hl:5d} ({100*hl/total:5.2f}%)")
print(f"    LH: {lh:5d} ({100*lh/total:5.2f}%)")
print(f"    LL: {ll:5d} ({100*ll/total:5.2f}%)")
print(f"    Unlabeled: {unlabeled:5d} ({100*unlabeled/total:5.2f}%)")
print(f"    TOTAL: {total:5d}")

print(f"\n  Last 30 points:")
print(f"  {'#':>3} {'Bar':>6} {'Label':>8} {'Type':>5} {'Price':>12} {'LastPt':>12} {'Dir':>4}")
print("  " + "-"*75)
for i, l in enumerate(labeled[-30:], 1):
    label = l['label'] if l['label'] else 'START'
    direction = 'UP' if l['direction'] == 1 else ('DN' if l['direction'] == -1 else 'N/A')
    lastpt = f"{l['lastPoint']:.0f}" if l['lastPoint'] else 'None'
    print(f"  {i:3d} {l['idx']:6d} {label:>8s} {l['type']:>5s} {l['price']:12.0f} {lastpt:>12s} {direction:>4s}")

print("\n[5/5] Creating visualization...")
fig, ax = plot_zigzag(df, labeled, 
                     title="BTC-USD ZigZag (1h, depth=12, deviation=5%)",
                     num_bars=150)
plt.savefig('zigzag_exact.png', dpi=150, bbox_inches='tight')
print("  OK - Saved to zigzag_exact.png")
plt.show()

export_df = pd.DataFrame([
    {
        'bar': l['idx'],
        'price': l['price'],
        'type': l['type'],
        'label': l['label'],
        'direction': 'UP' if l['direction'] == 1 else ('DOWN' if l['direction'] == -1 else None),
        'lastPoint': l['lastPoint']
    }
    for l in labeled
])
export_df.to_csv('zigzag_labeled.csv', index=False)
print("\nExported to zigzag_labeled.csv")

print("\n" + "="*80)
print("Complete!")
print("="*80 + "\n")
