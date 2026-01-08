#!/usr/bin/env python3
"""
ZigZag Debug Version - Check why only 1 point is shown
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("ZigZag - Debug Version")
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


# MAIN
print("\n[1/6] Fetching data...")
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

print("\n[2/6] Finding extrema...")
zz = ZigZagExact(depth=12, deviation=5, backstep=2)
extrema = zz.find_extrema(df['high'].values, df['low'].values)
print(f"  OK - Found {len(extrema)} extrema points")
print(f"       Extrema bar range: {min(e[0] for e in extrema)} to {max(e[0] for e in extrema)}")

print("\n[3/6] Filtering extrema...")
zigzag = zz.filter_extrema(extrema)
print(f"  OK - {len(zigzag)} zigzag points after filtering")
if zigzag:
    print(f"       ZigZag bar range: {zigzag[0][0]} to {zigzag[-1][0]}")
    print(f"\n       ZigZag points (all):")
    for i, (idx, price, ptype) in enumerate(zigzag):
        print(f"         {i+1:3d}. Bar {idx:5d} | Type {ptype} | Price {price:10.2f}")

print("\n[4/6] Labeling...")
labeled = zz.label_zigzag(zigzag)
print(f"  OK - Labeled {len(labeled)} points")

hh = sum(1 for l in labeled if l['label'] == 'HH')
hl = sum(1 for l in labeled if l['label'] == 'HL')
lh = sum(1 for l in labeled if l['label'] == 'LH')
ll = sum(1 for l in labeled if l['label'] == 'LL')
unlabeled = sum(1 for l in labeled if l['label'] is None)
total = len(labeled)

print(f"\n  Label distribution:")
print(f"    HH: {hh:5d} ({100*hh/total:5.2f}%)")
print(f"    HL: {hl:5d} ({100*hl/total:5.2f}%)")
print(f"    LH: {lh:5d} ({100*lh/total:5.2f}%)")
print(f"    LL: {ll:5d} ({100*ll/total:5.2f}%)")
print(f"    Unlabeled: {unlabeled:5d} ({100*unlabeled/total:5.2f}%)")
print(f"    TOTAL: {total:5d}")

print(f"\n  ALL labeled points:")
print(f"  {'#':>3} {'Bar':>6} {'Label':>8} {'Type':>5} {'Price':>12}")
print("  " + "-"*40)
for i, l in enumerate(labeled, 1):
    label = l['label'] if l['label'] else 'START'
    print(f"  {i:3d} {l['idx']:6d} {label:>8s} {l['type']:>5s} {l['price']:12.0f}")

print("\n[5/6] Creating visualization with ALL bars...")

# KEY FIX: Show ALL bars, not just last 150
fig, ax = plt.subplots(figsize=(28, 12))

# Plot entire dataset
ax.plot(df.index, df['high'], color='#e0e0e0', linewidth=0.3, alpha=0.3)
ax.plot(df.index, df['low'], color='#e0e0e0', linewidth=0.3, alpha=0.3)
ax.plot(df.index, df['close'], color='#333333', linewidth=0.8, alpha=0.7)

colors = {
    'HH': '#0052cc',
    'HL': '#ff6b35',
    'LH': '#16a34a',
    'LL': '#dc2626',
    None: '#999999'
}

print(f"\n  Plotting {len(labeled)} points on {len(df)} bars...")

# Plot ALL labeled points
for i, l in enumerate(labeled):
    idx = l['idx']
    price = l['price']
    label = l['label']
    
    color = colors.get(label, '#999999')
    
    if label:
        ax.scatter(idx, price, color=color, s=300, marker='o', 
                  edgecolors='black', linewidth=2, zorder=5, alpha=0.9)
        ax.text(idx, price * 1.003, label, ha='center', va='bottom',
               fontsize=9, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.8, edgecolor='none'))
    else:
        ax.scatter(idx, price, color='gray', s=100, marker='s',
                  edgecolors='black', linewidth=1.5, zorder=5, alpha=0.4)

# Draw zigzag lines
if len(labeled) > 1:
    for i in range(len(labeled) - 1):
        x1, y1 = labeled[i]['idx'], labeled[i]['price']
        x2, y2 = labeled[i+1]['idx'], labeled[i+1]['price']
        next_label = labeled[i+1]['label']
        line_color = colors.get(next_label, '#aaaaaa')
        ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=1.5, alpha=0.6, zorder=3)

ax.set_title("BTC-USD ZigZag (All Bars, depth=12, deviation=5%)", 
            fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Bar Index (Full Range)', fontsize=12)
ax.set_ylabel('Price (USDT)', fontsize=12)
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
ax.set_facecolor('#fafafa')

legend_elements = [
    mpatches.Patch(facecolor='#0052cc', edgecolor='black', label='HH - Higher High'),
    mpatches.Patch(facecolor='#ff6b35', edgecolor='black', label='HL - Higher Low'),
    mpatches.Patch(facecolor='#16a34a', edgecolor='black', label='LH - Lower High'),
    mpatches.Patch(facecolor='#dc2626', edgecolor='black', label='LL - Lower Low'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('zigzag_debug_all_bars.png', dpi=120, bbox_inches='tight')
print(f"  OK - Saved to zigzag_debug_all_bars.png (ALL {len(df)} bars shown)")
plt.show()

print("\n[6/6] Creating zoomed-in visualization (last 300 bars)...")

fig, ax = plt.subplots(figsize=(28, 12))

start_idx = max(0, len(df) - 300)
df_zoom = df.iloc[start_idx:].reset_index(drop=True)

ax.plot(df_zoom.index, df_zoom['high'], color='#e0e0e0', linewidth=0.4, alpha=0.4)
ax.plot(df_zoom.index, df_zoom['low'], color='#e0e0e0', linewidth=0.4, alpha=0.4)
ax.plot(df_zoom.index, df_zoom['close'], color='#333333', linewidth=1, alpha=0.8)

# Filter and plot labeled points in zoom range
for l in labeled:
    if start_idx <= l['idx'] < len(df):
        idx = l['idx'] - start_idx
        price = l['price']
        label = l['label']
        color = colors.get(label, '#999999')
        
        if label:
            ax.scatter(idx, price, color=color, s=350, marker='o', 
                      edgecolors='black', linewidth=2.2, zorder=5, alpha=0.95)
            ax.text(idx, price * 1.005, label, ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            alpha=0.85, edgecolor=color, linewidth=1))
        else:
            ax.scatter(idx, price, color='gray', s=120, marker='s',
                      edgecolors='black', linewidth=1.8, zorder=5, alpha=0.5)

# Draw lines
labeled_zoom = [l for l in labeled if start_idx <= l['idx'] < len(df)]
if len(labeled_zoom) > 1:
    for i in range(len(labeled_zoom) - 1):
        x1 = labeled_zoom[i]['idx'] - start_idx
        y1 = labeled_zoom[i]['price']
        x2 = labeled_zoom[i+1]['idx'] - start_idx
        y2 = labeled_zoom[i+1]['price']
        next_label = labeled_zoom[i+1]['label']
        line_color = colors.get(next_label, '#aaaaaa')
        ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=2, alpha=0.7, zorder=3)

ax.set_title(f"BTC-USD ZigZag (Last 300 bars, depth=12, deviation=5%)", 
            fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Bar Index (Last 300)', fontsize=12)
ax.set_ylabel('Price (USDT)', fontsize=12)
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
ax.set_facecolor('#fafafa')
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('zigzag_debug_zoom.png', dpi=150, bbox_inches='tight')
print(f"  OK - Saved to zigzag_debug_zoom.png")
plt.show()

print("\n" + "="*80)
print(f"Complete! Found {total} total points, {hh+hl+lh+ll} labeled")
print("="*80 + "\n")
