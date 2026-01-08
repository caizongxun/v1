#!/usr/bin/env python3
"""
ZigZag - Fixed LH/HL labeling
Problem: Missing LH/HL between LL-HH
Solution: Ensure alternating H-L-H-L pattern preserved in zigzag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("ZigZag - Fixed LH/HL Labeling")
print("="*80)

class ZigZagFixed:
    """Fixed version that preserves alternating H-L pattern"""
    
    def __init__(self, depth=3, deviation=2, backstep=2):
        self.depth = depth
        self.deviation = deviation / 100.0
        self.backstep = backstep
    
    def find_extrema(self, highs, lows):
        """Find all local extrema"""
        n = len(highs)
        extrema = []
        if n < self.depth * 2:
            return extrema
        
        for i in range(self.depth, n - self.depth):
            # Check if local high
            left_high = np.max(highs[max(0, i - self.depth):i])
            right_high = np.max(highs[i + 1:min(n, i + self.depth + 1)])
            if highs[i] >= left_high and highs[i] >= right_high:
                extrema.append((i, highs[i], 'H'))
            
            # Check if local low
            left_low = np.min(lows[max(0, i - self.depth):i])
            right_low = np.min(lows[i + 1:min(n, i + self.depth + 1)])
            if lows[i] <= left_low and lows[i] <= right_low:
                extrema.append((i, lows[i], 'L'))
        
        return extrema
    
    def filter_extrema_strict(self, extrema):
        """Filter with deviation, PRESERVE alternating H-L-H-L pattern"""
        if not extrema:
            return []
        
        extrema = sorted(extrema, key=lambda x: x[0])
        
        # Step 1: Remove duplicates at same bar (keep more extreme)
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
        
        # Step 2: Apply deviation filter while PRESERVING alternation
        zigzag = []
        for idx, price, ptype in extrema:
            if not zigzag:
                zigzag.append((idx, price, ptype))
                continue
            
            last_idx, last_price, last_type = zigzag[-1]
            
            # CASE 1: Same type (both H or both L)
            if ptype == last_type:
                # Replace with more extreme
                if ptype == 'H' and price > last_price:
                    zigzag[-1] = (idx, price, ptype)
                elif ptype == 'L' and price < last_price:
                    zigzag[-1] = (idx, price, ptype)
                continue
            
            # CASE 2: Different type (H->L or L->H) - CHECK DEVIATION
            # This is the key: we MUST accept it to preserve alternation
            # We only skip if deviation threshold not met
            if ptype == 'H':
                threshold = last_price * (1 + self.deviation)
                if price >= threshold:
                    zigzag.append((idx, price, ptype))
                # If doesn't meet threshold, DON'T add (will be merged with next)
            else:  # ptype == 'L'
                threshold = last_price * (1 - self.deviation)
                if price <= threshold:
                    zigzag.append((idx, price, ptype))
                # If doesn't meet threshold, DON'T add
        
        return zigzag
    
    def label_zigzag_strict(self, zigzag):
        """Label with strict alternating pattern
        
        Key insight: Since we preserve H-L-H-L-H-L pattern,
        the labels should naturally cycle through all 4 types.
        """
        if len(zigzag) < 2:
            return []
        
        labeled = []
        
        # Point 1: No label (no previous point to compare)
        labeled.append({
            'idx': zigzag[0][0],
            'price': zigzag[0][1],
            'type': zigzag[0][2],
            'label': None,
            'direction': None,
            'lastPoint': None
        })
        
        # Point 2: No label yet
        direction = 1 if zigzag[1][1] > zigzag[0][1] else -1
        labeled.append({
            'idx': zigzag[1][0],
            'price': zigzag[1][1],
            'type': zigzag[1][2],
            'label': None,
            'direction': direction,
            'lastPoint': None
        })
        
        # Rest of points
        for i in range(2, len(zigzag)):
            curr_idx, curr_price, curr_type = zigzag[i]
            prev_idx, prev_price, prev_type = zigzag[i-1]
            prev2_idx, prev2_price, prev2_type = zigzag[i-2]
            
            # When we see a type change, the direction changes
            last_point = None
            
            if curr_type != prev_type:
                # Direction changed - update direction
                direction = 1 if curr_type == 'H' else -1
                # lastPoint is the previous pivot (before the last one)
                last_point = prev2_price
            else:
                # Same type - direction continues
                # This shouldn't happen if alternation preserved
                last_point = labeled[-1]['lastPoint']
                direction = labeled[-1]['direction']
            
            # Label based on direction and lastPoint
            label = None
            if last_point is not None:
                if direction > 0:  # Currently uptrend (H type)
                    label = 'HH' if curr_price > last_point else 'LH'
                else:  # Currently downtrend (L type)
                    label = 'LL' if curr_price < last_point else 'HL'
            
            labeled.append({
                'idx': curr_idx,
                'price': curr_price,
                'type': curr_type,
                'label': label,
                'direction': direction,
                'lastPoint': last_point
            })
        
        return labeled


def plot_zigzag_range(df, labeled_zigzag, depth, deviation, start_idx=None, end_idx=None, title_suffix=""):
    """Plot a specific range"""
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(df)
    
    fig, ax = plt.subplots(figsize=(28, 12))
    
    df_plot = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    # Filter points in range
    zz_plot = []
    for zz in labeled_zigzag:
        if start_idx <= zz['idx'] < end_idx:
            zz_copy = zz.copy()
            zz_copy['idx'] = zz['idx'] - start_idx
            zz_plot.append(zz_copy)
    
    ax.plot(df_plot.index, df_plot['high'], color='#e0e0e0', linewidth=0.4, alpha=0.3)
    ax.plot(df_plot.index, df_plot['low'], color='#e0e0e0', linewidth=0.4, alpha=0.3)
    ax.plot(df_plot.index, df_plot['close'], color='#333333', linewidth=1.2, alpha=0.8)
    
    colors = {
        'HH': '#0052cc',
        'HL': '#ff6b35',
        'LH': '#16a34a',
        'LL': '#dc2626',
        None: '#999999'
    }
    
    # Plot points
    for zz in zz_plot:
        idx = zz['idx']
        price = zz['price']
        label = zz['label']
        
        color = colors.get(label, '#999999')
        
        if label:
            ax.scatter(idx, price, color=color, s=350, marker='o', 
                      edgecolors='black', linewidth=2, zorder=5, alpha=0.95)
            ax.text(idx, price * 1.005, label, ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            alpha=0.85, edgecolor=color, linewidth=1))
        else:
            ax.scatter(idx, price, color='gray', s=120, marker='s',
                      edgecolors='black', linewidth=1.5, zorder=5, alpha=0.5)
    
    # Draw lines
    if len(zz_plot) > 1:
        for i in range(len(zz_plot) - 1):
            x1, y1 = zz_plot[i]['idx'], zz_plot[i]['price']
            x2, y2 = zz_plot[i+1]['idx'], zz_plot[i+1]['price']
            next_label = zz_plot[i+1]['label']
            line_color = colors.get(next_label, '#aaaaaa')
            ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=2, alpha=0.7, zorder=3)
    
    bar_count = end_idx - start_idx
    ax.set_title(f"BTC-USD ZigZag (depth={depth}, deviation={deviation}%) | {bar_count} bars{title_suffix}", 
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Bar Index', fontsize=12)
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

print("\n[2/5] Finding extrema (depth=3)...")
zz = ZigZagFixed(depth=3, deviation=2, backstep=2)
extrema = zz.find_extrema(df['high'].values, df['low'].values)
print(f"  Found {len(extrema)} extrema points")

print("\n[3/5] Filtering with strict alternation...")
zigzag = zz.filter_extrema_strict(extrema)
print(f"  {len(zigzag)} zigzag points (STRICTLY alternating H-L-H-L...)")

# Verify alternation
if len(zigzag) > 1:
    types = [z[2] for z in zigzag]
    alternating = all(types[i] != types[i+1] for i in range(len(types)-1))
    print(f"  Alternation preserved: {alternating}")
    if not alternating:
        print("  WARNING: Alternation NOT preserved!")

print(f"\n  Zigzag points (first 30):")
for i, (idx, price, ptype) in enumerate(zigzag[:30], 1):
    print(f"    {i:3d}. Bar {idx:5d} | Type {ptype} | Price {price:10.2f}")

print("\n[4/5] Labeling with correct LH/HL...")
labeled = zz.label_zigzag_strict(zigzag)

hh = sum(1 for l in labeled if l['label'] == 'HH')
hl = sum(1 for l in labeled if l['label'] == 'HL')
lh = sum(1 for l in labeled if l['label'] == 'LH')
ll = sum(1 for l in labeled if l['label'] == 'LL')
other = sum(1 for l in labeled if l['label'] is None)
total = len(labeled)

print(f"\n  Label distribution:")
print(f"    HH: {hh:5d} ({100*hh/total:5.2f}%) - Higher High")
print(f"    HL: {hl:5d} ({100*hl/total:5.2f}%) - Higher Low")
print(f"    LH: {lh:5d} ({100*lh/total:5.2f}%) - Lower High")
print(f"    LL: {ll:5d} ({100*ll/total:5.2f}%) - Lower Low")
print(f"    Unlabeled: {other:5d} ({100*other/total:5.2f}%)")
print(f"    TOTAL: {total:5d}")

if len(labeled) > 1:
    bar_diffs = [labeled[i+1]['idx'] - labeled[i]['idx'] for i in range(len(labeled)-1)]
    print(f"\n  Bar spacing:")
    print(f"    Avg: {np.mean(bar_diffs):.1f} bars between points")
    print(f"    Min: {np.min(bar_diffs)} bars")
    print(f"    Max: {np.max(bar_diffs)} bars")

print(f"\n  Last 40 labeled points:")
print(f"  {'#':>3} {'Bar':>6} {'Type':>5} {'Label':>8} {'Price':>12} {'LastPt':>12} {'Dir':>4}")
print("  " + "-"*75)
for i, l in enumerate(labeled[-40:], 1):
    label = l['label'] if l['label'] else 'START'
    direction = 'UP' if l['direction'] == 1 else ('DN' if l['direction'] == -1 else 'N/A')
    lastpt = f"{l['lastPoint']:.0f}" if l['lastPoint'] else 'None'
    print(f"  {i:3d} {l['idx']:6d} {l['type']:>5s} {label:>8s} {l['price']:12.0f} {lastpt:>12s} {direction:>4s}")

print("\n[5/5] Creating visualizations...")

# Last 1000 bars
print("  Creating last 1000 bars chart...")
start = max(0, len(df) - 1000)
fig, ax = plot_zigzag_range(
    df, labeled, 3, 2,
    start_idx=start,
    end_idx=len(df),
    title_suffix=" | Last 1000 bars"
)
plt.savefig('zigzag_fixed_last1000.png', dpi=150, bbox_inches='tight')
print("  OK - Saved to zigzag_fixed_last1000.png")
plt.close()

# All bars
print("  Creating full chart...")
fig, ax = plot_zigzag_range(
    df, labeled, 3, 2,
    start_idx=0,
    end_idx=len(df),
    title_suffix=f" | All {len(df)} bars"
)
plt.savefig('zigzag_fixed_all.png', dpi=100, bbox_inches='tight')
print("  OK - Saved to zigzag_fixed_all.png")
plt.close()

print("\n" + "="*80)
print("Complete! Now you should see LH/HL between LL-HH transitions.")
print("="*80 + "\n")
