#!/usr/bin/env python3
"""
ZigZag with Tunable Parameters
Adjust depth, deviation to change granularity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("ZigZag - Tunable Parameters Version")
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

# TEST MULTIPLE CONFIGURATIONS
configs = [
    (12, 5, "original_depth12_dev5"),
    (6, 5, "small_depth6_dev5"),
    (4, 3, "tiny_depth4_dev3"),
    (3, 2, "micro_depth3_dev2"),
]

for depth, deviation, config_name in configs:
    print(f"\n{'='*80}")
    print(f"Config: depth={depth}, deviation={deviation}%")
    print(f"{'='*80}")
    
    print(f"\n[2/5] Finding extrema (depth={depth})...")
    zz = ZigZagExact(depth=depth, deviation=deviation, backstep=2)
    extrema = zz.find_extrema(df['high'].values, df['low'].values)
    print(f"  Found {len(extrema)} extrema points")
    
    print(f"\n[3/5] Filtering...")
    zigzag = zz.filter_extrema(extrema)
    print(f"  {len(zigzag)} zigzag points")
    
    print(f"\n[4/5] Labeling...")
    labeled = zz.label_zigzag(zigzag)
    
    hh = sum(1 for l in labeled if l['label'] == 'HH')
    hl = sum(1 for l in labeled if l['label'] == 'HL')
    lh = sum(1 for l in labeled if l['label'] == 'LH')
    ll = sum(1 for l in labeled if l['label'] == 'LL')
    total = len(labeled)
    
    print(f"\n  Distribution:")
    print(f"    HH: {hh:5d} ({100*hh/total:5.2f}%) - Higher High")
    print(f"    HL: {hl:5d} ({100*hl/total:5.2f}%) - Higher Low")
    print(f"    LH: {lh:5d} ({100*lh/total:5.2f}%) - Lower High")
    print(f"    LL: {ll:5d} ({100*ll/total:5.2f}%) - Lower Low")
    print(f"    TOTAL: {total:5d}")
    
    # Calculate average bars between points
    if len(labeled) > 1:
        bar_diffs = [labeled[i+1]['idx'] - labeled[i]['idx'] for i in range(len(labeled)-1)]
        avg_bars = np.mean(bar_diffs)
        min_bars = np.min(bar_diffs)
        max_bars = np.max(bar_diffs)
        print(f"\n  Bar spacing:")
        print(f"    Avg: {avg_bars:.1f} bars between points")
        print(f"    Min: {min_bars} bars")
        print(f"    Max: {max_bars} bars")
    
    # Plot last 1000 bars
    print(f"\n[5/5] Creating visualization (last 1000 bars)...")
    start = max(0, len(df) - 1000)
    fig, ax = plot_zigzag_range(
        df, labeled, depth, deviation,
        start_idx=start,
        end_idx=len(df),
        title_suffix=" | Last 1000 bars"
    )
    plt.savefig(f'zigzag_{config_name}_last1000.png', dpi=150, bbox_inches='tight')
    print(f"  Saved to zigzag_{config_name}_last1000.png")
    plt.close()
    
    # Plot all bars
    print(f"  Creating full chart...")
    fig, ax = plot_zigzag_range(
        df, labeled, depth, deviation,
        start_idx=0,
        end_idx=len(df),
        title_suffix=f" | All {len(df)} bars"
    )
    plt.savefig(f'zigzag_{config_name}_all.png', dpi=100, bbox_inches='tight')
    print(f"  Saved to zigzag_{config_name}_all.png")
    plt.close()

print(f"\n\n" + "="*80)
print("SUMMARY - Compare the different configurations:")
print("="*80)
print("""
Depth parameter effect:
  depth=12, dev=5%  → Large zigzag, ~161 points (broad trends)
  depth=6, dev=5%   → Medium zigzag, more points (normal trends)
  depth=4, dev=3%   → Small zigzag, many points (swing trading)
  depth=3, dev=2%   → Tiny zigzag, dense points (scalping)

Small depth = more frequent label changes = better for swing/scalp trading
Large depth = fewer label changes = better for trend following

Adjust depth and deviation based on your trading timeframe!
""")
print("="*80 + "\n")
