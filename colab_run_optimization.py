#!/usr/bin/env python3
"""
ZigZag 模型优化 - Colab 快速运行脚本
直接复制到 Colab 执行即可
"""

import subprocess
import sys

print("\n" + "="*80)
print("ZigZag 模型优化 - Colab 快速执行")
print("="*80)

# 检查必要的库
print("\n检查必要的库...")

required_packages = {
    'yfinance': 'yfinance',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scikit-learn': 'sklearn',
    'xgboost': 'xgboost'
}

missing_packages = []

for package_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"  ✓ {package_name}")
    except ImportError:
        print(f"  ✗ {package_name} (缺失)")
        missing_packages.append(package_name)

if missing_packages:
    print(f"\n安装缺失的包: {', '.join(missing_packages)}...")
    for package in missing_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    print("✓ 所有包已安装")

# 下载并执行优化脚本
print("\n" + "="*80)
print("下载优化脚本...")
print("="*80)

import urllib.request

url = "https://raw.githubusercontent.com/caizongxun/v1/main/train_optimized_model.py"

try:
    print(f"正在从 GitHub 下载...")
    code = urllib.request.urlopen(url).read().decode('utf-8')
    print("✓ 下载成功")
    
    print("\n" + "="*80)
    print("执行优化脚本...")
    print("="*80 + "\n")
    
    # 执行脚本
    exec(code)
    
except urllib.error.URLError as e:
    print(f"✗ 下载失败: {e}")
    print("\n请检查网络连接或手动访问:")
    print("https://github.com/caizongxun/v1/blob/main/train_optimized_model.py")
except Exception as e:
    print(f"✗ 执行失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("脚本执行完成")
print("="*80)
