#!/bin/bash

# 清理所有 Python 缓存和编译文件，然后重新编译 Cython 模块
# 修改代码后运行此脚本可避免缓存问题导致的运行时异常

set -e  # 遇到错误立即退出

echo "正在清理 Python 缓存和编译文件..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
find ./src -name "*.so" -delete 2>/dev/null
find ./src -name "*.c" -delete 2>/dev/null
rm -rf build/

echo "清理完成！"
echo "正在重新编译 Cython 模块..."
python setup.py build_ext --inplace

echo "✅ 重新编译完成！"
