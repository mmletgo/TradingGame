#!/bin/bash

# 彻底清理所有 Python 缓存和编译文件，然后重新编译 Cython 模块
# 修改代码后运行此脚本可避免缓存问题导致的运行时异常

set -e  # 遇到错误立即退出

echo "=========================================="
echo "彻底清理 Python 缓存和编译文件..."
echo "=========================================="

# 1. 清理 __pycache__ 目录
echo "[1/6] 清理 __pycache__ 目录..."
find . -type d -name "__pycache__" -print -exec rm -rf {} + 2>/dev/null || true

# 2. 清理 .pyc 和 .pyo 文件
echo "[2/6] 清理 .pyc 和 .pyo 文件..."
find . -name "*.pyc" -print -delete 2>/dev/null || true
find . -name "*.pyo" -print -delete 2>/dev/null || true

# 3. 彻底删除 build 目录（包含旧的 .so 文件）
echo "[3/6] 删除 build 目录..."
if [ -d "build" ]; then
    rm -rf build/
    echo "    已删除 build/"
fi

# 4. 清理 src 目录下的 Cython 编译产物
echo "[4/6] 清理 Cython 编译产物..."
find ./src -name "*.so" -print -delete 2>/dev/null || true
find ./src -name "*.c" -print -delete 2>/dev/null || true

# 5. 清理其他缓存目录
echo "[5/6] 清理其他缓存..."
rm -rf .pytest_cache 2>/dev/null || true
rm -rf .mypy_cache 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
rm -rf dist 2>/dev/null || true
rm -rf .eggs 2>/dev/null || true

# 6. 重新编译 Cython 模块
echo "[6/6] 重新编译 Cython 模块..."
echo "=========================================="
python setup.py build_ext --inplace

echo ""
echo "=========================================="
echo "✅ 清理和编译完成！"
echo "=========================================="

# 验证编译结果
echo ""
echo "编译的 .so 文件："
find ./src -name "*.so" -print
