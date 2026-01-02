#!/bin/bash

# 彻底清理所有 Python 缓存和编译文件，然后重新编译 Cython 模块
# 修改代码后运行此脚本可避免缓存问题导致的运行时异常

set -e  # 遇到错误立即退出

# 获取脚本所在目录（支持从任意位置执行）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "彻底清理 Python 缓存和编译文件..."
echo "工作目录: $(pwd)"
echo "=========================================="

# 1. 清理 __pycache__ 目录（使用 -depth 确保先删除子目录）
echo "[1/7] 清理 __pycache__ 目录..."
find . -depth -type d -name "__pycache__" -exec rm -rf {} \; 2>/dev/null || true

# 2. 清理 .pyc 和 .pyo 文件
echo "[2/7] 清理 .pyc 和 .pyo 文件..."
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# 3. 彻底删除 build 目录（包含旧的 .so 文件）
echo "[3/7] 删除 build 目录..."
rm -rf build/ 2>/dev/null || true
echo "    build/ 已清理"

# 4. 清理 Cython 编译产物（.so 和 .c 文件）
echo "[4/7] 清理 Cython 编译产物..."
find ./src -type f -name "*.so" -delete 2>/dev/null || true
find ./src -type f -name "*.c" -delete 2>/dev/null || true
# 同时清理 .pyd (Windows) 和 .html (Cython annotate)
find ./src -type f -name "*.pyd" -delete 2>/dev/null || true
find ./src -type f -name "*.html" -delete 2>/dev/null || true
echo "    Cython 产物已清理"

# 5. 清理其他缓存目录
echo "[5/7] 清理其他缓存..."
rm -rf .pytest_cache 2>/dev/null || true
rm -rf .mypy_cache 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
rm -rf dist 2>/dev/null || true
rm -rf .eggs 2>/dev/null || true
rm -rf .ruff_cache 2>/dev/null || true

# 6. 清理 pip 缓存中可能残留的本项目模块（可选，更彻底）
echo "[6/7] 清理 Python importlib 缓存..."
python3 -c "import importlib; importlib.invalidate_caches()" 2>/dev/null || true

# 7. 重新编译 Cython 模块（使用 --force 强制重新编译）
echo "[7/7] 重新编译 Cython 模块..."
echo "=========================================="
# 设置环境变量禁止生成 .pyc 文件（编译期间）
PYTHONDONTWRITEBYTECODE=1 python setup.py build_ext --inplace --force

echo ""
echo "=========================================="
echo "清理和编译完成！"
echo "=========================================="

# 验证编译结果
echo ""
echo "编译的 .so 文件："
find ./src -name "*.so" -type f 2>/dev/null || echo "（无）"

# 验证清理结果
echo ""
echo "残留的 __pycache__ 目录数量："
pycache_count=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
echo "    $pycache_count 个"

echo ""
echo "残留的 .c 文件数量："
c_count=$(find ./src -type f -name "*.c" 2>/dev/null | wc -l)
echo "    $c_count 个"
