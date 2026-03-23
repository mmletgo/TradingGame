"""
快速数学函数模块（Numba JIT 加速）

提供对数归一化等高频数学运算的加速实现。
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


# 类型协议（支持默认参数）
class LogNormalizeFunc(Protocol):
    def __call__(
        self, arr: NDArray[np.float32], scale: float = 10.0
    ) -> NDArray[np.float32]: ...


def _log_normalize_unsigned_numpy(
    arr: NDArray[np.float32], scale: float = 10.0
) -> NDArray[np.float32]:
    """无符号对数归一化：log10(x + 1) / scale（纯 NumPy 版本）"""
    return (np.log10(arr + 1.0) / scale).astype(np.float32)


def _log_normalize_signed_numpy(
    arr: NDArray[np.float32], scale: float = 10.0
) -> NDArray[np.float32]:
    """带符号对数归一化（纯 NumPy 版本）"""
    return (np.sign(arr) * np.log10(np.abs(arr) + 1.0) / scale).astype(np.float32)


# 尝试导入 Numba 并创建 JIT 加速版本
try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _log_normalize_unsigned_numba(
        arr: NDArray[np.float32], scale: float = 10.0
    ) -> NDArray[np.float32]:
        """无符号对数归一化：log10(x + 1) / scale"""
        result = np.empty_like(arr)
        for i in range(len(arr)):
            result[i] = np.log10(arr[i] + 1.0) / scale
        return result

    @njit(cache=True, fastmath=True)
    def _log_normalize_signed_numba(
        arr: NDArray[np.float32], scale: float = 10.0
    ) -> NDArray[np.float32]:
        """带符号对数归一化：sign(x) * log10(|x| + 1) / scale"""
        result = np.empty_like(arr)
        for i in range(len(arr)):
            val = arr[i]
            if val >= 0:
                result[i] = np.log10(val + 1.0) / scale
            else:
                result[i] = -np.log10(-val + 1.0) / scale
        return result

    HAS_NUMBA = True
    log_normalize_unsigned: LogNormalizeFunc = _log_normalize_unsigned_numba
    log_normalize_signed: LogNormalizeFunc = _log_normalize_signed_numba

except ImportError:
    HAS_NUMBA = False
    log_normalize_unsigned: LogNormalizeFunc = _log_normalize_unsigned_numpy  # type: ignore[no-redef]
    log_normalize_signed: LogNormalizeFunc = _log_normalize_signed_numpy  # type: ignore[no-redef]
