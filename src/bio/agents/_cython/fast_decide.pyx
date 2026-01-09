# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""快速决策辅助函数 - Cython 实现

提供 Agent decide() 方法中的关键计算路径的加速实现：
- fast_argmax: 快速查找最大值索引
- fast_round_price: 快速价格取整
- fast_copy_to_buffer: 快速数据复制
"""

import numpy as np
cimport numpy as np
from libc.math cimport fmax, round as c_round

ctypedef np.float64_t DTYPE_t


cpdef int fast_argmax(double[:] arr, int start, int end) noexcept nogil:
    """快速 argmax，释放 GIL

    在指定范围内查找最大值的索引。

    Args:
        arr: 输入数组（memoryview）
        start: 起始索引（包含）
        end: 结束索引（不包含）

    Returns:
        最大值的相对索引（相对于 start）
    """
    cdef int max_idx = start
    cdef double max_val = arr[start]
    cdef int i
    for i in range(start + 1, end):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_idx - start


cpdef double fast_round_price(double price, double tick_size) noexcept nogil:
    """快速价格取整

    将价格取整到 tick_size 的整数倍，确保最小值为 tick_size。

    Args:
        price: 原始价格
        tick_size: 最小价格变动单位

    Returns:
        取整后的价格，最小为 tick_size
    """
    return fmax(tick_size, c_round(price / tick_size) * tick_size)


cpdef void fast_copy_to_buffer(
    double[:] buffer,
    double[:] source,
    int dest_offset,
    int length,
) noexcept nogil:
    """快速复制数据到缓冲区

    将 source 数组的数据复制到 buffer 的指定偏移位置。

    Args:
        buffer: 目标缓冲区（memoryview）
        source: 源数据（memoryview）
        dest_offset: 目标偏移量
        length: 复制长度
    """
    cdef int i
    for i in range(length):
        buffer[dest_offset + i] = source[i]


cpdef double fast_clip(double value, double min_val, double max_val) noexcept nogil:
    """快速裁剪值到指定范围

    Args:
        value: 输入值
        min_val: 最小值
        max_val: 最大值

    Returns:
        裁剪后的值
    """
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value
