# cython: language_level=3
"""
批量决策 Cython 加速模块

提供完全 nogil 的 OpenMP 并行批量决策功能：
- batch_observe_*: 批量观察
- batch_forward: 批量神经网络前向传播
- batch_parse_*: 批量解析动作
- batch_decide_*: 完整批量决策流程
"""

try:
    from .batch_decide_openmp import (
        batch_decide_retail,
        batch_decide_full,
        batch_decide_market_maker,
    )
except ImportError:
    # Cython 模块未编译时提供占位
    pass
