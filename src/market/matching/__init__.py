"""
撮合引擎模块

提供订单撮合相关的类和函数。
优先使用 Cython + C++ 加速的 FastMatchingEngine，
通过 MatchingEngine 别名保持向后兼容。
"""

from src.market.matching.trade import Trade
from src.market.matching.fast_matching import FastTrade, FastMatchingEngine

# 别名：所有 `from src.market.matching import MatchingEngine` 的代码继续工作
MatchingEngine = FastMatchingEngine

__all__ = [
    "MatchingEngine",
    "FastMatchingEngine",
    "Trade",
    "FastTrade",
]
