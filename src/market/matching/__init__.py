"""
撮合引擎模块

提供订单撮合相关的类和函数。
"""

from src.market.matching.matching_engine import MatchingEngine
from src.market.matching.trade import Trade
from src.market.matching.fast_matching import FastTrade, fast_match_orders

# 尝试导入 Cython 加速的撮合引擎
try:
    from src.market.matching.fast_matching import FastMatchingEngine
except ImportError:
    FastMatchingEngine = None

__all__ = [
    "MatchingEngine",
    "FastMatchingEngine",
    "Trade",
    "FastTrade",
    "fast_match_orders",
]
