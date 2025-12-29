"""
配置模块

定义系统的各类配置数据类。
"""

from dataclasses import dataclass


@dataclass
class MarketConfig:
    """
    市场配置

    定义交易市场的核心参数。

    Attributes:
        initial_price: 初始价格
        tick_size: 最小变动单位
        lot_size: 最小交易单位
        depth: 盘口深度（买卖各多少档）
    """

    initial_price: float
    tick_size: float
    lot_size: float
    depth: int
