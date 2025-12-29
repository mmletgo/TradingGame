"""
配置模块

定义系统的各类配置数据类。
"""

from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """Agent 类型枚举

    定义系统中三种 AI Agent 的类型。
    """

    RETAIL = "RETAIL"  # 散户
    WHALE = "WHALE"  # 庄家
    MARKET_MAKER = "MARKET_MAKER"  # 做市商


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


@dataclass
class AgentConfig:
    """
    Agent 配置

    定义特定类型 Agent 的交易参数。

    Attributes:
        count: 数量
        initial_balance: 初始资产
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率
        maker_fee_rate: 挂单费率
        taker_fee_rate: 吃单费率
    """

    count: int
    initial_balance: float
    leverage: float
    maintenance_margin_rate: float
    maker_fee_rate: float
    taker_fee_rate: float
