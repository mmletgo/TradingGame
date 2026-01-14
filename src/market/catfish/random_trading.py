"""
随机买卖型鲶鱼模块

实现随机买卖策略的鲶鱼，以随机概率进行买卖操作。
"""

import random
from typing import TYPE_CHECKING

from src.config.config import CatfishConfig
from src.market.catfish.catfish_base import CatfishBase

if TYPE_CHECKING:
    from src.market.orderbook.orderbook import OrderBook


class RandomTradingCatfish(CatfishBase):
    """
    随机买卖型鲶鱼

    策略逻辑：
    - 每个 tick 随机概率决定是否触发交易（使用 config.action_probability）
    - 交易方向随机决定（买入或卖出）

    Attributes:
        catfish_id: 鲶鱼ID
        config: 鲶鱼配置
    """

    def __init__(
        self,
        catfish_id: int,
        config: CatfishConfig,
        initial_balance: float = 0.0,
        leverage: float = 10.0,
        maintenance_margin_rate: float = 0.05,
    ) -> None:
        """
        初始化随机买卖型鲶鱼

        Args:
            catfish_id: 鲶鱼ID（应为负数）
            config: 鲶鱼配置
            initial_balance: 初始余额
            leverage: 杠杆倍数
            maintenance_margin_rate: 维持保证金率
        """
        super().__init__(
            catfish_id, config,
            initial_balance, leverage, maintenance_margin_rate
        )

    def decide(
        self,
        orderbook: "OrderBook",
        tick: int,
        price_history: list[float],
    ) -> tuple[bool, int]:
        """
        决策是否行动以及行动方向

        随机概率决定是否交易，交易方向也随机决定。

        Args:
            orderbook: 订单簿
            tick: 当前tick
            price_history: 历史价格列表

        Returns:
            (should_act, direction): 是否行动和方向（1=买，-1=卖）
        """
        # 随机概率判断是否行动
        if not self.can_act():
            return False, 0

        # 随机决定方向：买入或卖出
        direction = 1 if random.random() < 0.5 else -1

        return True, direction
