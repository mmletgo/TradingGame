"""
趋势创造者鲶鱼模块

实现趋势创造策略的鲶鱼，每个 Episode 开始时随机选择方向，
整个 Episode 保持该方向持续操作，主动创造趋势。
"""

import random
from typing import TYPE_CHECKING

from src.config.config import CatfishConfig
from src.market.catfish.catfish_base import CatfishBase

if TYPE_CHECKING:
    from src.market.orderbook.orderbook import OrderBook


class TrendCreatorCatfish(CatfishBase):
    """
    趋势创造者鲶鱼

    策略逻辑：
    - Episode 开始时随机选择方向（买或卖）
    - 整个 Episode 保持该方向持续操作
    - 按冷却时间 action_cooldown 间隔操作

    Attributes:
        catfish_id: 鲶鱼ID
        config: 鲶鱼配置
        _current_direction: 当前方向（1=买，-1=卖）
    """

    def __init__(
        self,
        catfish_id: int,
        config: CatfishConfig,
        phase_offset: int = 0,
        initial_balance: float = 0.0,
        leverage: float = 10.0,
        maintenance_margin_rate: float = 0.05,
    ) -> None:
        """
        初始化趋势创造者鲶鱼

        Args:
            catfish_id: 鲶鱼ID（应为负数）
            config: 鲶鱼配置
            phase_offset: 相位偏移（用于错开触发时间）
            initial_balance: 初始余额
            leverage: 杠杆倍数
            maintenance_margin_rate: 维持保证金率
        """
        super().__init__(
            catfish_id, config, phase_offset,
            initial_balance, leverage, maintenance_margin_rate
        )
        # 初始化时随机选择方向
        self._current_direction: int = random.choice([1, -1])

    def decide(
        self,
        orderbook: "OrderBook",
        tick: int,
        price_history: list[float],
    ) -> tuple[bool, int]:
        """
        决策是否行动以及行动方向

        始终按当前方向操作，仅检查冷却时间。

        Args:
            orderbook: 订单簿
            tick: 当前tick
            price_history: 历史价格列表（未使用）

        Returns:
            (should_act, direction): 是否行动和方向（1=买，-1=卖）
        """
        # 检查冷却时间
        if not self.can_act(tick):
            return False, 0

        # 返回当前方向
        return True, self._current_direction

    def reset(self) -> None:
        """
        重置鲶鱼状态

        每个 Episode 开始时调用，重置账户并重新随机选择方向。
        """
        super().reset()
        # 重新随机选择方向
        self._current_direction = random.choice([1, -1])


# 保持向后兼容的别名
TrendFollowingCatfish = TrendCreatorCatfish
