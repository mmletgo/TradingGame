"""
趋势追踪型鲶鱼模块

实现趋势追踪策略的鲶鱼，根据历史价格变化率顺势下单。
"""

from typing import TYPE_CHECKING

from src.config.config import CatfishConfig
from src.market.catfish.catfish_base import CatfishBase

if TYPE_CHECKING:
    from src.market.orderbook.orderbook import OrderBook


class TrendFollowingCatfish(CatfishBase):
    """
    趋势追踪型鲶鱼

    策略逻辑：
    - 回看 lookback_period 个 tick 的价格
    - 计算价格变化率
    - 若变化率超过 trend_threshold，顺势下单
    - 有冷却时间 action_cooldown

    Attributes:
        catfish_id: 鲶鱼ID
        config: 鲶鱼配置
    """

    def __init__(self, catfish_id: int, config: CatfishConfig) -> None:
        """
        初始化趋势追踪型鲶鱼

        Args:
            catfish_id: 鲶鱼ID（应为负数）
            config: 鲶鱼配置
        """
        super().__init__(catfish_id, config)

    def decide(
        self,
        orderbook: "OrderBook",
        tick: int,
        price_history: list[float],
    ) -> tuple[bool, int]:
        """
        决策是否行动以及行动方向

        根据历史价格计算变化率，若超过阈值则顺势下单。

        Args:
            orderbook: 订单簿
            tick: 当前tick
            price_history: 历史价格列表

        Returns:
            (should_act, direction): 是否行动和方向（1=买，-1=卖）
        """
        # 检查冷却时间
        if not self.can_act(tick):
            return False, 0

        # 检查历史数据是否足够
        lookback = self.config.lookback_period
        if len(price_history) < lookback:
            return False, 0

        # 获取回看周期的起始价格和当前价格
        start_price = price_history[-lookback]
        current_price = price_history[-1]

        # 避免除以零
        if start_price <= 0:
            return False, 0

        # 计算价格变化率
        change_rate = (current_price - start_price) / start_price

        # 检查是否超过阈值
        threshold = self.config.trend_threshold
        if abs(change_rate) < threshold:
            return False, 0

        # 顺势下单：价格上涨则买入，价格下跌则卖出
        direction = 1 if change_rate > 0 else -1

        return True, direction
