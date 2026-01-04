"""
周期摆动型鲶鱼模块

实现周期摆动策略的鲶鱼，按固定周期交替买卖。
"""

from typing import TYPE_CHECKING

from src.config.config import CatfishConfig
from src.market.catfish.catfish_base import CatfishBase

if TYPE_CHECKING:
    from src.market.orderbook.orderbook import OrderBook


class CycleSwingCatfish(CatfishBase):
    """
    周期摆动型鲶鱼

    策略逻辑：
    - 每 half_cycle_length 个 tick 切换方向
    - 每 action_interval 个 tick 行动一次
    - 无视市场状态，按固定周期交替买卖

    Attributes:
        catfish_id: 鲶鱼ID
        config: 鲶鱼配置
        _current_direction: 当前方向（1=买，-1=卖）
    """

    def __init__(self, catfish_id: int, config: CatfishConfig) -> None:
        """
        初始化周期摆动型鲶鱼

        Args:
            catfish_id: 鲶鱼ID（应为负数）
            config: 鲶鱼配置
        """
        super().__init__(catfish_id, config)
        self._current_direction: int = 1  # 初始方向为买入

    def decide(
        self,
        orderbook: "OrderBook",
        tick: int,
        price_history: list[float],
    ) -> tuple[bool, int]:
        """
        决策是否行动以及行动方向

        按固定周期交替买卖，无视市场状态。

        Args:
            orderbook: 订单簿（本策略不使用）
            tick: 当前tick
            price_history: 历史价格列表（本策略不使用）

        Returns:
            (should_act, direction): 是否行动和方向（1=买，-1=卖）
        """
        half_cycle = self.config.half_cycle_length
        action_interval = self.config.action_interval

        # 计算当前处于哪个周期阶段
        # 完整周期 = 2 * half_cycle_length
        full_cycle = 2 * half_cycle
        position_in_cycle = tick % full_cycle

        # 根据周期位置确定方向
        # 前半周期买入，后半周期卖出
        if position_in_cycle < half_cycle:
            self._current_direction = 1  # 买入阶段
        else:
            self._current_direction = -1  # 卖出阶段

        # 检查是否到达行动间隔
        if tick % action_interval != 0:
            return False, 0

        return True, self._current_direction
