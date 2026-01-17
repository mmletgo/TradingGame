"""
逆势操作型鲶鱼模块

实现均值回归策略的鲶鱼，当价格偏离均线时反向操作。
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.config.config import CatfishConfig
from src.market.catfish.catfish_base import CatfishBase

if TYPE_CHECKING:
    from src.market.orderbook.orderbook import OrderBook


class MeanReversionCatfish(CatfishBase):
    """
    逆势操作型鲶鱼

    策略逻辑：
    - 维护 EMA 均值（周期 ma_period）
    - 当当前价格偏离 EMA 超过 deviation_threshold，反向操作
    - 有冷却时间 action_cooldown

    Attributes:
        catfish_id: 鲶鱼ID
        config: 鲶鱼配置
        _ema: 当前EMA值
        _ema_initialized: EMA是否已初始化
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
        初始化逆势操作型鲶鱼

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
        self._ema: float = 0.0
        self._ema_initialized: bool = False

    def _update_ema(self, price: float) -> None:
        """
        更新EMA值

        使用指数移动平均公式：EMA = alpha * price + (1 - alpha) * prev_EMA
        其中 alpha = 2 / (period + 1)

        Args:
            price: 当前价格
        """
        if not self._ema_initialized:
            self._ema = price
            self._ema_initialized = True
        else:
            alpha = 2.0 / (self.config.ma_period + 1)
            self._ema = alpha * price + (1 - alpha) * self._ema

    def decide(
        self,
        orderbook: "OrderBook",
        tick: int,
        price_history: Sequence[float],
    ) -> tuple[bool, int]:
        """
        决策是否行动以及行动方向

        计算当前价格与EMA的偏离程度，超过阈值时反向操作，随机概率决定是否行动。

        Args:
            orderbook: 订单簿
            tick: 当前tick
            price_history: 历史价格序列

        Returns:
            (should_act, direction): 是否行动和方向（1=买，-1=卖）
        """
        # 检查历史数据
        if len(price_history) == 0:
            return False, 0

        current_price = price_history[-1]

        # 更新EMA
        self._update_ema(current_price)

        # 检查是否有足够数据进行决策（需要至少ma_period个数据点初始化EMA）
        if len(price_history) < self.config.ma_period:
            return False, 0

        # 避免除以零
        if self._ema <= 0:
            return False, 0

        # 计算价格偏离率
        deviation = (current_price - self._ema) / self._ema

        # 检查是否超过阈值
        threshold = self.config.deviation_threshold
        if abs(deviation) < threshold:
            return False, 0

        # 随机概率判断是否行动
        if not self.can_act():
            return False, 0

        # 逆势操作：价格高于均线则卖出，价格低于均线则买入
        direction = -1 if deviation > 0 else 1

        return True, direction

    def reset(self) -> None:
        """
        重置EMA和账户状态

        在新的episode开始时调用。
        """
        super().reset()  # 调用父类 reset
        self._ema = 0.0
        self._ema_initialized = False
