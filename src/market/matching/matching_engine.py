"""
撮合引擎模块（纯 Python 兼容层）

保留 MatchingEngine 接口，内部委托给 FastMatchingEngine (Cython + C++)。
非热路径代码（强平、ADL、演示等）可直接导入本模块。
"""

from src.config.config import MarketConfig
from src.market.orderbook.order import Order, OrderType
from src.market.matching.trade import Trade
from src.market.matching.fast_matching import FastMatchingEngine, FastTrade

from src.core.log_engine.logger import get_logger


class MatchingEngine:
    """
    撮合引擎（纯 Python 兼容层）

    内部委托给 FastMatchingEngine (Cython + C++)，
    保留与旧代码相同的属性和方法签名。

    Attributes:
        _config: 市场配置
        _orderbook: 订单簿实例
        _logger: 日志器
        _next_trade_id: 下一个成交ID
    """

    def __init__(self, config: MarketConfig) -> None:
        """
        初始化撮合引擎

        Args:
            config: 市场配置
        """
        self._config = config
        self._fast = FastMatchingEngine(config)
        self._logger = get_logger(__name__)
        self._logger.info("撮合引擎初始化完成")

    @property
    def _orderbook(self):
        """获取订单簿"""
        return self._fast._orderbook

    @property
    def _next_trade_id(self) -> int:
        """获取下一个成交ID"""
        return self._fast._next_trade_id

    @_next_trade_id.setter
    def _next_trade_id(self, value: int) -> None:
        self._fast._next_trade_id = value

    @property
    def _fee_rates(self) -> dict:
        """获取费率配置"""
        return self._fast._fee_rates

    @property
    def orderbook(self):
        """获取订单簿"""
        return self._fast._orderbook

    @property
    def _tick_size(self) -> float:
        return self._fast._tick_size

    def register_agent(
        self, agent_id: int, maker_rate: float, taker_rate: float
    ) -> None:
        """
        注册/更新 Agent 的费率配置

        Args:
            agent_id: Agent ID
            maker_rate: 挂单费率
            taker_rate: 吃单费率
        """
        self._fast.register_agent(agent_id, maker_rate, taker_rate)

    def calculate_fee(self, agent_id: int, amount: float, is_maker: bool) -> float:
        """
        计算手续费

        Args:
            agent_id: Agent ID
            amount: 成交金额
            is_maker: True 表示挂单，False 表示吃单

        Returns:
            手续费金额
        """
        return self._fast.calculate_fee(agent_id, amount, is_maker)

    def match_limit_order(self, order: Order) -> list[FastTrade]:
        """限价单撮合"""
        return self._fast.process_order(order)

    def match_market_order(self, order: Order) -> list[FastTrade]:
        """市价单撮合"""
        return self._fast.process_order(order)

    def process_order(self, order: Order) -> list[FastTrade]:
        """处理订单，执行撮合"""
        return self._fast.process_order(order)

    def process_order_raw(self, order_id: int, agent_id: int,
                          side: int, order_type: int,
                          price: float, quantity: int) -> list[FastTrade]:
        """不创建 Order 对象的快速撮合"""
        return self._fast.process_order_raw(order_id, agent_id, side, order_type, price, quantity)

    def cancel_order(self, order_id: int) -> bool:
        """撤单"""
        return self._fast.cancel_order(order_id)
