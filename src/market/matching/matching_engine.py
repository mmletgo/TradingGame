"""
撮合引擎模块

负责订单撮合的核心逻辑，管理订单簿，处理订单提交、撤销和撮合。
"""

from src.config.config import MarketConfig
from src.core.event_engine.event_bus import EventBus
from src.core.event_engine.events import Event, EventType
from src.core.log_engine.logger import get_logger
from src.market.orderbook.orderbook import OrderBook


class MatchingEngine:
    """
    撮合引擎

    交易市场的核心组件，负责：
    - 管理订单簿
    - 处理订单提交和撤销
    - 执行撮合逻辑，生成成交记录
    - 发布成交事件

    Attributes:
        _event_bus: 事件总线
        _config: 市场配置
        _orderbook: 订单簿实例
        _logger: 日志器
        _next_trade_id: 下一个成交ID
    """

    def __init__(self, event_bus: EventBus, config: MarketConfig) -> None:
        """
        初始化撮合引擎

        Args:
            event_bus: 事件总线，用于发布成交事件
            config: 市场配置
        """
        self._event_bus = event_bus
        self._config = config
        self._orderbook = OrderBook(tick_size=config.tick_size)
        self._logger = get_logger(__name__)
        self._next_trade_id: int = 1
        self._fee_rates: dict[int, tuple[float, float]] = {}

        # 订阅订单事件
        event_bus.subscribe(EventType.ORDER_PLACED, self._handle_order_placed)
        event_bus.subscribe(
            EventType.ORDER_CANCELLED, self._handle_order_cancelled
        )

        self._logger.info("撮合引擎初始化完成")

    def _handle_order_placed(self, event: Event) -> None:
        """处理订单提交事件（占位）"""
        pass

    def _handle_order_cancelled(self, event: Event) -> None:
        """处理订单撤销事件（占位）"""
        pass

    def register_agent(self, agent_id: int, maker_rate: float, taker_rate: float) -> None:
        """
        注册/更新 Agent 的费率配置

        Args:
            agent_id: Agent ID
            maker_rate: 挂单费率（如 0.0002 表示万2）
            taker_rate: 吃单费率（如 0.0005 表示万5）
        """
        self._fee_rates[agent_id] = (maker_rate, taker_rate)

    def unregister_agent(self, agent_id: int) -> None:
        """
        移除 Agent 的费率配置（淘汰时调用）

        Args:
            agent_id: Agent ID
        """
        self._fee_rates.pop(agent_id, None)

    def calculate_fee(self, agent_id: int, amount: float, is_maker: bool) -> float:
        """
        计算手续费

        根据 Agent 类型和订单类型（挂单/吃单）计算手续费。

        费率配置（默认）：
        - 散户: 挂单万2 (0.0002)，吃单万5 (0.0005)
        - 庄家: 挂单0，吃单万1 (0.0001)
        - 做市商: 挂单0，吃单万1 (0.0001)

        Args:
            agent_id: Agent ID
            amount: 成交金额
            is_maker: True 表示挂单，False 表示吃单

        Returns:
            手续费金额

        Raises:
            ValueError: 如果 agent_id 未注册且金额为正
        """
        rates = self._fee_rates.get(agent_id)
        if rates is None:
            # 未注册的 Agent 使用默认散户费率
            rates = (0.0002, 0.0005)

        maker_rate, taker_rate = rates
        rate = maker_rate if is_maker else taker_rate
        return amount * rate
