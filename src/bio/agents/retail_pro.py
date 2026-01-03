"""高级散户 Agent 模块

本模块定义高级散户 Agent 类，继承自 Agent 基类。
与普通散户不同，高级散户可以看到完整的100档订单簿和100笔成交。
"""

from typing import TYPE_CHECKING, Any

from src.bio.agents.base import ActionType
from src.bio.agents.base import Agent
from src.config.config import AgentConfig, AgentType
from src.bio.brain.brain import Brain
from src.market.matching.trade import Trade
from src.market.orderbook.order import OrderSide

if TYPE_CHECKING:
    from src.market.matching.matching_engine import MatchingEngine


class RetailProAgent(Agent):
    """高级散户 Agent

    代表市场中具有更多信息优势的散户交易者。
    与普通散户相比，高级散户可以看到完整的100档订单簿和100笔成交，
    但仍遵循散户的交易规则（同时只能挂一单）。

    Attributes:
        agent_id: Agent ID
        brain: NEAT 神经网络
        account: 交易账户
    """

    agent_id: int
    brain: Brain

    def __init__(self, agent_id: int, brain: Brain, config: AgentConfig) -> None:
        """创建高级散户 Agent

        调用父类构造函数，设置类型为 RETAIL_PRO。
        使用父类默认的输入缓冲区大小（607），因为高级散户可以看到完整的市场数据。

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
        """
        super().__init__(agent_id, AgentType.RETAIL_PRO, brain, config)

    def get_action_space(self) -> list[ActionType]:
        """获取高级散户可用动作空间

        高级散户可以执行的动作包括：不动、挂买单、挂卖单、撤单、市价买入、市价卖出。

        Returns:
            可用动作类型列表
        """
        return [
            ActionType.HOLD,
            ActionType.PLACE_BID,
            ActionType.PLACE_ASK,
            ActionType.CANCEL,
            ActionType.MARKET_BUY,
            ActionType.MARKET_SELL,
        ]

    def execute_action(
        self,
        action: ActionType,
        params: dict[str, Any],
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """执行动作

        高级散户特定实现：PLACE_BID/PLACE_ASK 会先撤旧单再挂新单。

        Args:
            action: 动作类型
            params: 动作参数字典
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        if self.is_liquidated:
            return []

        trades: list[Trade] = []

        if action == ActionType.PLACE_BID or action == ActionType.PLACE_ASK:
            # 高级散户特定：先撤旧单再挂新单
            if self.account.pending_order_id is not None:
                matching_engine.cancel_order(self.account.pending_order_id)
                self.account.pending_order_id = None  # 清除旧挂单ID
            side = OrderSide.BUY if action == ActionType.PLACE_BID else OrderSide.SELL
            trades = self._place_limit_order(
                side, params["price"], params["quantity"], matching_engine
            )
        else:
            # 其他动作使用父类实现
            trades = super().execute_action(action, params, matching_engine)

        return trades
