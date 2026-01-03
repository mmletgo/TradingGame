"""庄家 Agent 模块

本模块定义庄家 Agent 类，继承自 Agent 基类。
"""

from typing import TYPE_CHECKING, Any

from src.bio.agents.base import ActionType, Agent
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType
from src.market.matching.trade import Trade
from src.market.orderbook.order import OrderSide

if TYPE_CHECKING:
    from src.market.matching.matching_engine import MatchingEngine


class WhaleAgent(Agent):
    """庄家 Agent

    代表市场中拥有大量资金的参与者，初始资产 1000万，杠杆 10 倍。

    庄家"绝不不动"，必须持续参与市场，同时只能挂一单。

    Attributes:
        agent_id: Agent ID
        brain: NEAT 神经网络
        account: 交易账户
    """

    agent_id: int
    brain: Brain

    def __init__(
        self, agent_id: int, brain: Brain, config: AgentConfig
    ) -> None:
        """创建庄家 Agent

        调用父类构造函数，设置类型为 WHALE。

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
        """
        super().__init__(agent_id, AgentType.WHALE, brain, config)

    def get_action_space(self) -> list[ActionType]:
        """获取庄家可用动作空间

        庄家"绝不不动"，不能选择 HOLD 动作，也不能单纯撤单。
        可选动作：
        - PLACE_BID: 挂买单
        - PLACE_ASK: 挂卖单
        - MARKET_BUY: 市价买入
        - MARKET_SELL: 市价卖出

        Returns:
            庄家可用的动作类型列表（不包含 HOLD、CANCEL 和 CLEAR_POSITION）
        """
        return [
            ActionType.PLACE_BID,
            ActionType.PLACE_ASK,
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

        庄家特定实现：所有动作都会先撤旧单再执行。

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

        # 庄家所有动作都先撤旧单
        if self.account.pending_order_id is not None:
            matching_engine.cancel_order(self.account.pending_order_id)
            self.account.pending_order_id = None  # 清除旧挂单ID

        if action == ActionType.PLACE_BID:
            trades = self._place_limit_order(
                OrderSide.BUY, params["price"], params["quantity"], matching_engine
            )
        elif action == ActionType.PLACE_ASK:
            trades = self._place_limit_order(
                OrderSide.SELL, params["price"], params["quantity"], matching_engine
            )
        elif action == ActionType.MARKET_BUY:
            trades = self._place_market_order(
                OrderSide.BUY, params["quantity"], matching_engine
            )
        elif action == ActionType.MARKET_SELL:
            trades = self._place_market_order(
                OrderSide.SELL, params["quantity"], matching_engine
            )
        # 庄家没有 HOLD、CANCEL、CLEAR_POSITION 动作

        return trades
