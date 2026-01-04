"""空头庄家 Agent 模块

本模块定义空头庄家 Agent，继承自 WhaleBaseAgent。
空头庄家只做卖出方向的交易（挂卖单或市价卖出）。
"""

from src.bio.agents.base import ActionType
from src.bio.agents.whale import WhaleBaseAgent
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType
from src.market.orderbook.order import OrderSide


class BearWhaleAgent(WhaleBaseAgent):
    """空头庄家 Agent

    只做卖出方向的交易，继承自 WhaleBaseAgent。

    动作空间：
    - PLACE_ASK: 挂卖单
    - MARKET_SELL: 市价卖出
    """

    def __init__(self, agent_id: int, brain: Brain, config: AgentConfig) -> None:
        """创建空头庄家 Agent

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
        """
        super().__init__(agent_id, AgentType.BEAR_WHALE, brain, config)

    def get_action_space(self) -> list[ActionType]:
        """获取空头庄家可用动作空间

        Returns:
            空头庄家可用的动作类型列表（卖出方向）
        """
        return [ActionType.PLACE_ASK, ActionType.MARKET_SELL]

    def _get_limit_action(self) -> ActionType:
        """获取限价单动作类型

        Returns:
            PLACE_ASK 动作
        """
        return ActionType.PLACE_ASK

    def _get_market_action(self) -> ActionType:
        """获取市价单动作类型

        Returns:
            MARKET_SELL 动作
        """
        return ActionType.MARKET_SELL

    def _get_order_side(self) -> OrderSide:
        """获取订单方向

        Returns:
            卖出方向
        """
        return OrderSide.SELL

    def _is_buy_direction(self) -> bool:
        """判断是否为买入方向

        Returns:
            False，空头庄家始终为卖出方向
        """
        return False
