"""多头庄家 Agent 模块

本模块定义多头庄家 Agent，继承自 WhaleBaseAgent。
多头庄家只做买入方向的交易（挂买单或市价买入）。
"""

from src.bio.agents.base import ActionType
from src.bio.agents.whale import WhaleBaseAgent
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType
from src.market.orderbook.order import OrderSide


class BullWhaleAgent(WhaleBaseAgent):
    """多头庄家 Agent

    只做买入方向的交易，继承自 WhaleBaseAgent。

    动作空间：
    - PLACE_BID: 挂买单
    - MARKET_BUY: 市价买入
    """

    def __init__(self, agent_id: int, brain: Brain, config: AgentConfig) -> None:
        """创建多头庄家 Agent

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
        """
        super().__init__(agent_id, AgentType.BULL_WHALE, brain, config)

    def get_action_space(self) -> list[ActionType]:
        """获取多头庄家可用动作空间

        Returns:
            多头庄家可用的动作类型列表（买入方向）
        """
        return [ActionType.PLACE_BID, ActionType.MARKET_BUY]

    def _get_limit_action(self) -> ActionType:
        """获取限价单动作类型

        Returns:
            PLACE_BID 动作
        """
        return ActionType.PLACE_BID

    def _get_market_action(self) -> ActionType:
        """获取市价单动作类型

        Returns:
            MARKET_BUY 动作
        """
        return ActionType.MARKET_BUY

    def _get_order_side(self) -> OrderSide:
        """获取订单方向

        Returns:
            买入方向
        """
        return OrderSide.BUY

    def _is_buy_direction(self) -> bool:
        """判断是否为买入方向

        Returns:
            True，多头庄家始终为买入方向
        """
        return True
