"""庄家 Agent 基类模块

本模块定义庄家 Agent 抽象基类，是多头庄家和空头庄家的父类。
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from src.bio.agents.base import ActionType, Agent
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType
from src.market.matching.trade import Trade
from src.market.orderbook.order import OrderSide

if TYPE_CHECKING:
    from src.market.matching.matching_engine import MatchingEngine
from src.market.market_state import NormalizedMarketState
from src.market.orderbook.orderbook import OrderBook


class WhaleBaseAgent(Agent):
    """庄家 Agent 抽象基类

    代表市场中拥有大量资金的参与者，初始资产 1000万，杠杆 10 倍。

    庄家可以选择 HOLD（不动）、限价单或市价单，同时只能挂一单。
    子类（多头庄家/空头庄家）决定具体的交易方向。

    Attributes:
        agent_id: Agent ID
        brain: NEAT 神经网络
        account: 交易账户
    """

    agent_id: int
    brain: Brain

    def __init__(
        self, agent_id: int, agent_type: AgentType, brain: Brain, config: AgentConfig
    ) -> None:
        """创建庄家 Agent

        调用父类构造函数，设置类型为指定的庄家类型。

        Args:
            agent_id: Agent ID
            agent_type: Agent 类型（BULL_WHALE 或 BEAR_WHALE）
            brain: NEAT 神经网络
            config: Agent 配置
        """
        super().__init__(agent_id, agent_type, brain, config)

    @abstractmethod
    def get_action_space(self) -> list[ActionType]:
        """获取庄家可用动作空间

        由子类实现，返回对应方向的动作。

        Returns:
            庄家可用的动作类型列表
        """
        pass

    @abstractmethod
    def _get_limit_action(self) -> ActionType:
        """获取限价单动作类型

        由子类实现，返回 PLACE_BID 或 PLACE_ASK。

        Returns:
            限价单动作类型
        """
        pass

    @abstractmethod
    def _get_market_action(self) -> ActionType:
        """获取市价单动作类型

        由子类实现，返回 MARKET_BUY 或 MARKET_SELL。

        Returns:
            市价单动作类型
        """
        pass

    @abstractmethod
    def _get_order_side(self) -> OrderSide:
        """获取订单方向

        由子类实现，返回 OrderSide.BUY 或 OrderSide.SELL。

        Returns:
            订单方向
        """
        pass

    @abstractmethod
    def _is_buy_direction(self) -> bool:
        """判断是否为买入方向

        由子类实现，返回 True（买入）或 False（卖出）。

        Returns:
            是否为买入方向
        """
        pass

    def decide(
        self, market_state: NormalizedMarketState, orderbook: OrderBook
    ) -> tuple[ActionType, dict[str, Any]]:
        """决策下一步动作

        神经网络输出变为 5 节点：
        - outputs[0]: HOLD 动作得分
        - outputs[1]: 限价单动作得分
        - outputs[2]: 市价单动作得分
        - outputs[3]: 价格偏移（-1 到 1，映射到 ±100 个 tick）
        - outputs[4]: 数量比例（-1 到 1，映射到 0.1-1.0）

        通过比较前三个得分决定是 HOLD、限价单还是市价单。

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿

        Returns:
            (动作类型, 动作参数字典)
        """
        # 如果已被强平，直接返回 HOLD
        if self.is_liquidated:
            return ActionType.HOLD, {}

        # 1. 观察市场，获取神经网络输入
        inputs = self.observe(market_state, orderbook)

        # 2. 神经网络前向传播
        outputs = self.brain.forward(inputs)

        # 3. 验证输出维度（需要 5 个值：HOLD得分 + 限价得分 + 市价得分 + 价格偏移 + 数量比例）
        if len(outputs) < 5:
            raise ValueError(f"神经网络输出维度不足，期望 5，实际 {len(outputs)}")

        # 4. 解析动作类型（比较 HOLD、限价单和市价单得分）
        hold_score = outputs[0]
        limit_score = outputs[1]
        market_score = outputs[2]

        # 找出得分最高的动作
        if hold_score >= limit_score and hold_score >= market_score:
            action = ActionType.HOLD
        elif limit_score >= market_score:
            action = self._get_limit_action()
        else:
            action = self._get_market_action()

        # 5. 解析参数
        # 输出[3]: 价格偏移（-1 到 1，映射到 ±100 个 tick）
        # 输出[4]: 数量比例（-1 到 1，映射到 0.1-1.0）
        price_offset_norm = max(-1.0, min(1.0, outputs[3]))
        quantity_ratio_norm = max(-1.0, min(1.0, outputs[4]))

        # 获取参考价格
        mid_price = market_state.mid_price
        if mid_price == 0:
            mid_price = 100.0

        # 映射数量比例到 [0.1, 1.0]
        quantity_ratio = 0.1 + (quantity_ratio_norm + 1) * 0.45

        # 获取 tick_size
        tick_size = market_state.tick_size if market_state.tick_size > 0 else 0.1

        # 获取方向
        is_buy = self._is_buy_direction()

        # 根据动作类型计算参数
        params: dict[str, Any] = {}

        if action == self._get_limit_action():
            # 限价单：价格由神经网络决定（相对 mid_price 的偏移）
            price_offset_ticks = price_offset_norm * 100  # ±100 ticks
            raw_price = mid_price + price_offset_ticks * tick_size
            # 舍入到 tick_size 的整数倍，避免浮点数精度问题
            # 确保价格至少为一个 tick_size，防止出现负价格或零价格
            params["price"] = max(tick_size, round(raw_price / tick_size) * tick_size)
            # 数量由神经网络决定
            params["quantity"] = self._calculate_order_quantity(
                mid_price, quantity_ratio, is_buy=is_buy
            )
        else:
            # 市价单：数量由神经网络决定
            params["quantity"] = self._calculate_order_quantity(
                mid_price, quantity_ratio, is_buy=is_buy
            )

        return action, params

    def execute_action(
        self,
        action: ActionType,
        params: dict[str, Any],
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """执行动作

        庄家特定实现：
        - HOLD 动作：不执行任何操作，不撤旧单
        - 限价单/市价单动作：先撤旧单再执行

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

        # HOLD 动作：不执行任何操作，直接返回空列表
        if action == ActionType.HOLD:
            return trades

        # 非 HOLD 动作先撤旧单
        if self.account.pending_order_id is not None:
            matching_engine.cancel_order(self.account.pending_order_id)
            self.account.pending_order_id = None  # 清除旧挂单ID

        # 获取订单方向
        order_side = self._get_order_side()

        if action == self._get_limit_action():
            trades = self._place_limit_order(
                order_side, params["price"], params["quantity"], matching_engine
            )
        elif action == self._get_market_action():
            trades = self._place_market_order(
                order_side, params["quantity"], matching_engine
            )

        return trades
