"""庄家 Agent 模块

本模块定义庄家 Agent 类，继承自 Agent 基类。
庄家拥有大量资金（初始资产 1000万），使用 10 倍杠杆。
庄家可以执行散户相同的动作空间（挂单、撤单、吃单、清仓）。
"""

from typing import TYPE_CHECKING, Any

from src.bio.agents.base import ActionType, Agent
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType
from src.market.matching.trade import Trade
from src.market.market_state import NormalizedMarketState
from src.market.orderbook.order import OrderSide
from src.market.orderbook.orderbook import OrderBook

if TYPE_CHECKING:
    from src.market.matching.matching_engine import MatchingEngine


class WhaleAgent(Agent):
    """庄家 Agent

    代表市场中拥有大量资金的参与者，初始资产 1000万，杠杆 10 倍。

    庄家可以执行与散户相同的动作空间（挂单、撤单、吃单、清仓），
    但拥有更大的资金量和更全面的市场信息（可观察完整的 100 档订单簿）。

    Attributes:
        agent_id: Agent ID
        brain: NEAT 神经网络
        account: 交易账户
    """

    agent_id: int
    brain: Brain

    def __init__(self, agent_id: int, brain: Brain, config: AgentConfig) -> None:
        """创建庄家 Agent

        调用父类构造函数，设置类型为 WHALE。
        使用父类默认的输入缓冲区大小（607），因为庄家可以看到完整的市场数据。

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
        """
        super().__init__(agent_id, AgentType.WHALE, brain, config)

    def get_action_space(self) -> list[ActionType]:
        """获取庄家可用动作空间

        庄家可以执行的动作包括：不动、挂买单、挂卖单、撤单、市价买入、市价卖出、清仓。

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
            ActionType.CLEAR_POSITION,
        ]

    def decide(
        self, market_state: NormalizedMarketState, orderbook: OrderBook
    ) -> tuple[ActionType, dict[str, Any]]:
        """决策下一步动作（接收预计算的市场状态）

        观察市场状态，通过神经网络前向传播，解析输出为动作类型和参数。
        此方法实现了庄家的决策逻辑（9个输出节点，7种动作）。

        神经网络输出结构：
        - 输出[0-6]: 7 种动作类型的得分（选择最大值）
        - 输出[7]: 价格偏移（归一化值，-1 到 1，相对于 mid_price）
        - 输出[8]: 数量比例（归一化值，0 到 1，相对于可用购买力）

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿

        Returns:
            (动作类型, 动作参数字典)
            - PLACE_BID/PLACE_ASK: {"price": float, "quantity": float}
            - MARKET_BUY/MARKET_SELL: {"quantity": float}
            - CANCEL/CLEAR_POSITION/HOLD: {}
        """
        # 如果已被强平，直接返回 HOLD
        if self.is_liquidated:
            return ActionType.HOLD, {}

        # 1. 观察市场，获取神经网络输入
        inputs = self.observe(market_state, orderbook)

        # 2. 神经网络前向传播
        outputs = self.brain.forward(inputs)

        # 3. 验证输出维度（至少需要 9 个值：7个动作 + 价格偏移 + 数量比例）
        if len(outputs) < 9:
            raise ValueError(f"神经网络输出维度不足，期望 9，实际 {len(outputs)}")

        # 4. 解析动作类型（选择前 7 个输出中值最大的索引）
        action_idx = int(max(range(7), key=lambda i: outputs[i]))
        action = ActionType(action_idx)

        # 5. 解析参数（由神经网络决定）
        # 输出[7]: 价格偏移（-1 到 1，映射到 ±100 个 tick）
        # 输出[8]: 数量比例（-1 到 1，映射到 0.1-1.0 的购买力比例）
        price_offset_norm = max(-1.0, min(1.0, outputs[7]))  # 限制在 [-1, 1]
        quantity_ratio_norm = max(-1.0, min(1.0, outputs[8]))  # 限制在 [-1, 1]

        # 获取参考价格
        mid_price = market_state.mid_price
        if mid_price == 0:
            mid_price = 100.0

        # 映射数量比例到 [0.1, 1.0]
        quantity_ratio = 0.1 + (quantity_ratio_norm + 1) * 0.45  # -1→0.1, 0→0.55, 1→1.0

        # 获取 tick_size
        tick_size = market_state.tick_size if market_state.tick_size > 0 else 0.1

        # 根据动作类型计算参数
        params: dict[str, Any] = {}

        if action == ActionType.PLACE_BID:
            # 挂买单：价格由神经网络决定（相对 mid_price 的偏移）
            price_offset_ticks = price_offset_norm * 100  # ±100 ticks
            raw_price = mid_price + price_offset_ticks * tick_size
            # 舍入到 tick_size 的整数倍，避免浮点数精度问题
            # 确保价格至少为一个 tick_size，防止出现负价格或零价格
            params["price"] = max(tick_size, round(raw_price / tick_size) * tick_size)
            # 数量由神经网络决定（买入方向，限制总持仓）
            params["quantity"] = self._calculate_order_quantity(
                mid_price, quantity_ratio, is_buy=True
            )

        elif action == ActionType.PLACE_ASK:
            # 挂卖单：价格由神经网络决定（相对 mid_price 的偏移）
            price_offset_ticks = price_offset_norm * 100  # ±100 ticks
            raw_price = mid_price + price_offset_ticks * tick_size
            # 舍入到 tick_size 的整数倍，避免浮点数精度问题
            # 确保价格至少为一个 tick_size，防止出现负价格或零价格
            params["price"] = max(tick_size, round(raw_price / tick_size) * tick_size)
            # 数量由神经网络决定（卖出方向，限制总持仓）
            params["quantity"] = self._calculate_order_quantity(
                mid_price, quantity_ratio, is_buy=False
            )

        elif action == ActionType.MARKET_BUY:
            # 市价买入：数量由神经网络决定（买入方向，限制总持仓）
            params["quantity"] = self._calculate_order_quantity(
                mid_price, quantity_ratio, is_buy=True
            )

        elif action == ActionType.MARKET_SELL:
            # 市价卖出：数量由神经网络决定
            position_qty = self.account.position.quantity
            if position_qty > 0:
                # 有多仓时卖出（卖出比例由神经网络决定）
                # 取整并确保至少卖出1个单位
                sell_qty = max(1, int(position_qty * quantity_ratio))
                # 但不能超过持仓量
                params["quantity"] = min(sell_qty, int(position_qty))
            else:
                # 空仓或无持仓，开空仓（卖出方向，限制总持仓）
                params["quantity"] = self._calculate_order_quantity(
                    mid_price, quantity_ratio, is_buy=False
                )

        elif action == ActionType.CANCEL:
            # 撤单：无参数
            pass

        elif action == ActionType.CLEAR_POSITION:
            # 清仓：无参数（由调用方处理）
            pass

        elif action == ActionType.HOLD:
            # 不动：无参数
            pass

        return action, params

    def execute_action(
        self,
        action: ActionType,
        params: dict[str, Any],
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """执行动作

        庄家特定实现：PLACE_BID/PLACE_ASK 会先撤旧单再挂新单。
        其他动作（包括 CLEAR_POSITION）使用父类实现。

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
            # 庄家特定：先撤旧单再挂新单
            if self.account.pending_order_id is not None:
                matching_engine.cancel_order(self.account.pending_order_id)
                self.account.pending_order_id = None  # 清除旧挂单ID
            side = OrderSide.BUY if action == ActionType.PLACE_BID else OrderSide.SELL
            trades = self._place_limit_order(
                side, params["price"], params["quantity"], matching_engine
            )
        else:
            # 其他动作（包括 CLEAR_POSITION）使用父类实现
            # 父类的 _handle_clear_position 已经包含先撤单再平仓的逻辑
            trades = super().execute_action(action, params, matching_engine)

        return trades
