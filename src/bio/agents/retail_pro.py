"""高级散户 Agent 模块

本模块定义高级散户 Agent 类，继承自 Agent 基类。
与普通散户不同，高级散户可以看到完整的100档订单簿和100笔成交。
高级散户实现了散户动作空间的 decide 方法（9个输出，7种动作）。
"""

from typing import TYPE_CHECKING, Any

from src.bio.agents.base import (
    ActionType,
    Agent,
    fast_argmax,
    fast_round_price,
    fast_clip,
)
from src.config.config import AgentConfig, AgentType
from src.bio.brain.brain import Brain
from src.market.market_state import NormalizedMarketState
from src.market.matching.trade import Trade
from src.market.orderbook.orderbook import OrderBook

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

    def decide(
        self, market_state: NormalizedMarketState, orderbook: OrderBook
    ) -> tuple[ActionType, dict[str, Any]]:
        """决策下一步动作（接收预计算的市场状态）

        观察市场状态，通过神经网络前向传播，解析输出为动作类型和参数。
        此方法实现了散户/高级散户通用的决策逻辑（8个输出节点，6种动作）。

        神经网络输出结构：
        - 输出[0-5]: 6 种动作类型的得分（选择最大值）
        - 输出[6]: 价格偏移（归一化值，-1 到 1，相对于 mid_price）
        - 输出[7]: 数量比例（归一化值，0 到 1，相对于可用购买力）

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿

        Returns:
            (动作类型, 动作参数字典)
            - PLACE_BID/PLACE_ASK: {"price": float, "quantity": float}
            - MARKET_BUY/MARKET_SELL: {"quantity": float}
            - CANCEL/HOLD: {}
        """
        # 如果已被强平，直接返回 HOLD
        if self.is_liquidated:
            return ActionType.HOLD, {}

        # 1. 观察市场，获取神经网络输入
        inputs = self.observe(market_state, orderbook)

        # 2. 神经网络前向传播
        outputs = self.brain.forward(inputs)

        # 3. 验证输出维度（至少需要 8 个值：6个动作 + 价格偏移 + 数量比例）
        if len(outputs) < 8:
            raise ValueError(f"神经网络输出维度不足，期望 8，实际 {len(outputs)}")

        # 4. 解析动作类型（选择前 6 个输出中值最大的索引）
        action_idx = fast_argmax(outputs, 0, 6)
        action = ActionType(action_idx)

        # 5. 解析参数（由神经网络决定）
        # 输出[6]: 价格偏移（-1 到 1，映射到 ±100 个 tick）
        # 输出[7]: 数量比例（-1 到 1，映射到 0.1-1.0 的购买力比例）
        price_offset_norm = fast_clip(outputs[6], -1.0, 1.0)
        quantity_ratio_norm = fast_clip(outputs[7], -1.0, 1.0)

        # 获取参考价格
        mid_price = market_state.mid_price
        if mid_price == 0:
            mid_price = 100.0

        # 映射数量比例到 [0, 1.0]
        quantity_ratio = (quantity_ratio_norm + 1) * 0.5  # -1→0, 0→0.5, 1→1.0

        # 获取 tick_size
        tick_size = market_state.tick_size if market_state.tick_size > 0 else 0.01

        # 根据动作类型计算参数
        params: dict[str, Any] = {}

        if action == ActionType.PLACE_BID:
            # 挂买单：价格由神经网络决定（相对 mid_price 的偏移）
            price_offset_ticks = price_offset_norm * 100  # ±100 ticks
            raw_price = mid_price + price_offset_ticks * tick_size
            # 舍入到 tick_size 的整数倍，避免浮点数精度问题
            # 确保价格至少为一个 tick_size，防止出现负价格或零价格
            # 使用统一接口（Cython 或回退）
            params["price"] = fast_round_price(raw_price, tick_size)
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
            # 使用统一接口（Cython 或回退）
            params["price"] = fast_round_price(raw_price, tick_size)
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
            # 市价卖出：数量由神经网络决定（卖出方向，限制总持仓）
            # 与 MARKET_BUY 完全对称：无论当前持仓状态，统一使用 _calculate_order_quantity
            # 可以平多仓+开空仓，或直接开空仓
            params["quantity"] = self._calculate_order_quantity(
                mid_price, quantity_ratio, is_buy=False
            )

        elif action == ActionType.CANCEL:
            # 撤单：无参数
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

        # PLACE_BID/PLACE_ASK 需要先撤旧单再挂新单
        if action == ActionType.PLACE_BID or action == ActionType.PLACE_ASK:
            if self.account.pending_order_id is not None:
                matching_engine.cancel_order(self.account.pending_order_id)
                self.account.pending_order_id = None  # 清除旧挂单ID

        # 所有动作都委托给父类处理
        return super().execute_action(action, params, matching_engine)
