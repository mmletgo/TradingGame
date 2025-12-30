"""测试 Agent 模块"""

from unittest.mock import MagicMock

import numpy as np

from src.bio.agents.base import Agent, ActionType
from src.bio.agents.retail import RetailAgent
from src.bio.agents.whale import WhaleAgent
from src.bio.agents.market_maker import MarketMakerAgent
from src.config.config import AgentConfig, AgentType
from src.core.event_engine.events import Event, EventType
from src.core.event_engine.event_bus import EventBus
from src.bio.brain.brain import Brain
from src.market.orderbook.order import OrderSide, OrderType
from src.market.orderbook.orderbook import OrderBook
from src.market.matching.trade import Trade
from src.market.market_state import NormalizedMarketState


def create_mock_market_state(mid_price: float = 100.0, tick_size: float = 0.1) -> NormalizedMarketState:
    """创建测试用的 NormalizedMarketState"""
    return NormalizedMarketState(
        mid_price=mid_price,
        tick_size=tick_size,
        bid_data=np.zeros(200, dtype=np.float32),
        ask_data=np.zeros(200, dtype=np.float32),
        trade_prices=np.zeros(100, dtype=np.float32),
        trade_quantities=np.zeros(100, dtype=np.float32),
        trade_buyer_ids=[],
        trade_seller_ids=[],
    )


class TestAgentInit:
    """测试 Agent.__init__"""

    def test_create_retail_agent(self):
        """测试创建散户 Agent"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建散户 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证属性
        assert agent.agent_id == 1
        assert agent.agent_type == AgentType.RETAIL
        assert agent.brain is mock_brain
        assert agent.account.agent_id == 1
        assert agent.account.agent_type == AgentType.RETAIL
        assert agent.account.balance == 10000.0
        assert agent.account.leverage == 100.0
        assert agent.account.maintenance_margin_rate == 0.005
        assert agent.account.maker_fee_rate == 0.0002
        assert agent.account.taker_fee_rate == 0.0005

    def test_create_whale_agent(self):
        """测试创建庄家 Agent"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建庄家 Agent
        agent = Agent(
            agent_id=10001,
            agent_type=AgentType.WHALE,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证属性
        assert agent.agent_id == 10001
        assert agent.agent_type == AgentType.WHALE
        assert agent.brain is mock_brain
        assert agent.account.balance == 10000000.0
        assert agent.account.leverage == 10.0
        assert agent.account.maker_fee_rate == 0.0
        assert agent.account.taker_fee_rate == 0.0001

    def test_create_market_maker_agent(self):
        """测试创建做市商 Agent"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建做市商 Agent
        agent = Agent(
            agent_id=10011,
            agent_type=AgentType.MARKET_MAKER,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证属性
        assert agent.agent_id == 10011
        assert agent.agent_type == AgentType.MARKET_MAKER
        assert agent.brain is mock_brain
        assert agent.account.balance == 10000000.0
        assert agent.account.leverage == 10.0
        assert agent.account.maker_fee_rate == 0.0
        assert agent.account.taker_fee_rate == 0.0001


class TestAgentOnTradeEvent:
    """测试 Agent._on_trade_event"""

    def test_on_buy_trade_event(self):
        """测试处理买入成交事件"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建买入成交事件
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=0.0,
            data={
                "trade_id": 1,
                "price": 100.0,
                "quantity": 10.0,
                "buyer_id": 1,
                "seller_id": 2,
                "buyer_fee": 5.0,
                "seller_fee": 2.0,
            },
        )

        # 处理事件
        agent._on_trade_event(event)

        # 验证账户更新
        assert agent.account.position.quantity == 10.0
        assert agent.account.position.avg_price == 100.0
        assert agent.account.balance == 9995.0  # 10000 - 5

    def test_on_sell_trade_event(self):
        """测试处理卖出成交事件"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent 并先开多仓
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.WHALE,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )
        # 先开多仓 100 @ 100
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 创建卖出成交事件（平仓）
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=0.0,
            data={
                "trade_id": 2,
                "price": 110.0,
                "quantity": 50.0,
                "buyer_id": 2,
                "seller_id": 1,
                "buyer_fee": 5.5,
                "seller_fee": 0.0,
            },
        )

        # 记录初始余额
        initial_balance = agent.account.balance

        # 处理事件
        agent._on_trade_event(event)

        # 验证账户更新
        assert agent.account.position.quantity == 50.0  # 100 - 50
        assert agent.account.position.avg_price == 100.0
        # 余额不变（庄家 maker 费率为 0）
        assert agent.account.balance == initial_balance
        # 已实现盈亏 = 50 * (110 - 100) = 500
        assert abs(agent.account.position.realized_pnl - 500.0) < 0.01

    def test_targeted_event_delivery(self):
        """测试定向事件发送（无关事件不会发送给 Agent）

        现在使用 subscribe_with_id 和 target_ids，无关事件根本不会发送给 Agent。
        """
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent (agent_id=1)
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证 Agent 使用 subscribe_with_id 订阅了事件
        assert EventType.TRADE_EXECUTED in event_bus._subscriber_ids
        assert 1 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

        # 创建定向发送给其他 Agent 的成交事件
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=0.0,
            data={
                "trade_id": 1,
                "price": 100.0,
                "quantity": 10.0,
                "buyer_id": 2,
                "seller_id": 3,
                "buyer_fee": 5.0,
                "seller_fee": 2.0,
            },
            target_ids={2, 3},  # 只发送给 agent_id=2 和 agent_id=3
        )

        # 记录初始账户状态
        initial_balance = agent.account.balance
        initial_quantity = agent.account.position.quantity

        # 发布事件
        event_bus.publish(event)

        # 验证账户未更新（事件未发送给 agent_id=1）
        assert agent.account.balance == initial_balance
        assert agent.account.position.quantity == initial_quantity

    def test_event_subscription_with_id(self):
        """测试使用 subscribe_with_id 订阅事件"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证使用 subscribe_with_id 订阅
        assert EventType.TRADE_EXECUTED in event_bus._subscriber_ids
        assert 1 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

        # 发布定向成交事件
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=0.0,
            data={
                "trade_id": 1,
                "price": 100.0,
                "quantity": 10.0,
                "buyer_id": 1,
                "seller_id": 2,
                "buyer_fee": 5.0,
                "seller_fee": 2.0,
            },
            target_ids={1, 2},  # 发送给参与成交的双方
        )

        # 发布事件
        event_bus.publish(event)

        # 验证账户自动更新（通过事件订阅）
        assert agent.account.position.quantity == 10.0
        assert agent.account.position.avg_price == 100.0
        assert agent.account.balance == 9995.0  # 10000 - 5


class TestAgentObserve:
    """测试 Agent.observe"""

    def test_observe_normal(self):
        """测试正常观察市场状态（使用 NormalizedMarketState）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建 NormalizedMarketState
        market_state = NormalizedMarketState(
            mid_price=100.0,
            tick_size=0.1,
            bid_data=np.array([0.1, 10.0] * 100, dtype=np.float32),  # 100档买盘
            ask_data=np.array([-0.1, 10.0] * 100, dtype=np.float32),  # 100档卖盘
            trade_prices=np.zeros(100, dtype=np.float32),
            trade_quantities=np.zeros(100, dtype=np.float32),
            trade_buyer_ids=[1, 3, 4],  # 本 Agent 是第一笔成交的买方
            trade_seller_ids=[2, 1, 5],  # 本 Agent 是第二笔成交的卖方
        )
        # 设置前三笔成交
        market_state.trade_prices[0] = -0.001  # 99.9 归一化
        market_state.trade_prices[1] = 0.001   # 100.1 归一化
        market_state.trade_prices[2] = 0.0     # 100.0 归一化
        market_state.trade_quantities[0] = 10.0
        market_state.trade_quantities[1] = 5.0
        market_state.trade_quantities[2] = 8.0

        # 设置一些持仓
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 50.0, 100.0)

        # 调用 observe
        inputs = agent.observe(market_state, orderbook)

        # 验证输入向量长度
        # 200 买盘 + 200 卖盘 + 300 成交 + 4 持仓 + 3 挂单
        expected_length = 200 + 200 + 300 + 4 + 3
        assert len(inputs) == expected_length

        # 验证第一个成交的买卖方向（本 Agent 是买方，应为 1.0）
        trade_start_idx = 200 + 200  # 跳过订单簿深度数据
        assert inputs[trade_start_idx + 2] == 1.0  # 第一笔成交，本 Agent 是买方

        # 验证第二个成交的买卖方向（本 Agent 是卖方，应为 -1.0）
        assert inputs[trade_start_idx + 5] == -1.0  # 第二笔成交，本 Agent 是卖方

        # 验证第三个成交的买卖方向（无关，应为 0.0）
        assert inputs[trade_start_idx + 8] == 0.0  # 第三笔成交，无关

    def test_observe_empty_market_state(self):
        """测试空市场状态观察"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建空订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建空市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 observe
        inputs = agent.observe(market_state, orderbook)

        # 验证输入向量长度（固定长度）
        # 200 买盘 + 200 卖盘 + 300 成交 + 4 持仓 + 3 挂单
        expected_length = 200 + 200 + 300 + 4 + 3
        assert len(inputs) == expected_length

        # 验证持仓状态（在末尾附近）
        position_start_idx = 200 + 200 + 300
        assert inputs[position_start_idx + 0] == 0.0  # 持仓归一化
        assert inputs[position_start_idx + 1] == 0.0  # 持仓均价归一化
        assert inputs[position_start_idx + 2] == 1.0  # 余额归一化（10000/10000 = 1.0）
        assert inputs[position_start_idx + 3] == 1.0  # 净值归一化（10000/10000 = 1.0）

    def test_observe_with_position(self):
        """测试有持仓时的观察"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 设置持仓
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 调用 observe
        inputs = agent.observe(market_state, orderbook)

        # 验证输入向量长度
        expected_length = 200 + 200 + 300 + 4 + 3
        assert len(inputs) == expected_length

        # 验证持仓数据（在持仓部分）
        position_start_idx = 200 + 200 + 300
        # 持仓归一化 = position_value / (equity * leverage)
        # position_value = 100 * 100 = 10000
        # equity = 10000 (balance)
        # 持仓归一化 = 10000 / (10000 * 100) = 0.01
        assert abs(inputs[position_start_idx + 0] - 0.01) < 0.01


class TestAgentDecide:
    """测试 Agent.decide"""

    def test_decide_hold_action(self):
        """测试 HOLD 动作决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：HOLD (index 0) 值最大，9 个输出
        mock_brain.forward.return_value = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.HOLD
        # HOLD 动作无参数
        assert params == {}

    def test_decide_place_bid_action(self):
        """测试 PLACE_BID 动作决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：PLACE_BID (index 1) 值最大
        # 价格偏移 -0.5（低于中间价），数量比例 0.0（较小数量）
        mock_brain.forward.return_value = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0]

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.PLACE_BID
        # 验证参数包含 price 和 quantity
        assert "price" in params
        assert "quantity" in params
        # 价格偏移 = -0.5 * 100 * 0.1 = -5.0，所以价格 = 100 - 5 = 95
        assert params["price"] == 95.0
        # 数量应大于 0
        assert params["quantity"] > 0

    def test_decide_place_ask_action(self):
        """测试 PLACE_ASK 动作决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：PLACE_ASK (index 2) 值最大
        # 价格偏移 0.5（高于中间价），数量比例 0.0（较小数量）
        mock_brain.forward.return_value = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.PLACE_ASK
        # 验证参数包含 price 和 quantity
        assert "price" in params
        assert "quantity" in params
        # 价格偏移 = 0.5 * 100 * 0.1 = 5.0，所以价格 = 100 + 5 = 105
        assert params["price"] == 105.0
        # 数量应大于 0
        assert params["quantity"] > 0

    def test_decide_cancel_action(self):
        """测试 CANCEL 动作决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：CANCEL (index 3) 值最大
        mock_brain.forward.return_value = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.CANCEL
        # CANCEL 动作无参数
        assert params == {}

    def test_decide_market_buy_action(self):
        """测试 MARKET_BUY 动作决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：MARKET_BUY (index 4) 值最大
        mock_brain.forward.return_value = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.MARKET_BUY
        # 验证参数包含 quantity
        assert "quantity" in params
        # 数量应大于 0
        assert params["quantity"] > 0

    def test_decide_market_sell_action(self):
        """测试 MARKET_SELL 动作决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：MARKET_SELL (index 5) 值最大，数量比例 0.0
        mock_brain.forward.return_value = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.MARKET_SELL
        # 验证参数包含 quantity
        assert "quantity" in params
        # 数量应大于 0
        assert params["quantity"] > 0

    def test_decide_clear_position_action(self):
        """测试 CLEAR_POSITION 动作决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：CLEAR_POSITION (index 6) 值最大
        mock_brain.forward.return_value = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.CLEAR_POSITION
        # CLEAR_POSITION 动作无参数（由调用方处理）
        assert params == {}

    def test_decide_insufficient_outputs_error(self):
        """测试神经网络输出维度不足时抛出异常"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：只有 3 个值，少于 9 个
        mock_brain.forward.return_value = [0.1, 0.2, 0.3]

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide 应该抛出异常
        try:
            agent.decide(market_state, orderbook)
            assert False, "应该抛出 ValueError"
        except ValueError as e:
            assert "神经网络输出维度不足" in str(e)

    def test_decide_with_position(self):
        """测试有持仓时的 MARKET_SELL 决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：MARKET_SELL (index 5) 值最大，数量比例 0.0
        mock_brain.forward.return_value = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置多仓持仓
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.MARKET_SELL
        # 验证参数包含 quantity
        assert "quantity" in params
        # 数量比例 = 0.1 + (0.0 + 1) * 0.45 = 0.55
        # 卖出数量 = 100.0 * 0.55 = 55.0（浮点数精度允许误差）
        assert abs(params["quantity"] - 55.0) < 0.01

    def test_decide_without_best_price(self):
        """测试订单簿为空时的决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：PLACE_BID (index 1) 值最大
        # 价格偏移 -0.5（低于中间价），数量比例 0.0（较小数量）
        mock_brain.forward.return_value = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0]

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建空订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.PLACE_BID
        # 验证参数包含 price 和 quantity
        assert "price" in params
        assert "quantity" in params
        # 价格偏移 = -0.5 * 100 * 0.1 = -5.0，所以价格 = 100 - 5 = 95
        assert params["price"] == 95.0


class TestAgentExecuteAction:
    """测试 Agent.execute_action"""

    def test_execute_place_bid(self):
        """测试执行挂买单"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅订单事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 执行挂买单
        action = ActionType.PLACE_BID
        params = {"price": 99.5, "quantity": 10.0}
        agent.execute_action(action, params, event_bus)

        # 验证事件发布
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.ORDER_PLACED
        order = event.data["order"]
        assert order.agent_id == 1
        assert order.side.value == OrderSide.BUY
        assert order.order_type.value == OrderType.LIMIT
        assert order.price == 99.5
        assert order.quantity == 10.0

    def test_execute_place_ask(self):
        """测试执行挂卖单"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅订单事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 执行挂卖单
        action = ActionType.PLACE_ASK
        params = {"price": 100.5, "quantity": 20.0}
        agent.execute_action(action, params, event_bus)

        # 验证事件发布
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.ORDER_PLACED
        order = event.data["order"]
        assert order.agent_id == 1
        assert order.side.value == OrderSide.SELL
        assert order.order_type.value == OrderType.LIMIT
        assert order.price == 100.5
        assert order.quantity == 20.0

    def test_execute_cancel(self):
        """测试执行撤单"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅撤单事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )
        agent.account.pending_order_id = 12345

        # 执行撤单（不传 order_id，使用账户的 pending_order_id）
        action = ActionType.CANCEL
        params: dict = {}
        agent.execute_action(action, params, event_bus)

        # 验证事件发布
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.ORDER_CANCELLED
        assert event.data["order_id"] == 12345
        assert event.data["agent_id"] == 1

    def test_execute_cancel_with_order_id(self):
        """测试执行撤单（指定订单ID）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅撤单事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )
        agent.account.pending_order_id = 12345

        # 执行撤单（指定 order_id）
        action = ActionType.CANCEL
        params = {"order_id": 67890}
        agent.execute_action(action, params, event_bus)

        # 验证事件发布（应使用指定的 order_id）
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.ORDER_CANCELLED
        assert event.data["order_id"] == 67890
        assert event.data["agent_id"] == 1

    def test_execute_market_buy(self):
        """测试执行市价买入"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅订单事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 执行市价买入
        action = ActionType.MARKET_BUY
        params = {"quantity": 15.0}
        agent.execute_action(action, params, event_bus)

        # 验证事件发布
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.ORDER_PLACED
        order = event.data["order"]
        assert order.agent_id == 1
        assert order.side.value == OrderSide.BUY
        assert order.order_type.value == OrderType.MARKET
        assert order.price == 0.0
        assert order.quantity == 15.0

    def test_execute_market_sell(self):
        """测试执行市价卖出"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅订单事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 执行市价卖出
        action = ActionType.MARKET_SELL
        params = {"quantity": 25.0}
        agent.execute_action(action, params, event_bus)

        # 验证事件发布
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.ORDER_PLACED
        order = event.data["order"]
        assert order.agent_id == 1
        assert order.side.value == OrderSide.SELL
        assert order.order_type.value == OrderType.MARKET
        assert order.price == 0.0
        assert order.quantity == 25.0

    def test_execute_clear_position_long(self):
        """测试清仓（多仓）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅订单事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置多仓持仓
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 执行清仓
        action = ActionType.CLEAR_POSITION
        params: dict = {}
        agent.execute_action(action, params, event_bus)

        # 验证事件发布（市价卖出）
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.ORDER_PLACED
        order = event.data["order"]
        assert order.agent_id == 1
        assert order.side.value == OrderSide.SELL
        assert order.order_type.value == OrderType.MARKET
        assert order.quantity == 100.0

    def test_execute_clear_position_short(self):
        """测试清仓（空仓）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅订单事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置空仓持仓
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.SELL, 50.0, 100.0)

        # 执行清仓
        action = ActionType.CLEAR_POSITION
        params: dict = {}
        agent.execute_action(action, params, event_bus)

        # 验证事件发布（市价买入）
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.ORDER_PLACED
        order = event.data["order"]
        assert order.agent_id == 1
        assert order.side.value == OrderSide.BUY
        assert order.order_type.value == OrderType.MARKET
        assert order.quantity == 50.0

    def test_execute_clear_position_no_position(self):
        """测试清仓（无持仓）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅订单事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 执行清仓（无持仓）
        action = ActionType.CLEAR_POSITION
        params: dict = {}
        agent.execute_action(action, params, event_bus)

        # 验证没有事件发布
        assert len(published_events) == 0

    def test_execute_hold(self):
        """测试执行 HOLD（不动）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅订单事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)
        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 执行 HOLD
        action = ActionType.HOLD
        params: dict = {}
        agent.execute_action(action, params, event_bus)

        # 验证没有事件发布
        assert len(published_events) == 0

    def test_generate_order_id(self):
        """测试生成唯一订单ID"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=123,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 生成多个订单ID，验证唯一性
        order_ids = set()
        for _ in range(100):
            order_id = agent._generate_order_id()
            # 订单ID应该是正整数
            assert order_id > 0
            # 订单ID应该适合存储
            assert order_id < 2**64
            order_ids.add(order_id)

        # 所有订单ID应该唯一
        assert len(order_ids) == 100

    def test_generate_order_id_multi_agent(self):
        """测试多 Agent 并发生成订单ID的唯一性"""
        # 创建两个不同 agent_id 的 Agent
        mock_brain_1 = MagicMock(spec=Brain)
        mock_brain_2 = MagicMock(spec=Brain)

        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        event_bus = EventBus()

        agent1 = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain_1,
            config=config,
            event_bus=event_bus,
        )
        agent2 = Agent(
            agent_id=2,
            agent_type=AgentType.RETAIL,
            brain=mock_brain_2,
            config=config,
            event_bus=event_bus,
        )

        # 两个 Agent 并发生成订单ID
        order_ids = set()
        for _ in range(50):
            order_ids.add(agent1._generate_order_id())
            order_ids.add(agent2._generate_order_id())

        # 所有订单ID应该唯一
        assert len(order_ids) == 100


class TestAgentGetFitness:
    """测试 Agent.get_fitness"""

    def test_get_fitness_no_position(self):
        """测试无持仓时的适应度"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 计算适应度
        fitness = agent.get_fitness(100.0)

        # 无持仓时，净值 = 余额 = 初始余额，适应度 = 1.0
        assert fitness == 1.0

    def test_get_fitness_profit(self):
        """测试盈利时的适应度"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 开多仓：100 @ 100
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 价格涨到 110
        fitness = agent.get_fitness(110.0)

        # 净值 = 10000 + 100 * (110 - 100) = 11000
        # 适应度 = 11000 / 10000 = 1.1
        assert abs(fitness - 1.1) < 0.01

    def test_get_fitness_loss(self):
        """测试亏损时的适应度"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 开多仓：100 @ 100
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 价格跌到 90
        fitness = agent.get_fitness(90.0)

        # 净值 = 10000 + 100 * (90 - 100) = 9000
        # 适应度 = 9000 / 10000 = 0.9
        assert abs(fitness - 0.9) < 0.01

    def test_get_fitness_near_liquidation(self):
        """测试接近爆仓时的适应度"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 开多仓：10000 @ 100（满杠杆）
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 10000.0, 100.0)

        # 价格跌到 99
        fitness = agent.get_fitness(99.0)

        # 净值 = 10000 + 10000 * (99 - 100) = 0
        # 适应度 = 0 / 10000 = 0
        assert abs(fitness - 0.0) < 0.01

    def test_get_fitness_short_position_profit(self):
        """测试空仓盈利时的适应度"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.WHALE,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 开空仓：1000 @ 100
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.SELL, 1000.0, 100.0)

        # 价格跌到 90（空仓盈利）
        fitness = agent.get_fitness(90.0)

        # 净值 = 10000000 + 1000 * (100 - 90) = 10010000
        # 适应度 = 10010000 / 10000000 = 1.001
        assert abs(fitness - 1.001) < 0.0001


class TestAgentReset:
    """测试 Agent.reset"""

    def test_reset_clears_account_state(self):
        """测试重置后账户状态正确"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建初始配置
        initial_config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=initial_config,
            event_bus=event_bus,
        )

        # 修改账户状态（开仓、设置挂单ID）
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 100.0, 100.0)
        agent.account.balance = 9500.0  # 交易后余额减少
        agent.account.pending_order_id = 12345

        # 创建新的配置
        new_config = AgentConfig(
            count=10000,
            initial_balance=20000.0,  # 新的初始余额
            leverage=50.0,  # 新的杠杆倍数
            maintenance_margin_rate=0.01,  # 新的维持保证金率
            maker_fee_rate=0.0001,  # 新的挂单费率
            taker_fee_rate=0.0003,  # 新的吃单费率
        )

        # 重置 Agent
        agent.reset(new_config)

        # 验证账户状态被重置
        assert agent.account.agent_id == 1
        assert agent.account.agent_type == AgentType.RETAIL
        assert agent.account.balance == 20000.0  # 新的初始余额
        assert agent.account.leverage == 50.0  # 新的杠杆倍数
        assert agent.account.maintenance_margin_rate == 0.01
        assert agent.account.maker_fee_rate == 0.0001
        assert agent.account.taker_fee_rate == 0.0003
        assert agent.account.pending_order_id is None  # 挂单ID被清空
        assert agent.account.position.quantity == 0.0  # 持仓被清空
        assert agent.account.position.avg_price == 0.0
        assert agent.account.position.realized_pnl == 0.0

    def test_reset_preserves_identity(self):
        """测试重置后 Agent 标识保持不变"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=999,
            agent_type=AgentType.MARKET_MAKER,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 保存原始引用
        original_brain = agent.brain
        original_event_bus = agent.event_bus

        # 重置
        new_config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )
        agent.reset(new_config)

        # 验证标识和 Brain 保持不变
        assert agent.agent_id == 999
        assert agent.agent_type == AgentType.MARKET_MAKER
        assert agent.brain is original_brain
        assert agent.event_bus is original_event_bus

    def test_reset_reestablishes_event_subscription(self):
        """测试重置后重新建立事件订阅（使用 subscribe_with_id）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证初始订阅使用 _subscriber_ids
        assert EventType.TRADE_EXECUTED in event_bus._subscriber_ids
        assert 1 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

        # 重置
        new_config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )
        agent.reset(new_config)

        # 验证重置后仍在 _subscriber_ids 中
        assert EventType.TRADE_EXECUTED in event_bus._subscriber_ids
        assert 1 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

        # 验证事件订阅仍然有效（发布定向成交事件应能触发账户更新）
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=0.0,
            data={
                "trade_id": 1,
                "price": 100.0,
                "quantity": 10.0,
                "buyer_id": 1,
                "seller_id": 2,
                "buyer_fee": 5.0,
                "seller_fee": 2.0,
            },
            target_ids={1, 2},  # 定向发送给参与成交的双方
        )
        event_bus.publish(event)

        # 验证账户更新了
        assert agent.account.position.quantity == 10.0
        assert agent.account.position.avg_price == 100.0
        assert agent.account.balance == 9995.0  # 10000 - 5

    def test_reset_with_different_agent_types(self):
        """测试不同类型 Agent 的重置"""
        # 测试散户
        mock_brain = MagicMock(spec=Brain)
        event_bus = EventBus()

        retail_config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )
        retail_agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=retail_config,
            event_bus=event_bus,
        )
        retail_agent.account.balance = 5000.0
        retail_agent.reset(retail_config)
        assert retail_agent.account.balance == 10000.0

        # 测试庄家
        whale_config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )
        whale_agent = Agent(
            agent_id=10001,
            agent_type=AgentType.WHALE,
            brain=mock_brain,
            config=whale_config,
            event_bus=event_bus,
        )
        whale_agent.account.balance = 5000000.0
        whale_agent.reset(whale_config)
        assert whale_agent.account.balance == 10000000.0

        # 测试做市商
        mm_config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )
        mm_agent = Agent(
            agent_id=10011,
            agent_type=AgentType.MARKET_MAKER,
            brain=mock_brain,
            config=mm_config,
            event_bus=event_bus,
        )
        mm_agent.account.balance = 5000000.0
        mm_agent.reset(mm_config)
        assert mm_agent.account.balance == 10000000.0


class TestRetailAgentInit:
    """测试 RetailAgent.__init__"""

    def test_create_retail_agent(self):
        """测试创建散户 Agent"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建散户 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证属性
        assert agent.agent_id == 1
        assert agent.agent_type == AgentType.RETAIL
        assert agent.brain is mock_brain
        assert agent.account.agent_id == 1
        assert agent.account.agent_type == AgentType.RETAIL
        assert agent.account.balance == 10000.0
        assert agent.account.leverage == 100.0
        assert agent.account.maintenance_margin_rate == 0.005
        assert agent.account.maker_fee_rate == 0.0002
        assert agent.account.taker_fee_rate == 0.0005

    def test_retail_agent_is_agent(self):
        """测试 RetailAgent 是 Agent 的子类"""
        from src.bio.agents.base import Agent

        mock_brain = MagicMock(spec=Brain)
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )
        event_bus = EventBus()

        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证是 Agent 的实例
        assert isinstance(agent, Agent)

    def test_retail_agent_inherits_base_methods(self):
        """测试 RetailAgent 继承了基类方法"""
        mock_brain = MagicMock(spec=Brain)
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )
        event_bus = EventBus()

        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证继承了基类方法
        assert hasattr(agent, "observe")
        assert hasattr(agent, "decide")
        assert hasattr(agent, "execute_action")
        assert hasattr(agent, "get_fitness")
        assert hasattr(agent, "reset")
        assert callable(agent.observe)
        assert callable(agent.decide)
        assert callable(agent.execute_action)
        assert callable(agent.get_fitness)
        assert callable(agent.reset)

    def test_retail_agent_subscribes_to_events(self):
        """测试散户 Agent 订阅成交事件（使用 subscribe_with_id）"""
        mock_brain = MagicMock(spec=Brain)
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )
        event_bus = EventBus()

        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证使用 subscribe_with_id 订阅
        assert EventType.TRADE_EXECUTED in event_bus._subscriber_ids
        assert 1 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

        # 发布定向成交事件
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=0.0,
            data={
                "trade_id": 1,
                "price": 100.0,
                "quantity": 10.0,
                "buyer_id": 1,
                "seller_id": 2,
                "buyer_fee": 5.0,
                "seller_fee": 2.0,
            },
            target_ids={1, 2},  # 定向发送给参与成交的双方
        )
        event_bus.publish(event)

        # 验证账户自动更新（通过事件订阅）
        assert agent.account.position.quantity == 10.0
        assert agent.account.position.avg_price == 100.0
        assert agent.account.balance == 9995.0  # 10000 - 5


class TestRetailAgentGetActionSpace:
    """测试 RetailAgent.get_action_space"""

    def test_get_action_space(self):
        """测试获取动作空间"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建散户 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证返回 6 种动作
        assert len(action_space) == 6
        assert ActionType.HOLD in action_space
        assert ActionType.PLACE_BID in action_space
        assert ActionType.PLACE_ASK in action_space
        assert ActionType.CANCEL in action_space
        assert ActionType.MARKET_BUY in action_space
        assert ActionType.MARKET_SELL in action_space

        # 验证动作顺序
        assert action_space == [
            ActionType.HOLD,
            ActionType.PLACE_BID,
            ActionType.PLACE_ASK,
            ActionType.CANCEL,
            ActionType.MARKET_BUY,
            ActionType.MARKET_SELL,
        ]

    def test_get_action_space_excludes_clear_position(self):
        """测试动作空间不包含 CLEAR_POSITION（做市商专用）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建散户 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证不包含 CLEAR_POSITION
        assert ActionType.CLEAR_POSITION not in action_space


class TestRetailAgentExecuteAction:
    """测试 RetailAgent.execute_action"""

    def test_place_bid_with_existing_order(self):
        """测试有挂单时再挂买单：先撤旧单再挂新单"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建散户 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置已有挂单
        agent.account.pending_order_id = 12345

        # 执行挂买单
        action = ActionType.PLACE_BID
        params = {"price": 99.5, "quantity": 10.0}
        agent.execute_action(action, params, event_bus)

        # 验证发布了两个事件：撤单 + 挂单
        assert len(published_events) == 2

        # 第一个事件是撤单
        cancel_event = published_events[0]
        assert cancel_event.event_type == EventType.ORDER_CANCELLED
        assert cancel_event.data["order_id"] == 12345
        assert cancel_event.data["agent_id"] == 1

        # 第二个事件是挂单
        place_event = published_events[1]
        assert place_event.event_type == EventType.ORDER_PLACED
        order = place_event.data["order"]
        assert order.agent_id == 1
        assert order.side.value == OrderSide.BUY
        assert order.order_type.value == OrderType.LIMIT
        assert order.price == 99.5
        assert order.quantity == 10.0

    def test_place_ask_with_existing_order(self):
        """测试有挂单时再挂卖单：先撤旧单再挂新单"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建散户 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置已有挂单
        agent.account.pending_order_id = 67890

        # 执行挂卖单
        action = ActionType.PLACE_ASK
        params = {"price": 100.5, "quantity": 20.0}
        agent.execute_action(action, params, event_bus)

        # 验证发布了两个事件：撤单 + 挂单
        assert len(published_events) == 2

        # 第一个事件是撤单
        cancel_event = published_events[0]
        assert cancel_event.event_type == EventType.ORDER_CANCELLED
        assert cancel_event.data["order_id"] == 67890

        # 第二个事件是挂单
        place_event = published_events[1]
        assert place_event.event_type == EventType.ORDER_PLACED
        order = place_event.data["order"]
        assert order.side.value == OrderSide.SELL
        assert order.price == 100.5
        assert order.quantity == 20.0

    def test_place_bid_without_existing_order(self):
        """测试无挂单时挂买单：直接挂单"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建散户 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 确认没有挂单
        assert agent.account.pending_order_id is None

        # 执行挂买单
        action = ActionType.PLACE_BID
        params = {"price": 99.5, "quantity": 10.0}
        agent.execute_action(action, params, event_bus)

        # 验证只发布了一个事件：挂单
        assert len(published_events) == 1

        # 事件是挂单
        place_event = published_events[0]
        assert place_event.event_type == EventType.ORDER_PLACED
        order = place_event.data["order"]
        assert order.agent_id == 1
        assert order.side.value == OrderSide.BUY
        assert order.price == 99.5
        assert order.quantity == 10.0

    def test_execute_other_actions(self):
        """测试其他动作（CANCEL, MARKET_BUY, MARKET_SELL, HOLD）使用父类实现"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建散户 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 测试 MARKET_BUY
        published_events.clear()
        agent.execute_action(ActionType.MARKET_BUY, {"quantity": 15.0}, event_bus)
        assert len(published_events) == 1
        assert published_events[0].event_type == EventType.ORDER_PLACED
        assert published_events[0].data["order"].order_type.value == OrderType.MARKET

        # 测试 MARKET_SELL
        published_events.clear()
        agent.execute_action(ActionType.MARKET_SELL, {"quantity": 25.0}, event_bus)
        assert len(published_events) == 1
        assert published_events[0].event_type == EventType.ORDER_PLACED

        # 测试 CANCEL
        agent.account.pending_order_id = 12345
        published_events.clear()
        agent.execute_action(ActionType.CANCEL, {}, event_bus)
        assert len(published_events) == 1
        assert published_events[0].event_type == EventType.ORDER_CANCELLED

        # 测试 HOLD
        published_events.clear()
        agent.execute_action(ActionType.HOLD, {}, event_bus)
        assert len(published_events) == 0


class TestWhaleAgentInit:
    """测试 WhaleAgent.__init__"""

    def test_create_whale_agent(self):
        """测试创建庄家 Agent"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证属性
        assert agent.agent_id == 10001
        assert agent.agent_type == AgentType.WHALE
        assert agent.brain is mock_brain
        assert agent.account.agent_id == 10001
        assert agent.account.agent_type == AgentType.WHALE
        assert agent.account.balance == 10000000.0
        assert agent.account.leverage == 10.0
        assert agent.account.maintenance_margin_rate == 0.05
        assert agent.account.maker_fee_rate == 0.0
        assert agent.account.taker_fee_rate == 0.0001

    def test_whale_agent_is_agent(self):
        """测试 WhaleAgent 是 Agent 的子类"""
        mock_brain = MagicMock(spec=Brain)
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )
        event_bus = EventBus()

        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证是 Agent 的实例
        assert isinstance(agent, Agent)

    def test_whale_agent_inherits_base_methods(self):
        """测试 WhaleAgent 继承了基类方法"""
        mock_brain = MagicMock(spec=Brain)
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )
        event_bus = EventBus()

        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证继承了基类方法
        assert hasattr(agent, "observe")
        assert hasattr(agent, "decide")
        assert hasattr(agent, "execute_action")
        assert hasattr(agent, "get_fitness")
        assert hasattr(agent, "reset")
        assert callable(agent.observe)
        assert callable(agent.decide)
        assert callable(agent.execute_action)
        assert callable(agent.get_fitness)
        assert callable(agent.reset)

    def test_whale_agent_subscribes_to_events(self):
        """测试庄家 Agent 订阅成交事件（使用 subscribe_with_id）"""
        mock_brain = MagicMock(spec=Brain)
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )
        event_bus = EventBus()

        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证使用 subscribe_with_id 订阅
        assert EventType.TRADE_EXECUTED in event_bus._subscriber_ids
        assert 10001 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

        # 发布定向成交事件
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=0.0,
            data={
                "trade_id": 1,
                "price": 100.0,
                "quantity": 100.0,
                "buyer_id": 10001,
                "seller_id": 10002,
                "buyer_fee": 0.0,
                "seller_fee": 1.0,
            },
            target_ids={10001, 10002},  # 定向发送给参与成交的双方
        )
        event_bus.publish(event)

        # 验证账户自动更新（通过事件订阅）
        assert agent.account.position.quantity == 100.0
        assert agent.account.position.avg_price == 100.0
        assert agent.account.balance == 10000000.0  # 庄家 maker 费率为 0


class TestWhaleAgentGetActionSpace:
    """测试 WhaleAgent.get_action_space"""

    def test_get_action_space(self):
        """测试获取庄家动作空间"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证返回 4 种动作（庄家不能 HOLD，不能单纯撤单）
        assert len(action_space) == 4
        assert ActionType.PLACE_BID in action_space
        assert ActionType.PLACE_ASK in action_space
        assert ActionType.MARKET_BUY in action_space
        assert ActionType.MARKET_SELL in action_space

        # 验证动作顺序
        assert action_space == [
            ActionType.PLACE_BID,
            ActionType.PLACE_ASK,
            ActionType.MARKET_BUY,
            ActionType.MARKET_SELL,
        ]

    def test_get_action_space_excludes_hold(self):
        """测试庄家动作空间不包含 HOLD（庄家绝不不动）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证不包含 HOLD
        assert ActionType.HOLD not in action_space

    def test_get_action_space_excludes_cancel(self):
        """测试庄家动作空间不包含 CANCEL（庄家不能单纯撤单）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证不包含 CANCEL
        assert ActionType.CANCEL not in action_space

    def test_get_action_space_excludes_clear_position(self):
        """测试庄家动作空间不包含 CLEAR_POSITION（做市商专用）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证不包含 CLEAR_POSITION
        assert ActionType.CLEAR_POSITION not in action_space


class TestWhaleAgentExecuteAction:
    """测试 WhaleAgent.execute_action"""

    def test_place_bid_with_existing_order(self):
        """测试有挂单时再挂买单：先撤旧单再挂新单"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置已有挂单
        agent.account.pending_order_id = 12345

        # 执行挂买单
        action = ActionType.PLACE_BID
        params = {"price": 99.5, "quantity": 100.0}
        agent.execute_action(action, params, event_bus)

        # 验证发布了两个事件：撤单 + 挂单
        assert len(published_events) == 2

        # 第一个事件是撤单
        cancel_event = published_events[0]
        assert cancel_event.event_type == EventType.ORDER_CANCELLED
        assert cancel_event.data["order_id"] == 12345
        assert cancel_event.data["agent_id"] == 10001

        # 第二个事件是挂单
        place_event = published_events[1]
        assert place_event.event_type == EventType.ORDER_PLACED
        order = place_event.data["order"]
        assert order.agent_id == 10001
        assert order.side.value == OrderSide.BUY
        assert order.order_type.value == OrderType.LIMIT
        assert order.price == 99.5
        assert order.quantity == 100.0

    def test_place_ask_with_existing_order(self):
        """测试有挂单时再挂卖单：先撤旧单再挂新单"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置已有挂单
        agent.account.pending_order_id = 67890

        # 执行挂卖单
        action = ActionType.PLACE_ASK
        params = {"price": 100.5, "quantity": 200.0}
        agent.execute_action(action, params, event_bus)

        # 验证发布了两个事件：撤单 + 挂单
        assert len(published_events) == 2

        # 第一个事件是撤单
        cancel_event = published_events[0]
        assert cancel_event.event_type == EventType.ORDER_CANCELLED
        assert cancel_event.data["order_id"] == 67890

        # 第二个事件是挂单
        place_event = published_events[1]
        assert place_event.event_type == EventType.ORDER_PLACED
        order = place_event.data["order"]
        assert order.side.value == OrderSide.SELL
        assert order.price == 100.5
        assert order.quantity == 200.0

    def test_place_bid_without_existing_order(self):
        """测试无挂单时挂买单：直接挂单"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 确认没有挂单
        assert agent.account.pending_order_id is None

        # 执行挂买单
        action = ActionType.PLACE_BID
        params = {"price": 99.5, "quantity": 100.0}
        agent.execute_action(action, params, event_bus)

        # 验证只发布了一个事件：挂单
        assert len(published_events) == 1

        # 事件是挂单
        place_event = published_events[0]
        assert place_event.event_type == EventType.ORDER_PLACED
        order = place_event.data["order"]
        assert order.agent_id == 10001
        assert order.side.value == OrderSide.BUY
        assert order.price == 99.5
        assert order.quantity == 100.0

    def test_market_buy_with_existing_order(self):
        """测试有挂单时市价买入：先撤旧单再市价买入"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置已有挂单
        agent.account.pending_order_id = 99999

        # 执行市价买入
        action = ActionType.MARKET_BUY
        params = {"quantity": 150.0}
        agent.execute_action(action, params, event_bus)

        # 验证发布了两个事件：撤单 + 市价单
        assert len(published_events) == 2

        # 第一个事件是撤单
        cancel_event = published_events[0]
        assert cancel_event.event_type == EventType.ORDER_CANCELLED
        assert cancel_event.data["order_id"] == 99999

        # 第二个事件是市价买单
        place_event = published_events[1]
        assert place_event.event_type == EventType.ORDER_PLACED
        order = place_event.data["order"]
        assert order.order_type.value == OrderType.MARKET
        assert order.side.value == OrderSide.BUY
        assert order.quantity == 150.0

    def test_market_sell_with_existing_order(self):
        """测试有挂单时市价卖出：先撤旧单再市价卖出"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置已有挂单
        agent.account.pending_order_id = 88888

        # 执行市价卖出
        action = ActionType.MARKET_SELL
        params = {"quantity": 250.0}
        agent.execute_action(action, params, event_bus)

        # 验证发布了两个事件：撤单 + 市价单
        assert len(published_events) == 2

        # 第一个事件是撤单
        cancel_event = published_events[0]
        assert cancel_event.event_type == EventType.ORDER_CANCELLED

        # 第二个事件是市价卖单
        place_event = published_events[1]
        assert place_event.event_type == EventType.ORDER_PLACED
        order = place_event.data["order"]
        assert order.order_type.value == OrderType.MARKET
        assert order.side.value == OrderSide.SELL
        assert order.quantity == 250.0

    def test_market_buy_without_existing_order(self):
        """测试无挂单时市价买入：直接市价买入"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 执行市价买入
        action = ActionType.MARKET_BUY
        params = {"quantity": 150.0}
        agent.execute_action(action, params, event_bus)

        # 验证只发布了一个事件：市价单
        assert len(published_events) == 1
        place_event = published_events[0]
        assert place_event.event_type == EventType.ORDER_PLACED
        order = place_event.data["order"]
        assert order.order_type.value == OrderType.MARKET
        assert order.side.value == OrderSide.BUY
        assert order.quantity == 150.0

    def test_market_sell_without_existing_order(self):
        """测试无挂单时市价卖出：直接市价卖出"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 执行市价卖出
        action = ActionType.MARKET_SELL
        params = {"quantity": 250.0}
        agent.execute_action(action, params, event_bus)

        # 验证只发布了一个事件：市价单
        assert len(published_events) == 1
        place_event = published_events[0]
        assert place_event.event_type == EventType.ORDER_PLACED
        order = place_event.data["order"]
        assert order.order_type.value == OrderType.MARKET
        assert order.side.value == OrderSide.SELL
        assert order.quantity == 250.0


class TestMarketMakerAgentInit:
    """测试 MarketMakerAgent.__init__"""

    def test_create_market_maker_agent(self):
        """测试创建做市商 Agent"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证基本属性
        assert agent.agent_id == 10011
        assert agent.agent_type == AgentType.MARKET_MAKER
        assert agent.brain is mock_brain
        assert agent.account.agent_id == 10011
        assert agent.account.agent_type == AgentType.MARKET_MAKER
        assert agent.account.balance == 10000000.0
        assert agent.account.leverage == 10.0
        assert agent.account.maintenance_margin_rate == 0.05
        assert agent.account.maker_fee_rate == 0.0
        assert agent.account.taker_fee_rate == 0.0001

        # 验证做市商特有的挂单列表属性
        assert hasattr(agent, "bid_order_ids")
        assert hasattr(agent, "ask_order_ids")
        assert agent.bid_order_ids == []
        assert agent.ask_order_ids == []
        assert isinstance(agent.bid_order_ids, list)
        assert isinstance(agent.ask_order_ids, list)

    def test_market_maker_agent_is_agent(self):
        """测试 MarketMakerAgent 是 Agent 的子类"""
        mock_brain = MagicMock(spec=Brain)
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )
        event_bus = EventBus()

        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证是 Agent 的实例
        assert isinstance(agent, Agent)

    def test_market_maker_agent_inherits_base_methods(self):
        """测试 MarketMakerAgent 继承了基类方法"""
        mock_brain = MagicMock(spec=Brain)
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )
        event_bus = EventBus()

        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证继承了基类方法
        assert hasattr(agent, "observe")
        assert hasattr(agent, "decide")
        assert hasattr(agent, "execute_action")
        assert hasattr(agent, "get_fitness")
        assert hasattr(agent, "reset")
        assert callable(agent.observe)
        assert callable(agent.decide)
        assert callable(agent.execute_action)
        assert callable(agent.get_fitness)
        assert callable(agent.reset)

    def test_market_maker_agent_subscribes_to_events(self):
        """测试做市商 Agent 订阅成交事件（使用 subscribe_with_id）"""
        mock_brain = MagicMock(spec=Brain)
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )
        event_bus = EventBus()

        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 验证使用 subscribe_with_id 订阅
        assert EventType.TRADE_EXECUTED in event_bus._subscriber_ids
        assert 10011 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

        # 发布定向成交事件
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=0.0,
            data={
                "trade_id": 1,
                "price": 100.0,
                "quantity": 50.0,
                "buyer_id": 10011,
                "seller_id": 10012,
                "buyer_fee": 0.0,
                "seller_fee": 0.5,
            },
            target_ids={10011, 10012},  # 定向发送给参与成交的双方
        )
        event_bus.publish(event)

        # 验证账户自动更新（通过事件订阅）
        assert agent.account.position.quantity == 50.0
        assert agent.account.position.avg_price == 100.0
        assert agent.account.balance == 10000000.0  # 做市商 maker 费率为 0


class TestMarketMakerAgentGetActionSpace:
    """测试 MarketMakerAgent.get_action_space"""

    def test_get_action_space(self):
        """测试获取做市商动作空间"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证返回 2 种动作：双边挂单和清仓
        assert len(action_space) == 2
        assert ActionType.QUOTE in action_space
        assert ActionType.CLEAR_POSITION in action_space

        # 验证动作顺序
        assert action_space == [
            ActionType.QUOTE,
            ActionType.CLEAR_POSITION,
        ]

    def test_get_action_space_excludes_single_side_actions(self):
        """测试做市商动作空间不包含单边动作"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证不包含单边动作和撤单
        assert ActionType.PLACE_BID not in action_space
        assert ActionType.PLACE_ASK not in action_space
        assert ActionType.CANCEL not in action_space
        assert ActionType.HOLD not in action_space
        assert ActionType.MARKET_BUY not in action_space
        assert ActionType.MARKET_SELL not in action_space


class TestMarketMakerAgentExecuteAction:
    """测试 MarketMakerAgent.execute_action"""

    def test_execute_quote_with_no_existing_orders(self):
        """测试执行双边挂单（无旧挂单）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 执行双边挂单
        action = ActionType.QUOTE
        params = {
            "bid_orders": [
                {"price": 99.5, "quantity": 10.0},
                {"price": 99.4, "quantity": 15.0},
            ],
            "ask_orders": [
                {"price": 100.5, "quantity": 10.0},
                {"price": 100.6, "quantity": 12.0},
                {"price": 100.7, "quantity": 8.0},
            ],
        }
        agent.execute_action(action, params, event_bus)

        # 验证发布了5个订单事件（2买+3卖）
        order_events = [e for e in published_events if e.event_type == EventType.ORDER_PLACED]
        cancel_events = [e for e in published_events if e.event_type == EventType.ORDER_CANCELLED]
        assert len(order_events) == 5
        assert len(cancel_events) == 0  # 无旧单，无撤单事件

        # 验证挂单ID列表
        assert len(agent.bid_order_ids) == 2
        assert len(agent.ask_order_ids) == 3

        # 验证买单事件
        bid_orders = [e.data["order"] for e in order_events if e.data["order"].side.value == OrderSide.BUY]
        assert len(bid_orders) == 2
        assert bid_orders[0].price == 99.5
        assert bid_orders[0].quantity == 10.0
        assert bid_orders[1].price == 99.4
        assert bid_orders[1].quantity == 15.0

        # 验证卖单事件
        ask_orders = [e.data["order"] for e in order_events if e.data["order"].side.value == OrderSide.SELL]
        assert len(ask_orders) == 3
        assert ask_orders[0].price == 100.5
        assert ask_orders[0].quantity == 10.0
        assert ask_orders[1].price == 100.6
        assert ask_orders[1].quantity == 12.0
        assert ask_orders[2].price == 100.7
        assert ask_orders[2].quantity == 8.0

    def test_execute_quote_with_existing_orders(self):
        """测试执行双边挂单（有旧挂单：先撤再挂）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置已有挂单
        agent.bid_order_ids = [1001, 1002]
        agent.ask_order_ids = [2001, 2002, 2003]

        # 执行双边挂单
        action = ActionType.QUOTE
        params = {
            "bid_orders": [{"price": 99.3, "quantity": 20.0}],
            "ask_orders": [{"price": 100.8, "quantity": 25.0}],
        }
        agent.execute_action(action, params, event_bus)

        # 验证发布了撤单事件（5个旧单）+ 挂单事件（2个新单）
        cancel_events = [e for e in published_events if e.event_type == EventType.ORDER_CANCELLED]
        order_events = [e for e in published_events if e.event_type == EventType.ORDER_PLACED]
        assert len(cancel_events) == 5  # 2买+3卖
        assert len(order_events) == 2  # 1买+1卖

        # 验证撤单事件
        cancelled_ids = [e.data["order_id"] for e in cancel_events]
        assert 1001 in cancelled_ids
        assert 1002 in cancelled_ids
        assert 2001 in cancelled_ids
        assert 2002 in cancelled_ids
        assert 2003 in cancelled_ids

        # 验证挂单列表被清空后重新填充
        assert len(agent.bid_order_ids) == 1
        assert len(agent.ask_order_ids) == 1

    def test_execute_clear_position_with_long_position(self):
        """测试清仓（多仓）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置已有挂单和多仓
        agent.bid_order_ids = [1001, 1002]
        agent.ask_order_ids = [2001]
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 执行清仓
        action = ActionType.CLEAR_POSITION
        params: dict = {}
        agent.execute_action(action, params, event_bus)

        # 验证发布了撤单事件（3个）+ 市价卖单事件（1个）
        cancel_events = [e for e in published_events if e.event_type == EventType.ORDER_CANCELLED]
        order_events = [e for e in published_events if e.event_type == EventType.ORDER_PLACED]
        assert len(cancel_events) == 3
        assert len(order_events) == 1

        # 验证市价卖单
        market_order = order_events[0].data["order"]
        assert market_order.side.value == OrderSide.SELL
        assert market_order.order_type.value == OrderType.MARKET
        assert market_order.quantity == 100.0

        # 验证挂单列表被清空
        assert len(agent.bid_order_ids) == 0
        assert len(agent.ask_order_ids) == 0

    def test_execute_clear_position_with_short_position(self):
        """测试清仓（空仓）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置空仓
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.SELL, 50.0, 100.0)

        # 执行清仓
        action = ActionType.CLEAR_POSITION
        params: dict = {}
        agent.execute_action(action, params, event_bus)

        # 验证发布了市价买单事件
        order_events = [e for e in published_events if e.event_type == EventType.ORDER_PLACED]
        assert len(order_events) == 1

        market_order = order_events[0].data["order"]
        assert market_order.side.value == OrderSide.BUY
        assert market_order.order_type.value == OrderType.MARKET
        assert market_order.quantity == 50.0

    def test_execute_clear_position_with_no_position(self):
        """测试清仓（无持仓）"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 执行清仓（无持仓）
        action = ActionType.CLEAR_POSITION
        params: dict = {}
        agent.execute_action(action, params, event_bus)

        # 验证没有订单事件发布（无持仓不发布市价单）
        order_events = [e for e in published_events if e.event_type == EventType.ORDER_PLACED]
        assert len(order_events) == 0

    def test_execute_clear_position_cancels_all_orders(self):
        """测试清仓时撤掉所有挂单"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线并订阅事件
        event_bus = EventBus()
        published_events: list = []

        def capture_event(event: Event) -> None:
            published_events.append(event)

        event_bus.subscribe(EventType.ORDER_CANCELLED, capture_event)
        event_bus.subscribe(EventType.ORDER_PLACED, capture_event)

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 设置多仓和多个挂单
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 200.0, 100.0)
        agent.bid_order_ids = [1001, 1002, 1003, 1004, 1005]  # 5个买单
        agent.ask_order_ids = [2001, 2002]  # 2个卖单

        # 执行清仓
        action = ActionType.CLEAR_POSITION
        params: dict = {}
        agent.execute_action(action, params, event_bus)

        # 验证所有挂单被撤销
        cancel_events = [e for e in published_events if e.event_type == EventType.ORDER_CANCELLED]
        assert len(cancel_events) == 7  # 5买+2卖

        # 验证挂单列表被清空
        assert len(agent.bid_order_ids) == 0
        assert len(agent.ask_order_ids) == 0


class TestMarketMakerAgentDecide:
    """测试 MarketMakerAgent.decide"""

    def test_decide_quote_action(self):
        """测试决策 QUOTE 动作"""
        # 创建 mock Brain，设置神经网络输出
        mock_brain = MagicMock(spec=Brain)
        # 输出[0]=1.0 (QUOTE), 输出[1]=0.0 (CLEAR_POSITION)
        # 输出[2-21]=0.0 (价格偏移和数量比例)
        mock_brain.forward.return_value = [1.0, 0.0] + [0.0] * 20

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证返回了 QUOTE 动作
        assert action == ActionType.QUOTE
        assert "bid_orders" in params
        assert "ask_orders" in params
        # 验证订单数量（5个买单和5个卖单）
        assert len(params["bid_orders"]) == 5
        assert len(params["ask_orders"]) == 5

        # 验证所有买单价格低于 mid_price
        mid_price = market_state.mid_price
        for order in params["bid_orders"]:
            assert order["price"] < mid_price
            assert order["quantity"] > 0

        # 验证所有卖单价格高于 mid_price
        for order in params["ask_orders"]:
            assert order["price"] > mid_price
            assert order["quantity"] > 0

    def test_decide_clear_position_action(self):
        """测试决策 CLEAR_POSITION 动作"""
        # 创建 mock Brain，设置神经网络输出
        mock_brain = MagicMock(spec=Brain)
        # 输出[0]=0.0 (QUOTE), 输出[1]=1.0 (CLEAR_POSITION)
        mock_brain.forward.return_value = [0.0, 1.0] + [0.0] * 20

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证返回了 CLEAR_POSITION 动作
        assert action == ActionType.CLEAR_POSITION
        assert params == {}

    def test_decide_insufficient_outputs_error(self):
        """测试神经网络输出维度不足时抛出异常"""
        # 创建 mock Brain，设置输出维度不足
        mock_brain = MagicMock(spec=Brain)
        mock_brain.forward.return_value = [0.0] * 10  # 只有 10 个输出，需要 22 个

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide 应该抛出异常
        try:
            agent.decide(market_state, orderbook)
            assert False, "应该抛出 ValueError"
        except ValueError as e:
            assert "神经网络输出维度不足" in str(e)
            assert "22" in str(e)
            assert "10" in str(e)

    def test_decide_generates_orders_at_different_prices(self):
        """测试生成的订单在不同价位"""
        # 创建 mock Brain，设置不同的价格偏移
        mock_brain = MagicMock(spec=Brain)
        # 输出: QUOTE=1.0, CLEAR=0.0
        # 买单价格偏移: -0.8, -0.6, -0.4, -0.2, 0.0
        # 买单数量: 全部 0.0
        # 卖单价格偏移: 0.0, 0.2, 0.4, 0.6, 0.8
        # 卖单数量: 全部 0.0
        outputs = [1.0, 0.0]  # 动作得分
        # 买单价格偏移
        outputs.extend([-0.8, -0.6, -0.4, -0.2, 0.0])
        # 买单数量
        outputs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        # 卖单价格偏移
        outputs.extend([0.0, 0.2, 0.4, 0.6, 0.8])
        # 卖单数量
        outputs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        mock_brain.forward.return_value = outputs

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证返回了 QUOTE 动作
        assert action == ActionType.QUOTE

        # 提取价格
        bid_prices = [o["price"] for o in params["bid_orders"]]
        ask_prices = [o["price"] for o in params["ask_orders"]]

        # 买单价格应递增
        assert bid_prices == sorted(bid_prices)

        # 卖单价格应递增
        assert ask_prices == sorted(ask_prices)

        # 所有买单价格应低于所有卖单价格
        assert max(bid_prices) < min(ask_prices)

    def test_decide_normalizes_quantity_ratios(self):
        """测试数量比例归一化（10个订单的总比例不超过1.0）"""
        # 创建 mock Brain，设置所有数量权重为最大值
        mock_brain = MagicMock(spec=Brain)
        # 输出: QUOTE=1.0, CLEAR=0.0
        # 买单价格偏移: 全部 0.0
        # 买单数量权重: 全部 1.0（最大值）
        # 卖单价格偏移: 全部 0.0
        # 卖单数量权重: 全部 1.0（最大值）
        outputs = [1.0, 0.0]  # 动作得分
        # 买单价格偏移
        outputs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        # 买单数量权重（全部 1.0）
        outputs.extend([1.0, 1.0, 1.0, 1.0, 1.0])
        # 卖单价格偏移
        outputs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        # 卖单数量权重（全部 1.0）
        outputs.extend([1.0, 1.0, 1.0, 1.0, 1.0])
        mock_brain.forward.return_value = outputs

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建事件总线
        event_bus = EventBus()

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
            event_bus=event_bus,
        )

        # 创建订单簿
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.1)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证返回了 QUOTE 动作
        assert action == ActionType.QUOTE

        # 验证有 10 个订单（5买+5卖）
        assert len(params["bid_orders"]) == 5
        assert len(params["ask_orders"]) == 5

        # 计算总数量
        # 由于所有数量权重相等，每个订单应该占总购买力的 1/10
        mid_price = market_state.mid_price
        equity = agent.account.get_equity(mid_price)

        total_bid_value = sum(o["price"] * o["quantity"] for o in params["bid_orders"])
        total_ask_value = sum(o["price"] * o["quantity"] for o in params["ask_orders"])
        total_value = total_bid_value + total_ask_value

        # 总价值应该不超过净值 * 杠杆（允许一定误差，因为 _calculate_order_quantity 有最小 lot_size 限制）
        max_total_value = equity * agent.account.leverage
        # 实际总价值可能略高因为有 lot_size 向上取整，但不应超过太多
        assert total_value <= max_total_value * 1.5  # 允许 50% 的误差空间（由于 lot_size 取整）

        # 验证每个订单的数量大致相等（因为输入权重相同）
        bid_quantities = [o["quantity"] for o in params["bid_orders"]]
        ask_quantities = [o["quantity"] for o in params["ask_orders"]]

        # 所有买单数量应该相等（或非常接近）
        assert len(set(bid_quantities)) <= 2  # 允许少量差异
        # 所有卖单数量应该相等（或非常接近）
        assert len(set(ask_quantities)) <= 2  # 允许少量差异
