"""测试 Agent 模块"""

from unittest.mock import MagicMock

from src.bio.agents.base import Agent, ActionType
from src.bio.agents.retail import RetailAgent
from src.bio.agents.whale import WhaleAgent
from src.config.config import AgentConfig, AgentType
from src.core.event_engine.events import Event, EventType
from src.core.event_engine.event_bus import EventBus
from src.bio.brain.brain import Brain
from src.market.orderbook.order import OrderSide, OrderType


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

    def test_on_irrelevant_trade_event(self):
        """测试处理不相关的成交事件"""
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

        # 创建不相关的成交事件（agent_id=2 和 agent_id=3 成交）
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
        )

        # 记录初始账户状态
        initial_balance = agent.account.balance
        initial_quantity = agent.account.position.quantity

        # 处理事件
        agent._on_trade_event(event)

        # 验证账户未更新
        assert agent.account.balance == initial_balance
        assert agent.account.position.quantity == initial_quantity

    def test_event_subscription(self):
        """测试事件订阅"""
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

        # 发布买入成交事件
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

        # 发布事件
        event_bus.publish(event)

        # 验证账户自动更新（通过事件订阅）
        assert agent.account.position.quantity == 10.0
        assert agent.account.position.avg_price == 100.0
        assert agent.account.balance == 9995.0  # 10000 - 5


class TestAgentObserve:
    """测试 Agent.observe"""

    def test_observe_normal(self):
        """测试正常观察市场状态"""
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

        # 创建订单簿（需要先编译 Cython 模块）
        # 使用 mock 对象模拟 OrderBook
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.1)
        orderbook.last_price = 100.0

        # 添加一些买卖盘
        from src.market.orderbook.order import Order, OrderSide, OrderType
        for i in range(5):
            # 买盘（从 99.9 开始递减）
            bid_order = Order(
                order_id=i * 2,
                agent_id=10 + i,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=100.0 - (i + 1) * 0.1,  # 99.9, 99.8, ...
                quantity=10.0 + i,
            )
            orderbook.add_order(bid_order)
            # 卖盘（从 100.1 开始递增）
            ask_order = Order(
                order_id=i * 2 + 1,
                agent_id=20 + i,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=100.0 + (i + 1) * 0.1,  # 100.1, 100.2, ...
                quantity=10.0 + i,
            )
            orderbook.add_order(ask_order)

        # 创建成交记录
        from src.market.matching.trade import Trade
        trades = [
            Trade(
                trade_id=1,
                price=99.9,
                quantity=10.0,
                buyer_id=1,  # 本 Agent 是买方
                seller_id=2,
                buyer_fee=0.5,
                seller_fee=0.2,
            ),
            Trade(
                trade_id=2,
                price=100.1,
                quantity=5.0,
                buyer_id=3,
                seller_id=1,  # 本 Agent 是卖方
                buyer_fee=0.25,
                seller_fee=0.0,
            ),
            Trade(
                trade_id=3,
                price=100.0,
                quantity=8.0,
                buyer_id=4,
                seller_id=5,  # 无关成交
                buyer_fee=0.4,
                seller_fee=0.16,
            ),
        ]

        # 设置一些持仓
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 50.0, 100.0)

        # 调用 observe
        inputs = agent.observe(orderbook, trades)

        # 验证输入向量长度
        # 5档买盘 * 2 + 5档卖盘 * 2 + 3笔成交 * 3 + 4个自身状态
        expected_length = 5 * 2 + 5 * 2 + 3 * 3 + 4
        assert len(inputs) == expected_length

        # 验证买盘价格归一化（第一个买盘 99.9，mid_price 约 100）
        assert inputs[0] < 0  # 价格低于中间价，归一化为负

        # 验证第一个成交的买卖方向（本 Agent 是买方，应为 1.0）
        bid_start_idx = 5 * 2 + 5 * 2  # 跳过订单簿深度数据
        assert inputs[bid_start_idx + 2] == 1.0  # 第一笔成交，本 Agent 是买方

        # 验证第二个成交的买卖方向（本 Agent 是卖方，应为 -1.0）
        assert inputs[bid_start_idx + 5] == -1.0  # 第二笔成交，本 Agent 是卖方

        # 验证第三个成交的买卖方向（无关，应为 0.0）
        assert inputs[bid_start_idx + 8] == 0.0  # 第三笔成交，无关

    def test_observe_empty_orderbook(self):
        """测试空订单簿观察"""
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

        # 空成交记录
        trades: list = []

        # 调用 observe
        inputs = agent.observe(orderbook, trades)

        # 验证输入向量长度（只有4个自身状态）
        assert len(inputs) == 4

        # 验证持仓状态
        assert inputs[0] == 0.0  # 持仓数量归一化
        assert inputs[1] == 0.0  # 持仓均价归一化
        assert inputs[2] == 1.0  # 余额归一化（10000/10000 = 1.0）
        assert inputs[3] == 1.0  # 净值归一化（10000/10000 = 1.0）

    def test_observe_empty_trades(self):
        """测试空成交记录观察"""
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

        # 添加一些买卖盘
        from src.market.orderbook.order import Order, OrderSide, OrderType
        for i in range(3):
            bid_order = Order(
                order_id=i * 2,
                agent_id=10 + i,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=100.0 - i * 0.1,
                quantity=10.0,
            )
            orderbook.add_order(bid_order)

        # 空成交记录
        trades: list = []

        # 调用 observe
        inputs = agent.observe(orderbook, trades)

        # 验证输入向量长度
        # 3档买盘 * 2 + 0档卖盘 * 2 + 0笔成交 * 3 + 4个自身状态
        expected_length = 3 * 2 + 0 + 0 + 4
        assert len(inputs) == expected_length


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

        # 空成交记录
        trades: list = []

        # 调用 decide
        action, params = agent.decide(orderbook, trades)

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

        # 空成交记录
        trades: list = []

        # 调用 decide
        action, params = agent.decide(orderbook, trades)

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

        # 空成交记录
        trades: list = []

        # 调用 decide
        action, params = agent.decide(orderbook, trades)

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

        # 空成交记录
        trades: list = []

        # 调用 decide
        action, params = agent.decide(orderbook, trades)

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

        # 空成交记录
        trades: list = []

        # 调用 decide
        action, params = agent.decide(orderbook, trades)

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

        # 空成交记录
        trades: list = []

        # 调用 decide
        action, params = agent.decide(orderbook, trades)

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

        # 空成交记录
        trades: list = []

        # 调用 decide
        action, params = agent.decide(orderbook, trades)

        # 验证动作类型
        assert action == ActionType.CLEAR_POSITION
        # CLEAR_POSITION 动作无参数（由调用方处理）
        assert params == {}

    def test_decide_insufficient_outputs_error(self):
        """测试神经网络输出维度不足时抛出异常"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：只有 3 个值，少于 7 个
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

        # 空成交记录
        trades: list = []

        # 调用 decide 应该抛出异常
        try:
            agent.decide(orderbook, trades)
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

        # 空成交记录
        trades: list = []

        # 调用 decide
        action, params = agent.decide(orderbook, trades)

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

        # 空成交记录
        trades: list = []

        # 调用 decide
        action, params = agent.decide(orderbook, trades)

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
        """测试重置后重新建立事件订阅"""
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

        # 验证事件订阅仍然有效（发布成交事件应能触发账户更新）
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
        """测试散户 Agent 订阅成交事件"""
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

        # 发布成交事件
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
        """测试庄家 Agent 订阅成交事件"""
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

        # 发布成交事件
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
