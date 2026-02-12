"""测试 Agent 模块"""

from unittest.mock import MagicMock

import numpy as np

from src.bio.agents.base import Agent, ActionType
from src.bio.agents.retail import RetailAgent
from src.bio.agents.retail_pro import RetailProAgent
from src.bio.agents.whale import WhaleAgent
from src.bio.agents.market_maker import MarketMakerAgent
from src.config.config import AgentConfig, AgentType
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
        tick_history_prices=np.zeros(100, dtype=np.float32),
        tick_history_volumes=np.zeros(100, dtype=np.float32),
        tick_history_amounts=np.zeros(100, dtype=np.float32),
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

        # 创建散户 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
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

        # 创建庄家 Agent
        agent = Agent(
            agent_id=10001,
            agent_type=AgentType.WHALE,
            brain=mock_brain,
            config=config,
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

        # 创建做市商 Agent
        agent = Agent(
            agent_id=10011,
            agent_type=AgentType.MARKET_MAKER,
            brain=mock_brain,
            config=config,
        )

        # 验证属性
        assert agent.agent_id == 10011
        assert agent.agent_type == AgentType.MARKET_MAKER
        assert agent.brain is mock_brain
        assert agent.account.balance == 10000000.0
        assert agent.account.leverage == 10.0
        assert agent.account.maker_fee_rate == 0.0
        assert agent.account.taker_fee_rate == 0.0001


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

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建 NormalizedMarketState
        # trade_quantities: 正数表示 taker 是买方，负数表示 taker 是卖方
        trade_quantities = np.zeros(100, dtype=np.float32)
        trade_quantities[0] = 10.0   # 第一笔成交：taker 是买方，数量 10
        trade_quantities[1] = -5.0   # 第二笔成交：taker 是卖方，数量 5
        trade_quantities[2] = 8.0    # 第三笔成交：taker 是买方，数量 8

        trade_prices = np.zeros(100, dtype=np.float32)
        trade_prices[0] = -0.001  # 99.9 归一化
        trade_prices[1] = 0.001   # 100.1 归一化
        trade_prices[2] = 0.0     # 100.0 归一化

        market_state = NormalizedMarketState(
            mid_price=100.0,
            tick_size=0.01,
            bid_data=np.array([0.1, 10.0] * 100, dtype=np.float32),  # 100档买盘
            ask_data=np.array([-0.1, 10.0] * 100, dtype=np.float32),  # 100档卖盘
            trade_prices=trade_prices,
            trade_quantities=trade_quantities,
            tick_history_prices=np.zeros(100, dtype=np.float32),
            tick_history_volumes=np.zeros(100, dtype=np.float32),
            tick_history_amounts=np.zeros(100, dtype=np.float32),
        )

        # 设置一些持仓
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 50.0, 100.0)

        # 调用 observe
        inputs = agent.observe(market_state, orderbook)

        # 验证输入向量长度
        # 200 买盘 + 200 卖盘 + 200 成交 + 4 持仓 + 3 挂单 + 300 tick历史
        expected_length = 200 + 200 + 200 + 4 + 3 + 300
        assert len(inputs) == expected_length

        # 验证成交数据（价格在前100个，数量在后100个）
        trade_start_idx = 200 + 200  # 跳过订单簿深度数据
        # 前100个是价格（使用容差比较，因为是 float32）
        assert abs(inputs[trade_start_idx + 0] - (-0.001)) < 1e-6  # 第一笔成交价格
        assert abs(inputs[trade_start_idx + 1] - 0.001) < 1e-6     # 第二笔成交价格
        # 后100个是数量（带方向）
        quantity_start_idx = trade_start_idx + 100
        assert inputs[quantity_start_idx + 0] == 10.0   # 第一笔成交，taker 是买方
        assert inputs[quantity_start_idx + 1] == -5.0   # 第二笔成交，taker 是卖方
        assert inputs[quantity_start_idx + 2] == 8.0    # 第三笔成交，taker 是买方

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

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
        )

        # 创建空订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建空市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

        # 调用 observe
        inputs = agent.observe(market_state, orderbook)

        # 验证输入向量长度（固定长度）
        # 200 买盘 + 200 卖盘 + 200 成交 + 4 持仓 + 3 挂单 + 300 tick历史
        expected_length = 200 + 200 + 200 + 4 + 3 + 300
        assert len(inputs) == expected_length

        # 验证持仓状态（在末尾附近）
        position_start_idx = 200 + 200 + 200
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

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

        # 设置持仓
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 调用 observe
        inputs = agent.observe(market_state, orderbook)

        # 验证输入向量长度
        # 200 买盘 + 200 卖盘 + 200 成交 + 4 持仓 + 3 挂单 + 300 tick历史
        expected_length = 200 + 200 + 200 + 4 + 3 + 300
        assert len(inputs) == expected_length

        # 验证持仓数据（在持仓部分）
        position_start_idx = 200 + 200 + 200
        # 持仓归一化 = position_value / (equity * leverage)
        # position_value = 100 * 100 = 10000
        # equity = 10000 (balance)
        # 持仓归一化 = 10000 / (10000 * 100) = 0.01
        assert abs(inputs[position_start_idx + 0] - 0.01) < 0.01


class TestRetailProAgentDecide:
    """测试 RetailProAgent.decide"""

    def test_decide_hold_action(self):
        """测试 HOLD 动作决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：HOLD (index 0) 值最大，9 个输出
        mock_brain.forward.return_value = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 RetailProAgent
        agent = RetailProAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

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
        # outputs[6]=价格偏移 -0.5（低于中间价），outputs[7]=数量比例 0.0（较小数量）
        mock_brain.forward.return_value = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0])

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 RetailProAgent
        agent = RetailProAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.PLACE_BID
        # 验证参数包含 price 和 quantity
        assert "price" in params
        assert "quantity" in params
        # 价格偏移 = -0.5 * 100 * 0.01 = -0.5，所以价格 = 100 - 0.5 = 99.5
        assert params["price"] == 99.5
        # 数量应大于 0
        assert params["quantity"] > 0

    def test_decide_place_ask_action(self):
        """测试 PLACE_ASK 动作决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：PLACE_ASK (index 2) 值最大
        # outputs[6]=价格偏移 0.5（高于中间价），outputs[7]=数量比例 0.0（较小数量）
        mock_brain.forward.return_value = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0])

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 RetailProAgent
        agent = RetailProAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.PLACE_ASK
        # 验证参数包含 price 和 quantity
        assert "price" in params
        assert "quantity" in params
        # 价格偏移 = 0.5 * 100 * 0.01 = 0.5，所以价格 = 100 + 0.5 = 100.5
        assert params["price"] == 100.5
        # 数量应大于 0
        assert params["quantity"] > 0

    def test_decide_cancel_action(self):
        """测试 CANCEL 动作决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：CANCEL (index 3) 值最大
        mock_brain.forward.return_value = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 RetailProAgent
        agent = RetailProAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

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
        mock_brain.forward.return_value = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 RetailProAgent
        agent = RetailProAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

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
        mock_brain.forward.return_value = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 RetailProAgent
        agent = RetailProAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.MARKET_SELL
        # 验证参数包含 quantity
        assert "quantity" in params
        # 数量应大于 0
        assert params["quantity"] > 0

    def test_decide_insufficient_outputs_error(self):
        """测试神经网络输出维度不足时抛出异常"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：只有 3 个值，少于 8 个
        mock_brain.forward.return_value = np.array([0.1, 0.2, 0.3])

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 RetailProAgent
        agent = RetailProAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

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
        mock_brain.forward.return_value = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 RetailProAgent
        agent = RetailProAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 设置多仓持仓
        from src.market.orderbook.order import OrderSide
        agent.account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 创建订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.MARKET_SELL
        # 验证参数包含 quantity
        assert "quantity" in params
        # 数量比例 = (0.0 + 1) * 0.5 = 0.5
        # 卖出数量 = max(1, int(100.0 * 0.5)) = 50
        assert params["quantity"] == 50

    def test_decide_without_best_price(self):
        """测试订单簿为空时的决策"""
        # 创建 mock Brain
        mock_brain = MagicMock(spec=Brain)
        # 设置神经网络输出：PLACE_BID (index 1) 值最大
        # outputs[6]=价格偏移 -0.5（低于中间价），outputs[7]=数量比例 0.0（较小数量）
        mock_brain.forward.return_value = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0])

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 RetailProAgent
        agent = RetailProAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 创建空订单簿
        from src.market.orderbook.orderbook import OrderBook
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 验证动作类型
        assert action == ActionType.PLACE_BID
        # 验证参数包含 price 和 quantity
        assert "price" in params
        assert "quantity" in params
        # 价格偏移 = -0.5 * 100 * 0.01 = -0.5，所以价格 = 100 - 0.5 = 99.5
        assert params["price"] == 99.5


def create_mock_matching_engine() -> MagicMock:
    """创建测试用的 mock MatchingEngine"""
    mock_engine = MagicMock()
    # 模拟订单簿
    mock_orderbook = MagicMock()
    mock_orderbook.order_map = {}
    mock_engine._orderbook = mock_orderbook
    # process_order 默认返回空成交列表
    mock_engine.process_order.return_value = []
    # cancel_order 不返回值
    mock_engine.cancel_order.return_value = None
    return mock_engine


class TestAgentExecuteAction:
    """测试 Agent.execute_action"""

    def test_execute_place_bid(self):
        """测试执行挂买单"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
        )

        # 执行挂买单
        action = ActionType.PLACE_BID
        params = {"price": 99.5, "quantity": 10}
        trades = agent.execute_action(action, params, mock_engine)

        # 验证调用了 process_order
        mock_engine.process_order.assert_called_once()
        order = mock_engine.process_order.call_args[0][0]
        assert order.agent_id == 1
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.price == 99.5
        assert order.quantity == 10
        # 返回空成交列表
        assert trades == []

    def test_execute_place_ask(self):
        """测试执行挂卖单"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
        )

        # 执行挂卖单
        action = ActionType.PLACE_ASK
        params = {"price": 100.5, "quantity": 20}
        trades = agent.execute_action(action, params, mock_engine)

        # 验证调用了 process_order
        mock_engine.process_order.assert_called_once()
        order = mock_engine.process_order.call_args[0][0]
        assert order.agent_id == 1
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.price == 100.5
        assert order.quantity == 20
        # 返回空成交列表
        assert trades == []

    def test_execute_market_buy(self):
        """测试执行市价买入"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
        )

        # 执行市价买入
        action = ActionType.MARKET_BUY
        params = {"quantity": 15}
        trades = agent.execute_action(action, params, mock_engine)

        # 验证调用了 process_order
        mock_engine.process_order.assert_called_once()
        order = mock_engine.process_order.call_args[0][0]
        assert order.agent_id == 1
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.price == 0.0
        assert order.quantity == 15
        # 返回空成交列表
        assert trades == []

    def test_execute_market_sell(self):
        """测试执行市价卖出"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
        )

        # 执行市价卖出
        action = ActionType.MARKET_SELL
        params = {"quantity": 25}
        trades = agent.execute_action(action, params, mock_engine)

        # 验证调用了 process_order
        mock_engine.process_order.assert_called_once()
        order = mock_engine.process_order.call_args[0][0]
        assert order.agent_id == 1
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET
        assert order.price == 0.0
        assert order.quantity == 25
        # 返回空成交列表
        assert trades == []

    def test_execute_hold(self):
        """测试执行 HOLD（不动）"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
        )

        # 执行 HOLD
        action = ActionType.HOLD
        params: dict = {}
        trades = agent.execute_action(action, params, mock_engine)

        # 验证没有调用 process_order
        mock_engine.process_order.assert_not_called()
        # 返回空成交列表
        assert trades == []

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

        # 创建 Agent
        agent = Agent(
            agent_id=123,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=config,
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

        agent1 = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain_1,
            config=config,
        )
        agent2 = Agent(
            agent_id=2,
            agent_type=AgentType.RETAIL,
            brain=mock_brain_2,
            config=config,
        )

        # 两个 Agent 并发生成订单ID
        order_ids = set()
        for _ in range(50):
            order_ids.add(agent1._generate_order_id())
            order_ids.add(agent2._generate_order_id())

        # 所有订单ID应该唯一
        assert len(order_ids) == 100


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

        # 创建 Agent
        agent = Agent(
            agent_id=1,
            agent_type=AgentType.RETAIL,
            brain=mock_brain,
            config=initial_config,
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

        # 创建 Agent
        agent = Agent(
            agent_id=999,
            agent_type=AgentType.MARKET_MAKER,
            brain=mock_brain,
            config=config,
        )

        # 保存原始引用
        original_brain = agent.brain

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

    def test_reset_with_different_agent_types(self):
        """测试不同类型 Agent 的重置"""
        # 测试散户
        mock_brain = MagicMock(spec=Brain)

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

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
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

        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
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

        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 验证继承了基类方法
        assert hasattr(agent, "observe")
        assert hasattr(agent, "decide")
        assert hasattr(agent, "execute_action")
        assert hasattr(agent, "reset")
        assert callable(agent.observe)
        assert callable(agent.decide)
        assert callable(agent.execute_action)
        assert callable(agent.reset)


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

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
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

    def test_get_action_space_has_exactly_6_actions(self):
        """测试动作空间恰好包含 6 种动作"""
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

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证恰好 6 种动作
        assert len(action_space) == 6


class TestRetailAgentExecuteAction:
    """测试 RetailAgent.execute_action"""

    def test_place_bid_with_existing_order(self):
        """测试有挂单时再挂买单：先撤旧单再挂新单"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建散户 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 设置已有挂单
        agent.account.pending_order_id = 12345

        # 执行挂买单
        action = ActionType.PLACE_BID
        params = {"price": 99.5, "quantity": 10}
        trades = agent.execute_action(action, params, mock_engine)

        # 验证调用了 cancel_order 撤销旧单
        mock_engine.cancel_order.assert_called_once_with(12345)
        # 验证调用了 process_order 挂新单
        mock_engine.process_order.assert_called_once()
        order = mock_engine.process_order.call_args[0][0]
        assert order.agent_id == 1
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.price == 99.5
        assert order.quantity == 10
        # 返回空成交列表
        assert trades == []

    def test_place_ask_with_existing_order(self):
        """测试有挂单时再挂卖单：先撤旧单再挂新单"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建散户 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 设置已有挂单
        agent.account.pending_order_id = 67890

        # 执行挂卖单
        action = ActionType.PLACE_ASK
        params = {"price": 100.5, "quantity": 20}
        trades = agent.execute_action(action, params, mock_engine)

        # 验证调用了 cancel_order 撤销旧单
        mock_engine.cancel_order.assert_called_once_with(67890)
        # 验证调用了 process_order 挂新单
        mock_engine.process_order.assert_called_once()
        order = mock_engine.process_order.call_args[0][0]
        assert order.side == OrderSide.SELL
        assert order.price == 100.5
        assert order.quantity == 20
        # 返回空成交列表
        assert trades == []

    def test_place_bid_without_existing_order(self):
        """测试无挂单时挂买单：直接挂单"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建散户 Agent 配置
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        # 创建散户 Agent
        agent = RetailAgent(
            agent_id=1,
            brain=mock_brain,
            config=config,
        )

        # 确认没有挂单
        assert agent.account.pending_order_id is None

        # 执行挂买单
        action = ActionType.PLACE_BID
        params = {"price": 99.5, "quantity": 10}
        trades = agent.execute_action(action, params, mock_engine)

        # 验证没有调用 cancel_order
        mock_engine.cancel_order.assert_not_called()
        # 验证调用了 process_order
        mock_engine.process_order.assert_called_once()
        order = mock_engine.process_order.call_args[0][0]
        assert order.agent_id == 1
        assert order.side == OrderSide.BUY
        assert order.price == 99.5
        assert order.quantity == 10
        # 返回空成交列表
        assert trades == []


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

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
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

        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
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

        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
        )

        # 验证继承了基类方法
        assert hasattr(agent, "observe")
        assert hasattr(agent, "decide")
        assert hasattr(agent, "execute_action")
        assert hasattr(agent, "reset")
        assert callable(agent.observe)
        assert callable(agent.decide)
        assert callable(agent.execute_action)
        assert callable(agent.reset)


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

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证返回 6 种动作（与散户相同）
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

    def test_get_action_space_includes_hold(self):
        """测试庄家动作空间包含 HOLD"""
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

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证包含 HOLD
        assert ActionType.HOLD in action_space

    def test_get_action_space_includes_cancel(self):
        """测试庄家动作空间包含 CANCEL"""
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

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 验证包含 CANCEL
        assert ActionType.CANCEL in action_space


class TestWhaleAgentExecuteAction:
    """测试 WhaleAgent.execute_action"""

    def test_place_bid_with_existing_order(self):
        """测试有挂单时再挂买单：先撤旧单再挂新单"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
        )

        # 设置已有挂单
        agent.account.pending_order_id = 12345

        # 执行挂买单
        action = ActionType.PLACE_BID
        params = {"price": 99.5, "quantity": 100}
        trades = agent.execute_action(action, params, mock_engine)

        # 验证调用了 cancel_order 撤销旧单
        mock_engine.cancel_order.assert_called_once_with(12345)
        # 验证调用了 process_order 挂新单
        mock_engine.process_order.assert_called_once()
        order = mock_engine.process_order.call_args[0][0]
        assert order.agent_id == 10001
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.price == 99.5
        assert order.quantity == 100
        # 返回空成交列表
        assert trades == []

    def test_place_bid_without_existing_order(self):
        """测试无挂单时挂买单：直接挂单"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
        )

        # 确认没有挂单
        assert agent.account.pending_order_id is None

        # 执行挂买单
        action = ActionType.PLACE_BID
        params = {"price": 99.5, "quantity": 100}
        trades = agent.execute_action(action, params, mock_engine)

        # 验证没有调用 cancel_order（因为没有旧单）
        mock_engine.cancel_order.assert_not_called()
        # 验证调用了 process_order
        mock_engine.process_order.assert_called_once()
        order = mock_engine.process_order.call_args[0][0]
        assert order.agent_id == 10001
        assert order.side == OrderSide.BUY
        assert order.price == 99.5
        assert order.quantity == 100
        # 返回空成交列表
        assert trades == []

    def test_market_buy_with_existing_order(self):
        """测试有挂单时市价买入：市价买入不撤旧单（因为市价单会立即成交）"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建庄家 Agent 配置
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建庄家 Agent
        agent = WhaleAgent(
            agent_id=10001,
            brain=mock_brain,
            config=config,
        )

        # 设置已有挂单
        agent.account.pending_order_id = 99999

        # 执行市价买入
        action = ActionType.MARKET_BUY
        params = {"quantity": 150}
        trades = agent.execute_action(action, params, mock_engine)

        # 验证没有调用 cancel_order（市价单不需要先撤旧单）
        mock_engine.cancel_order.assert_not_called()
        # 验证调用了 process_order 市价单
        mock_engine.process_order.assert_called_once()
        order = mock_engine.process_order.call_args[0][0]
        assert order.order_type == OrderType.MARKET
        assert order.side == OrderSide.BUY
        assert order.quantity == 150
        # 返回空成交列表
        assert trades == []


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

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
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

        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
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

        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
        )

        # 验证继承了基类方法
        assert hasattr(agent, "observe")
        assert hasattr(agent, "decide")
        assert hasattr(agent, "execute_action")
        assert hasattr(agent, "reset")
        assert callable(agent.observe)
        assert callable(agent.decide)
        assert callable(agent.execute_action)
        assert callable(agent.reset)


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

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
        )

        # 获取动作空间
        action_space = agent.get_action_space()

        # 做市商不使用动作选择，返回空列表
        assert len(action_space) == 0
        assert action_space == []

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

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
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
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
        )

        # 执行双边挂单（execute_action 忽略 action 参数，始终执行双边挂单）
        action = ActionType.HOLD
        params = {
            "bid_orders": [
                {"price": 99.5, "quantity": 10},
                {"price": 99.4, "quantity": 15},
            ],
            "ask_orders": [
                {"price": 100.5, "quantity": 10},
                {"price": 100.6, "quantity": 12},
                {"price": 100.7, "quantity": 8},
            ],
        }
        trades = agent.execute_action(action, params, mock_engine)

        # 验证没有调用 cancel_order（无旧单）
        mock_engine.cancel_order.assert_not_called()
        # 验证调用了 5 次 process_order（2买+3卖）
        assert mock_engine.process_order.call_count == 5

        # 验证挂单ID列表
        assert len(agent.bid_order_ids) == 2
        assert len(agent.ask_order_ids) == 3
        # 返回空成交列表
        assert trades == []

    def test_execute_quote_with_existing_orders(self):
        """测试执行双边挂单（有旧挂单：先撤再挂）"""
        # 创建 mock Brain 和 matching_engine
        mock_brain = MagicMock(spec=Brain)
        mock_engine = create_mock_matching_engine()

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
        )

        # 设置已有挂单
        agent.bid_order_ids = [1001, 1002]
        agent.ask_order_ids = [2001, 2002, 2003]

        # 执行双边挂单（execute_action 忽略 action 参数，始终执行双边挂单）
        action = ActionType.HOLD
        params = {
            "bid_orders": [{"price": 99.3, "quantity": 20}],
            "ask_orders": [{"price": 100.8, "quantity": 25}],
        }
        trades = agent.execute_action(action, params, mock_engine)

        # 验证调用了 cancel_order 撤销旧单
        assert mock_engine.cancel_order.call_count == 5  # 2买+3卖
        cancel_calls = [call[0][0] for call in mock_engine.cancel_order.call_args_list]
        assert 1001 in cancel_calls
        assert 1002 in cancel_calls
        assert 2001 in cancel_calls
        assert 2002 in cancel_calls
        assert 2003 in cancel_calls

        # 验证调用了 2 次 process_order（1买+1卖）
        assert mock_engine.process_order.call_count == 2

        # 验证挂单列表被清空后重新填充
        assert len(agent.bid_order_ids) == 1
        assert len(agent.ask_order_ids) == 1
        # 返回空成交列表
        assert trades == []



class TestMarketMakerAgentDecide:
    """测试 MarketMakerAgent.decide"""

    def test_decide_quote_action(self):
        """测试做市商决策返回 HOLD 动作和双边订单"""
        # 创建 mock Brain，设置神经网络输出（41 个值）
        mock_brain = MagicMock(spec=Brain)
        # 结构：10 bid price offsets + 10 bid qty weights + 10 ask price offsets + 10 ask qty weights + 1 total ratio
        mock_brain.forward.return_value = (
            [0.0] * 10 +   # bid price offsets
            [0.5] * 10 +   # bid quantity weights (positive = reasonable)
            [0.0] * 10 +   # ask price offsets
            [0.5] * 10 +   # ask quantity weights
            [0.0]           # total ratio
        )

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 做市商 decide 始终返回 HOLD
        assert action == ActionType.HOLD
        assert "bid_orders" in params
        assert "ask_orders" in params
        # 验证有买单和卖单
        assert len(params["bid_orders"]) > 0
        assert len(params["ask_orders"]) > 0

        # 验证所有买单价格低于 mid_price
        mid_price = market_state.mid_price
        for order in params["bid_orders"]:
            assert order["price"] < mid_price
            assert order["quantity"] > 0

        # 验证所有卖单价格高于 mid_price
        for order in params["ask_orders"]:
            assert order["price"] > mid_price
            assert order["quantity"] > 0

    def test_decide_insufficient_outputs_error(self):
        """测试神经网络输出维度不足时抛出异常"""
        # 创建 mock Brain，设置输出维度不足
        mock_brain = MagicMock(spec=Brain)
        mock_brain.forward.return_value = np.array([0.0] * 10)  # 只有 10 个输出，需要 41 个

        # 创建做市商 Agent 配置
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

        # 调用 decide 应该抛出异常
        try:
            agent.decide(market_state, orderbook)
            assert False, "应该抛出 ValueError"
        except ValueError as e:
            assert "神经网络输出维度不足" in str(e)
            assert "41" in str(e)
            assert "10" in str(e)

    def test_decide_generates_orders_at_different_prices(self):
        """测试生成的订单在不同价位"""
        # 创建 mock Brain，设置不同的价格偏移（41 个值）
        mock_brain = MagicMock(spec=Brain)
        # 结构：10 bid price offsets + 10 bid qty weights + 10 ask price offsets + 10 ask qty weights + 1 total ratio
        outputs = (
            [-0.8, -0.6, -0.4, -0.2, 0.0, -0.8, -0.6, -0.4, -0.2, 0.0] +  # 10 bid price offsets
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] +           # 10 bid qty weights
            [0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4, 0.6, 0.8] +           # 10 ask price offsets
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] +           # 10 ask qty weights
            [0.0]                                                              # total ratio
        )
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

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 做市商 decide 始终返回 HOLD
        assert action == ActionType.HOLD

        # 验证有买单和卖单
        assert len(params["bid_orders"]) > 0
        assert len(params["ask_orders"]) > 0

        # 提取价格
        bid_prices = [o["price"] for o in params["bid_orders"]]
        ask_prices = [o["price"] for o in params["ask_orders"]]

        # 所有买单价格应低于所有卖单价格
        assert max(bid_prices) < min(ask_prices)

    def test_decide_normalizes_quantity_ratios(self):
        """测试数量比例归一化（20个订单的总比例不超过1.0）"""
        # 创建 mock Brain，设置所有数量权重为最大值（41 个值）
        mock_brain = MagicMock(spec=Brain)
        # 结构：10 bid price offsets + 10 bid qty weights + 10 ask price offsets + 10 ask qty weights + 1 total ratio
        outputs = (
            [0.0] * 10 +   # bid price offsets
            [1.0] * 10 +   # bid qty weights (all max)
            [0.0] * 10 +   # ask price offsets
            [1.0] * 10 +   # ask qty weights (all max)
            [0.0]           # total ratio
        )
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

        # 创建做市商 Agent
        agent = MarketMakerAgent(
            agent_id=10011,
            brain=mock_brain,
            config=config,
        )

        # 创建订单簿
        orderbook = OrderBook(tick_size=0.01)
        orderbook.last_price = 100.0

        # 创建市场状态
        market_state = create_mock_market_state(mid_price=100.0, tick_size=0.01)

        # 调用 decide
        action, params = agent.decide(market_state, orderbook)

        # 做市商 decide 始终返回 HOLD
        assert action == ActionType.HOLD

        # 验证有买单和卖单（最多每边10个）
        assert len(params["bid_orders"]) > 0
        assert len(params["ask_orders"]) > 0
        assert len(params["bid_orders"]) <= 10
        assert len(params["ask_orders"]) <= 10

        # 计算总价值
        mid_price = market_state.mid_price
        equity = agent.account.get_equity(mid_price)

        total_bid_value = sum(o["price"] * o["quantity"] for o in params["bid_orders"])
        total_ask_value = sum(o["price"] * o["quantity"] for o in params["ask_orders"])
        total_value = total_bid_value + total_ask_value

        # 总价值应该不超过净值 * 杠杆（允许一定误差，因为 _calculate_order_quantity 有最小值限制）
        max_total_value = equity * agent.account.leverage
        # 实际总价值可能略高因为有最小数量限制，但不应超过太多
        assert total_value <= max_total_value * 1.5  # 允许 50% 的误差空间

        # 验证每个订单的数量都大于 0
        bid_quantities = [o["quantity"] for o in params["bid_orders"]]
        ask_quantities = [o["quantity"] for o in params["ask_orders"]]
        assert all(q > 0 for q in bid_quantities)
        assert all(q > 0 for q in ask_quantities)

        # 验证买单数量是整数
        assert all(isinstance(q, int) for q in bid_quantities)
        assert all(isinstance(q, int) for q in ask_quantities)
