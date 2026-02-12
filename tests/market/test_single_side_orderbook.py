"""测试订单簿只有买盘或卖盘时的边界情况

验证当订单簿只有买盘或只有卖盘时，Agent 能否正常推理。
"""

import numpy as np
import pytest

from src.bio.agents.retail import RetailAgent
from src.bio.agents.retail_pro import RetailProAgent
from src.bio.agents.whale import WhaleAgent
from src.bio.agents.base import ActionType
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType
from src.market.market_state import NormalizedMarketState
from src.market.orderbook.orderbook import OrderBook
from src.market.orderbook.order import Order, OrderSide, OrderType


def create_test_agent(agent_class, agent_id: int = 0):
    """创建测试用的 Agent"""
    from unittest.mock import MagicMock

    # 创建 mock Brain
    mock_brain = MagicMock()

    # 设置 mock 返回值
    mock_brain.forward.return_value = np.array([0, 0, 0, 0, 0, 0, 0.0, 0.0], dtype=np.float64)  # 6个动作得分 + 价格偏移 + 数量比例

    # 创建 Agent 配置（注意：参数名与现有测试保持一致）
    config = AgentConfig(
        count=1,
        initial_balance=100_000.0,
        leverage=100.0,
        maintenance_margin_rate=0.005,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
    )

    # RetailAgent 等子类的构造函数不需要 agent_type 参数（内部硬编码）
    return agent_class(agent_id, mock_brain, config)


def create_market_state_with_only_bids():
    """创建只有买盘的市场状态"""
    # 买盘数据：10档买单，从99.0向下
    bid_data = np.zeros(200, dtype=np.float32)
    for i in range(10):
        price = 99.0 - i * 1.0  # 99, 98, 97, ..., 90
        quantity = 100 + i * 10  # 100, 110, 120, ...
        bid_data[i * 2] = (price - 100.0) / 100.0  # 价格归一化
        bid_data[i * 2 + 1] = np.log10(quantity + 1) / 10.0  # 数量归一化

    # 卖盘数据：全空
    ask_data = np.zeros(200, dtype=np.float32)

    return NormalizedMarketState(
        mid_price=100.0,
        tick_size=0.01,
        bid_data=bid_data,
        ask_data=ask_data,
        trade_prices=np.zeros(100, dtype=np.float32),
        trade_quantities=np.zeros(100, dtype=np.float32),
        tick_history_prices=np.zeros(100, dtype=np.float32),
        tick_history_volumes=np.zeros(100, dtype=np.float32),
        tick_history_amounts=np.zeros(100, dtype=np.float32),
    )


def create_market_state_with_only_asks():
    """创建只有卖盘的市场状态"""
    # 买盘数据：全空
    bid_data = np.zeros(200, dtype=np.float32)

    # 卖盘数据：10档卖单，从101.0到110.0
    ask_data = np.zeros(200, dtype=np.float32)
    for i in range(10):
        price = 101.0 + i * 1.0  # 101, 102, 103, ...
        quantity = 100 + i * 10  # 100, 110, 120, ...
        ask_data[i * 2] = (price - 100.0) / 100.0  # 价格归一化
        ask_data[i * 2 + 1] = np.log10(quantity + 1) / 10.0  # 数量归一化

    return NormalizedMarketState(
        mid_price=100.0,
        tick_size=0.01,
        bid_data=bid_data,
        ask_data=ask_data,
        trade_prices=np.zeros(100, dtype=np.float32),
        trade_quantities=np.zeros(100, dtype=np.float32),
        tick_history_prices=np.zeros(100, dtype=np.float32),
        tick_history_volumes=np.zeros(100, dtype=np.float32),
        tick_history_amounts=np.zeros(100, dtype=np.float32),
    )


def create_market_state_empty():
    """创建完全空的市场状态（买卖盘都为空）"""
    return NormalizedMarketState(
        mid_price=100.0,
        tick_size=0.01,
        bid_data=np.zeros(200, dtype=np.float32),
        ask_data=np.zeros(200, dtype=np.float32),
        trade_prices=np.zeros(100, dtype=np.float32),
        trade_quantities=np.zeros(100, dtype=np.float32),
        tick_history_prices=np.zeros(100, dtype=np.float32),
        tick_history_volumes=np.zeros(100, dtype=np.float32),
        tick_history_amounts=np.zeros(100, dtype=np.float32),
    )


class TestSingleSideOrderbook:
    """测试单边订单簿情况"""

    def test_retail_agent_with_only_bids(self):
        """测试散户在只有买盘时能否正常推理"""
        agent = create_test_agent(RetailAgent)
        market_state = create_market_state_with_only_bids()
        orderbook = OrderBook(tick_size=0.01)

        # 测试 observe 方法是否正常工作
        inputs = agent.observe(market_state, orderbook)

        # 验证输入维度
        assert inputs.shape == (127,), f"期望输入维度127，实际{inputs.shape}"

        # 验证买盘数据被正确填充（前20个值）
        assert inputs[0] != 0.0, "买盘价格应该被填充"
        assert inputs[1] != 0.0, "买盘数量应该被填充"

        # 验证卖盘数据为空（第20-39个值）
        assert np.all(inputs[20:40] == 0.0), "卖盘数据应该全为0"

        # 测试 decide 方法是否正常工作
        action, params = agent.decide(market_state, orderbook)

        # 验证返回了有效的动作
        assert action is not None
        assert isinstance(params, dict)

    def test_retail_agent_with_only_asks(self):
        """测试散户在只有卖盘时能否正常推理"""
        agent = create_test_agent(RetailAgent)
        market_state = create_market_state_with_only_asks()
        orderbook = OrderBook(tick_size=0.01)

        # 测试 observe 方法是否正常工作
        inputs = agent.observe(market_state, orderbook)

        # 验证输入维度
        assert inputs.shape == (127,), f"期望输入维度127，实际{inputs.shape}"

        # 验证买盘数据为空（前20个值）
        assert np.all(inputs[0:20] == 0.0), "买盘数据应该全为0"

        # 验证卖盘数据被正确填充（第20-39个值）
        assert inputs[20] != 0.0, "卖盘价格应该被填充"
        assert inputs[21] != 0.0, "卖盘数量应该被填充"

        # 测试 decide 方法是否正常工作
        action, params = agent.decide(market_state, orderbook)

        # 验证返回了有效的动作
        assert action is not None
        assert isinstance(params, dict)

    def test_retail_agent_with_empty_orderbook(self):
        """测试散户在完全空的订单簿时能否正常推理"""
        agent = create_test_agent(RetailAgent)
        market_state = create_market_state_empty()
        orderbook = OrderBook(tick_size=0.01)

        # 测试 observe 方法是否正常工作
        inputs = agent.observe(market_state, orderbook)

        # 验证输入维度
        assert inputs.shape == (127,), f"期望输入维度127，实际{inputs.shape}"

        # 验证买卖盘数据都为空
        assert np.all(inputs[0:40] == 0.0), "买卖盘数据应该全为0"

        # 测试 decide 方法是否正常工作
        action, params = agent.decide(market_state, orderbook)

        # 验证返回了有效的动作
        assert action is not None
        assert isinstance(params, dict)

    def test_retail_pro_agent_with_only_bids(self):
        """测试高级散户在只有买盘时能否正常推理"""
        agent = create_test_agent(RetailProAgent)
        market_state = create_market_state_with_only_bids()
        orderbook = OrderBook(tick_size=0.01)

        # 测试 observe 方法是否正常工作
        inputs = agent.observe(market_state, orderbook)

        # 验证输入维度（高级散户使用完整的607维输入）
        assert inputs.shape == (907,), f"期望输入维度907，实际{inputs.shape}"

        # 验证买盘数据被正确填充
        assert inputs[0] != 0.0, "买盘价格应该被填充"
        assert inputs[1] != 0.0, "买盘数量应该被填充"

        # 验证卖盘数据为空
        assert np.all(inputs[200:400] == 0.0), "卖盘数据应该全为0"

        # 测试 decide 方法是否正常工作
        action, params = agent.decide(market_state, orderbook)
        assert action is not None

    def test_whale_agent_with_only_asks(self):
        """测试庄家在只有卖盘时能否正常推理"""
        agent = create_test_agent(WhaleAgent)
        market_state = create_market_state_with_only_asks()
        orderbook = OrderBook(tick_size=0.01)

        # 测试 observe 方法是否正常工作
        inputs = agent.observe(market_state, orderbook)

        # 验证输入维度（庄家使用完整的607维输入）
        assert inputs.shape == (907,), f"期望输入维度907，实际{inputs.shape}"

        # 验证买盘数据为空
        assert np.all(inputs[0:200] == 0.0), "买盘数据应该全为0"

        # 验证卖盘数据被正确填充
        assert inputs[200] != 0.0, "卖盘价格应该被填充"
        assert inputs[201] != 0.0, "卖盘数量应该被填充"

        # 测试 decide 方法是否正常工作
        action, params = agent.decide(market_state, orderbook)
        assert action is not None

    def test_neural_network_forward_with_zeros(self):
        """测试神经网络在输入包含大量零值时能否正常前向传播"""
        agent = create_test_agent(RetailAgent)
        market_state = create_market_state_empty()
        orderbook = OrderBook(tick_size=0.01)

        # 多次测试确保稳定性
        for _ in range(10):
            inputs = agent.observe(market_state, orderbook)
            outputs = agent.brain.forward(inputs)

            # 验证输出维度（8个值：6个动作 + 价格偏移 + 数量比例）
            assert len(outputs) == 8, f"期望输出维度8，实际{len(outputs)}"

            # 验证输出都是数值（不是 NaN 或 Inf）
            assert np.all(np.isfinite(outputs)), "神经网络输出包含 NaN 或 Inf"

    def test_price_protection_edge_case(self):
        """测试价格保护在极端 mid_price 时的行为"""
        agent = create_test_agent(RetailAgent)

        # 创建 mid_price 非常小的市场状态
        market_state = NormalizedMarketState(
            mid_price=0.001,  # 极小的中间价
            tick_size=0.01,
            bid_data=np.zeros(200, dtype=np.float32),
            ask_data=np.zeros(200, dtype=np.float32),
            trade_prices=np.zeros(100, dtype=np.float32),
            trade_quantities=np.zeros(100, dtype=np.float32),
            tick_history_prices=np.zeros(100, dtype=np.float32),
            tick_history_volumes=np.zeros(100, dtype=np.float32),
            tick_history_amounts=np.zeros(100, dtype=np.float32),
        )
        orderbook = OrderBook(tick_size=0.01)

        # 神经网络应该仍然能正常推理
        action, params = agent.decide(market_state, orderbook)

        assert action is not None

        # 如果是 PLACE_BID 或 PLACE_ASK，验证价格保护
        if action in [ActionType.PLACE_BID, ActionType.PLACE_ASK]:
            price = params.get("price", 0)
            # 价格应该至少是 tick_size
            assert price >= 0.01, f"价格 {price} 应该至少为 tick_size (0.01)"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
