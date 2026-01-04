"""散户 Agent 模块

本模块定义散户 Agent 类，继承自 RetailProAgent（高级散户）。
散户与高级散户共享相同的动作空间和决策逻辑，仅在 observe 方法上有所不同：
- 散户只能看到买卖各10档订单簿和最近10笔成交
- 高级散户可以看到完整的100档订单簿和100笔成交
"""

import numpy as np

from src.bio.agents.retail_pro import RetailProAgent
from src.config.config import AgentConfig, AgentType
from src.bio.brain.brain import Brain
from src.market.market_state import NormalizedMarketState
from src.market.orderbook.orderbook import OrderBook


class RetailAgent(RetailProAgent):
    """散户 Agent

    代表市场中数量最多的交易参与者，初始资产较少，杠杆倍数最高。
    散户只能看到买卖各10档订单簿和最近10笔成交。

    继承自 RetailProAgent，共享相同的 decide 和 execute_action 方法，
    仅重写 observe 方法以限制可见的市场数据。

    Attributes:
        agent_id: Agent ID
        brain: NEAT 神经网络
        account: 交易账户
    """

    # 散户可见的订单簿档位数和成交笔数
    ORDERBOOK_DEPTH: int = 10
    TRADE_HISTORY_SIZE: int = 10
    # 输入缓冲区大小: 10档买盘(20) + 10档卖盘(20) + 10笔成交价格(10) + 10笔成交数量(10) + 持仓(4) + 挂单(3) = 67
    INPUT_SIZE: int = 67

    agent_id: int
    brain: Brain

    def __init__(self, agent_id: int, brain: Brain, config: AgentConfig) -> None:
        """创建散户 Agent

        调用父类构造函数，但设置类型为 RETAIL（覆盖 RetailProAgent 的 RETAIL_PRO 类型）。
        重写输入缓冲区为更小的尺寸（67），因为散户只能看到有限的市场数据。

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
        """
        # 调用祖父类 Agent 的构造函数，设置类型为 RETAIL
        from src.bio.agents.base import Agent
        Agent.__init__(self, agent_id, AgentType.RETAIL, brain, config)
        # 覆盖父类的输入缓冲区，使用更小的尺寸
        self._input_buffer = np.zeros(self.INPUT_SIZE, dtype=np.float64)

    def observe(self, market_state: NormalizedMarketState, orderbook: OrderBook) -> np.ndarray:
        """从预计算的市场状态构建神经网络输入（散户限制版）

        散户只能看到买卖各10档订单簿和最近10笔成交。

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿（用于查询挂单信息）

        Returns:
            神经网络输入向量（67维 ndarray）
        """
        depth = self.ORDERBOOK_DEPTH
        trade_size = self.TRADE_HISTORY_SIZE

        # 买盘前10档: 每档2个值（价格归一化 + 数量），取前20个值
        self._input_buffer[:depth * 2] = market_state.bid_data[:depth * 2]

        # 卖盘前10档: 每档2个值，取前20个值
        offset = depth * 2  # 20
        self._input_buffer[offset:offset + depth * 2] = market_state.ask_data[:depth * 2]

        # 最近10笔成交价格
        offset += depth * 2  # 40
        self._input_buffer[offset:offset + trade_size] = market_state.trade_prices[:trade_size]

        # 最近10笔成交数量
        offset += trade_size  # 50
        self._input_buffer[offset:offset + trade_size] = market_state.trade_quantities[:trade_size]

        # 持仓信息（4个值）
        offset += trade_size  # 60
        self._input_buffer[offset:offset + 4] = self._get_position_inputs(market_state.mid_price)

        # 挂单信息（3个值）
        offset += 4  # 64
        self._input_buffer[offset:offset + 3] = self._get_pending_order_inputs(market_state.mid_price, orderbook)

        return self._input_buffer  # 不调用 .tolist()
