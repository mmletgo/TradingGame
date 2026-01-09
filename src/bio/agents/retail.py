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
from src.market.orderbook.order import OrderSide

# 尝试导入 Cython 加速的 observe 函数
try:
    from src.bio.agents._cython.fast_observe import (
        fast_observe_retail,
        get_position_inputs as cython_get_position_inputs,
        get_pending_order_inputs as cython_get_pending_order_inputs,
    )
    _HAS_CYTHON_OBSERVE = True
except ImportError:
    _HAS_CYTHON_OBSERVE = False


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
    TICK_HISTORY_SIZE: int = 20  # 新增：散户可见的 tick 历史数量
    # 输入缓冲区大小: 10档买盘(20) + 10档卖盘(20) + 10笔成交价格(10) + 10笔成交数量(10)
    #               + 持仓(4) + 挂单(3) + tick历史价格(20) + tick历史成交量(20) + tick历史成交额(20) = 127
    INPUT_SIZE: int = 127

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
            神经网络输入向量（127维 ndarray）
        """
        if _HAS_CYTHON_OBSERVE:
            # 使用 Cython 加速版本
            mid_price = market_state.mid_price
            equity = self.account.get_equity(mid_price)
            position_inputs = cython_get_position_inputs(
                equity,
                self.account.leverage,
                self.account.position.quantity,
                self.account.position.avg_price,
                self.account.balance,
                self.account.initial_balance,
                mid_price,
            )

            # 获取挂单信息
            pending_id = self.account.pending_order_id
            if pending_id is not None:
                order = orderbook.order_map.get(pending_id)
                if order is not None:
                    pending_inputs = cython_get_pending_order_inputs(
                        order.price,
                        order.quantity,
                        1 if order.side == OrderSide.BUY else 2,
                        mid_price,
                    )
                else:
                    pending_inputs = (0.0, 0.0, 0.0)
            else:
                pending_inputs = (0.0, 0.0, 0.0)

            # 调用 Cython 函数填充缓冲区
            fast_observe_retail(
                self._input_buffer,
                market_state.bid_data,
                market_state.ask_data,
                market_state.trade_prices,
                market_state.trade_quantities,
                market_state.tick_history_prices,
                market_state.tick_history_volumes,
                market_state.tick_history_amounts,
                position_inputs[0],
                position_inputs[1],
                position_inputs[2],
                position_inputs[3],
                pending_inputs[0],
                pending_inputs[1],
                pending_inputs[2],
            )
            return self._input_buffer
        else:
            # 纯 Python 实现
            depth = self.ORDERBOOK_DEPTH
            trade_size = self.TRADE_HISTORY_SIZE
            tick_size = self.TICK_HISTORY_SIZE

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

            # tick 历史价格（最近20个）
            offset += 3  # 67
            self._input_buffer[offset:offset + tick_size] = market_state.tick_history_prices[-tick_size:]

            # tick 历史成交量（最近20个）
            offset += tick_size  # 87
            self._input_buffer[offset:offset + tick_size] = market_state.tick_history_volumes[-tick_size:]

            # tick 历史成交额（最近20个）
            offset += tick_size  # 107
            self._input_buffer[offset:offset + tick_size] = market_state.tick_history_amounts[-tick_size:]

            return self._input_buffer  # 不调用 .tolist()
