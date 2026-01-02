# /home/rongheng/python_project/TradingGame_ui_dev/src/ui/data_collector.py
"""UI数据采集模块

采集训练器数据并转换为UI展示所需的格式。
"""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.config.config import AgentType

if TYPE_CHECKING:
    from src.training.trainer import Trainer


@dataclass
class TradeInfo:
    """成交信息（UI展示用）"""

    tick: int
    price: float
    quantity: float
    is_buyer_taker: bool  # True=买入, False=卖出


@dataclass
class PopulationStats:
    """种群统计信息"""

    avg_equity: float
    total_equity: float  # 存活个体资产总和
    max_equity: float
    min_equity: float
    alive_count: int
    total_count: int
    generation: int
    alive_equities: list[float]  # 存活个体的资产列表（用于小提琴图）


@dataclass
class UIDataSnapshot:
    """UI数据快照 - 每个tick的完整数据"""

    tick: int
    episode: int

    # 价格
    last_price: float
    mid_price: float

    # 订单簿100档
    bids: list[tuple[float, float]]  # [(price, qty), ...]
    asks: list[tuple[float, float]]

    # 成交记录
    recent_trades: list[TradeInfo]

    # 种群统计
    population_stats: dict[AgentType, PopulationStats]

    # 历史数据（曲线用）
    price_history: list[float]
    equity_history: dict[AgentType, list[float]]


class UIDataCollector:
    """UI数据采集器

    每tick收集数据，维护历史缓冲区。

    Attributes:
        history_length: 历史数据长度限制
        price_history: 价格历史缓冲区
        equity_history: 各种群净值历史缓冲区
    """

    history_length: int
    price_history: deque[float]
    equity_history: dict[AgentType, deque[float]]

    def __init__(self, history_length: int = 1000) -> None:
        """初始化数据采集器

        Args:
            history_length: 历史数据长度限制，默认1000
        """
        self.history_length = history_length
        self.price_history = deque(maxlen=history_length)
        self.equity_history = {
            agent_type: deque(maxlen=history_length) for agent_type in AgentType
        }

    def collect_tick_data(self, trainer: "Trainer") -> UIDataSnapshot:
        """收集当前tick的数据快照

        从训练器中提取所有UI展示所需的数据，包括价格、订单簿、
        成交记录和种群统计信息。

        Args:
            trainer: 训练器实例

        Returns:
            UIDataSnapshot: 当前tick的完整数据快照
        """
        # 获取订单簿和价格
        orderbook = trainer.matching_engine._orderbook
        depth = orderbook.get_depth(100)
        last_price = orderbook.last_price
        mid_price = orderbook.get_mid_price()
        if mid_price is None:
            mid_price = last_price

        # 记录价格历史
        self.price_history.append(last_price)

        # 计算各种群统计（使用NumPy向量化）
        population_stats: dict[AgentType, PopulationStats] = {}
        for agent_type, population in trainer.populations.items():
            stats = self._compute_population_stats(population, last_price)
            population_stats[agent_type] = stats
            self.equity_history[agent_type].append(stats.total_equity)

        # 转换成交记录
        recent_trades: list[TradeInfo] = [
            TradeInfo(
                tick=trainer.tick,
                price=trade.price,
                quantity=trade.quantity,
                is_buyer_taker=trade.is_buyer_taker,
            )
            for trade in trainer.recent_trades
        ]

        return UIDataSnapshot(
            tick=trainer.tick,
            episode=trainer.episode,
            last_price=last_price,
            mid_price=mid_price,
            bids=depth["bids"],
            asks=depth["asks"],
            recent_trades=recent_trades,
            population_stats=population_stats,
            price_history=list(self.price_history),
            equity_history={k: list(v) for k, v in self.equity_history.items()},
        )

    def _compute_population_stats(
        self, population: "Population", current_price: float
    ) -> PopulationStats:
        """向量化计算种群统计

        使用NumPy向量化操作计算种群的平均净值、最大净值、
        最小净值、存活数量等统计信息。

        Args:
            population: 种群实例
            current_price: 当前价格

        Returns:
            PopulationStats: 种群统计信息
        """
        # 避免循环导入
        from src.training.population import Population

        agents = population.agents
        n = len(agents)

        if n == 0:
            return PopulationStats(
                avg_equity=0.0,
                total_equity=0.0,
                max_equity=0.0,
                min_equity=0.0,
                alive_count=0,
                total_count=0,
                generation=population.generation,
                alive_equities=[],
            )

        # 预分配数组
        equities = np.zeros(n, dtype=np.float64)
        alive_mask = np.ones(n, dtype=bool)

        for i, agent in enumerate(agents):
            equities[i] = agent.account.get_equity(current_price)
            alive_mask[i] = not agent.is_liquidated

        alive_equities = equities[alive_mask]
        alive_count = int(np.sum(alive_mask))

        return PopulationStats(
            avg_equity=float(np.mean(alive_equities)) if alive_count > 0 else 0.0,
            total_equity=float(np.sum(alive_equities)),
            max_equity=float(np.max(equities)),
            min_equity=float(np.min(equities)),
            alive_count=alive_count,
            total_count=n,
            generation=population.generation,
            alive_equities=alive_equities.tolist(),
        )

    def reset(self) -> None:
        """重置历史数据（新episode开始时调用）"""
        self.price_history.clear()
        for history in self.equity_history.values():
            history.clear()
