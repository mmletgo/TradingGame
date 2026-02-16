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
class NoiseTraderInfo:
    """噪声交易者信息（UI展示用）"""

    name: str  # 噪声交易者类型名称
    equity: float  # 净值
    position_qty: int  # 持仓数量
    position_value: float  # 持仓市值 = abs(position_qty) * current_price
    initial_balance: float  # 初始资金
    is_liquidated: bool  # 是否被强平


@dataclass
class PopulationStats:
    """种群统计信息"""

    avg_equity: float
    total_equity: float  # 所有个体资产总和
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
    equity_history: dict[AgentType, list[float]]  # 所有个体平均资产历史
    alive_equity_history: dict[AgentType, list[float]]  # 存活个体平均资产历史

    # 噪声交易者数据
    noise_trader_data: list[NoiseTraderInfo]  # 噪声交易者的当前数据
    noise_trader_equity_history: list[list[float]]  # 噪声交易者的净值历史


class UIDataCollector:
    """UI数据采集器

    每tick收集数据，维护历史缓冲区。

    Attributes:
        history_length: 历史数据长度限制
        price_history: 价格历史缓冲区
        equity_history: 各种群所有个体平均资产历史缓冲区
        alive_equity_history: 各种群存活个体平均资产历史缓冲区
    """

    history_length: int
    price_history: deque[float]
    equity_history: dict[AgentType, deque[float]]
    alive_equity_history: dict[AgentType, deque[float]]
    noise_trader_equity_history: list[deque[float]]  # 噪声交易者的净值历史

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
        self.alive_equity_history = {
            agent_type: deque(maxlen=history_length) for agent_type in AgentType
        }
        # 初始化噪声交易者的净值历史
        self.noise_trader_equity_history = [
            deque(maxlen=history_length) for _ in range(4)
        ]

    def collect_tick_data(self, trainer: "Trainer") -> UIDataSnapshot:
        """收集当前tick的数据快照

        从训练器中提取所有UI展示所需的数据，包括价格、订单簿、
        成交记录和种群统计信息。

        价格处理说明：
        - last_price: 使用 orderbook.last_price（tick 结束后的实际价格），用于价格图表显示
        - price_for_equity: 使用 tick_start_price（tick 开始时的价格），用于资产计算，
          确保与淘汰检查使用同一价格，避免出现"负资产回正"的显示异常

        Args:
            trainer: 训练器实例

        Returns:
            UIDataSnapshot: 当前tick的完整数据快照
        """
        # 获取订单簿和价格
        orderbook = trainer.matching_engine._orderbook
        depth = orderbook.get_depth(100)

        # 价格图表使用 tick 结束后的实际价格（与盘口一致）
        last_price = orderbook.last_price
        if last_price <= 0:
            last_price = trainer.tick_start_price  # 兼容初始化阶段
        if last_price <= 0:
            last_price = 100.0  # 最终兜底

        # 资产计算使用 tick 开始时的价格（与淘汰检查一致）
        price_for_equity = trainer.tick_start_price
        if price_for_equity <= 0:
            price_for_equity = last_price

        mid_price = orderbook.get_mid_price()
        if mid_price is None:
            mid_price = last_price

        # 记录价格历史（使用实际成交价）
        self.price_history.append(last_price)

        # 计算各种群统计（使用NumPy向量化）
        population_stats: dict[AgentType, PopulationStats] = {}
        for agent_type, population in trainer.populations.items():
            stats = self._compute_population_stats(population, price_for_equity)
            population_stats[agent_type] = stats
            # 计算所有个体平均资产
            avg_all = stats.total_equity / stats.total_count if stats.total_count > 0 else 0.0
            self.equity_history[agent_type].append(avg_all)
            # 计算存活个体平均资产
            alive_avg = stats.avg_equity  # avg_equity 已经是存活个体的平均值
            self.alive_equity_history[agent_type].append(alive_avg)

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

        # 收集噪声交易者数据
        noise_trader_data = self._collect_noise_trader_data(trainer, price_for_equity)

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
            alive_equity_history={k: list(v) for k, v in self.alive_equity_history.items()},
            noise_trader_data=noise_trader_data,
            noise_trader_equity_history=[list(h) for h in self.noise_trader_equity_history],
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
            total_equity=float(np.sum(equities)),  # 所有个体资产总和
            max_equity=float(np.max(equities)),
            min_equity=float(np.min(equities)),
            alive_count=alive_count,
            total_count=n,
            generation=population.generation,
            alive_equities=alive_equities.tolist(),
        )

    def _collect_noise_trader_data(
        self, trainer: "Trainer", current_price: float
    ) -> list[NoiseTraderInfo]:
        """收集噪声交易者数据

        Args:
            trainer: 训练器实例
            current_price: 当前价格（用于计算净值）

        Returns:
            噪声交易者信息列表
        """
        noise_trader_data: list[NoiseTraderInfo] = []

        noise_traders = getattr(trainer, "noise_traders", [])
        if not noise_traders:
            return noise_trader_data

        # 收集每个噪声交易者的数据
        for i, trader_obj in enumerate(noise_traders):
            name = type(trader_obj).__name__

            equity = trader_obj.account.get_equity(current_price)
            position_qty = trader_obj.account.position.quantity
            position_value = abs(position_qty) * current_price
            initial_balance = trader_obj.account.initial_balance
            is_liquidated = trader_obj.is_liquidated

            noise_trader_data.append(
                NoiseTraderInfo(
                    name=name,
                    equity=equity,
                    position_qty=position_qty,
                    position_value=position_value,
                    initial_balance=initial_balance,
                    is_liquidated=is_liquidated,
                )
            )

            # 记录净值历史
            if i < len(self.noise_trader_equity_history):
                self.noise_trader_equity_history[i].append(equity)

        return noise_trader_data

    def reset(self) -> None:
        """重置历史数据（新episode开始时调用）"""
        self.price_history.clear()
        for history in self.equity_history.values():
            history.clear()
        for history in self.alive_equity_history.values():
            history.clear()
        # 重置噪声交易者净值历史
        for history in self.noise_trader_equity_history:
            history.clear()
