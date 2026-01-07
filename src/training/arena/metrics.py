"""竞技场指标收集模块"""

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.bio.agents.base import AgentType


@dataclass
class EpisodeMetrics:
    """单个 episode 的指标

    Attributes:
        arena_id: 竞技场 ID
        episode: Episode 编号
        tick_count: 实际运行的 tick 数
        high_price: 最高价
        low_price: 最低价
        final_price: 收盘价
        volatility: 波动率
        total_volume: 总成交量
        liquidation_count: 各类型强平数量
        avg_fitness: 各类型平均适应度
        elite_species_fitness: 各类型最精英 species 的平均适应度
    """
    arena_id: int
    episode: int
    tick_count: int = 0
    high_price: float = 0.0
    low_price: float = 0.0
    final_price: float = 0.0
    volatility: float = 0.0
    total_volume: float = 0.0
    liquidation_count: dict["AgentType", int] = field(default_factory=dict)
    avg_fitness: dict["AgentType", float] = field(default_factory=dict)
    elite_species_fitness: dict["AgentType", float] = field(default_factory=dict)


class ArenaMetrics:
    """竞技场指标收集器

    收集单个竞技场的训练指标。

    Attributes:
        arena_id: 竞技场 ID
        episode_history: Episode 指标历史（最多保留 1000 条）
    """

    arena_id: int
    episode_history: deque[EpisodeMetrics]

    def __init__(self, arena_id: int) -> None:
        """初始化指标收集器

        Args:
            arena_id: 竞技场 ID
        """
        self.arena_id = arena_id
        self.episode_history = deque(maxlen=1000)

    def record_episode(
        self,
        episode: int,
        price_stats: dict,
        population_stats: dict,
    ) -> EpisodeMetrics:
        """记录 episode 指标

        Args:
            episode: Episode 编号
            price_stats: 价格统计
            population_stats: 种群统计

        Returns:
            记录的指标
        """
        high_price = price_stats.get("high_price", 0.0)
        low_price = price_stats.get("low_price", 0.0)
        mid = (high_price + low_price) / 2 if high_price + low_price > 0 else 1.0
        volatility = (high_price - low_price) / mid if mid > 0 else 0.0

        metrics = EpisodeMetrics(
            arena_id=self.arena_id,
            episode=episode,
            tick_count=price_stats.get("tick_count", 0),
            high_price=high_price,
            low_price=low_price,
            final_price=price_stats.get("final_price", 0.0),
            volatility=volatility,
            total_volume=price_stats.get("total_volume", 0.0),
            liquidation_count=population_stats.get("liquidations", {}),
            avg_fitness=population_stats.get("avg_fitness", {}),
            elite_species_fitness=population_stats.get("elite_species_fitness", {}),
        )
        self.episode_history.append(metrics)
        return metrics

    def get_latest(self) -> EpisodeMetrics | None:
        """获取最新指标

        Returns:
            最新的 episode 指标，如果没有则返回 None
        """
        return self.episode_history[-1] if self.episode_history else None

    def get_summary(self, window: int = 100) -> dict:
        """获取最近 N 个 episode 的汇总指标

        Args:
            window: 统计窗口大小

        Returns:
            汇总指标字典
        """
        recent = list(self.episode_history)[-window:]
        if not recent:
            return {}

        return {
            "arena_id": self.arena_id,
            "episode_count": len(recent),
            "avg_volatility": float(np.mean([m.volatility for m in recent])),
            "price_range": (
                min(m.low_price for m in recent),
                max(m.high_price for m in recent),
            ),
        }


class MetricsAggregator:
    """多竞技场指标聚合器

    聚合所有竞技场的训练指标。

    Attributes:
        arena_metrics: 各竞技场的指标历史（每个竞技场最多保留 1000 条）
        _max_history: 每个竞技场最大历史记录数
    """

    arena_metrics: dict[int, deque[EpisodeMetrics]]
    _max_history: int

    def __init__(self, max_history: int = 1000) -> None:
        """初始化聚合器

        Args:
            max_history: 每个竞技场最大历史记录数，默认 1000
        """
        self.arena_metrics = {}
        self._max_history = max_history

    def update(self, metrics: EpisodeMetrics) -> None:
        """更新指标

        Args:
            metrics: Episode 指标
        """
        if metrics.arena_id not in self.arena_metrics:
            self.arena_metrics[metrics.arena_id] = deque(maxlen=self._max_history)
        self.arena_metrics[metrics.arena_id].append(metrics)

    def update_batch(self, metrics_list: list[EpisodeMetrics]) -> None:
        """批量更新指标

        Args:
            metrics_list: Episode 指标列表
        """
        for metrics in metrics_list:
            self.update(metrics)

    def get_summary(self, window: int = 100) -> dict:
        """获取所有竞技场的汇总

        Args:
            window: 统计窗口大小

        Returns:
            汇总字典
        """
        summaries = {}
        for arena_id, metrics_deque in self.arena_metrics.items():
            recent = list(metrics_deque)[-window:]
            if recent:
                summaries[arena_id] = {
                    "episodes": len(metrics_deque),
                    "avg_volatility": float(np.mean([m.volatility for m in recent])),
                }
        return summaries

    def get_global_summary(self, window: int = 100) -> dict:
        """获取全局汇总

        Args:
            window: 统计窗口大小

        Returns:
            全局汇总字典，包含 elite_species_fitness（各竞技场最新 episode 的平均值）
        """
        all_recent: list[EpisodeMetrics] = []
        for metrics_deque in self.arena_metrics.values():
            all_recent.extend(list(metrics_deque)[-window:])

        if not all_recent:
            return {}

        # 聚合各竞技场最新 episode 的 elite_species_fitness
        # 收集每个竞技场最新的 elite_species_fitness，然后对所有竞技场取平均
        elite_fitness_by_type: dict[str, list[float]] = {}
        for arena_id, metrics_deque in self.arena_metrics.items():
            if metrics_deque:
                latest = metrics_deque[-1]
                for agent_type, fitness in latest.elite_species_fitness.items():
                    type_key = agent_type.value if hasattr(agent_type, 'value') else str(agent_type)
                    if type_key not in elite_fitness_by_type:
                        elite_fitness_by_type[type_key] = []
                    elite_fitness_by_type[type_key].append(fitness)

        # 计算每个 AgentType 的平均 elite_species_fitness
        avg_elite_fitness: dict[str, float] = {}
        for type_key, fitness_list in elite_fitness_by_type.items():
            if fitness_list:
                avg_elite_fitness[type_key] = float(np.mean(fitness_list))

        return {
            "total_arenas": len(self.arena_metrics),
            "total_episodes": sum(len(m) for m in self.arena_metrics.values()),
            "avg_volatility": float(np.mean([m.volatility for m in all_recent])),
            "elite_species_fitness": avg_elite_fitness,
        }

    def get_history(self) -> dict[int, list[EpisodeMetrics]]:
        """获取完整历史（用于检查点）

        Returns:
            完整的指标历史（转换为 list 格式）
        """
        return {arena_id: list(metrics) for arena_id, metrics in self.arena_metrics.items()}
