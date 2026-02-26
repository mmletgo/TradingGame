"""混合竞技场适应度汇总器"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.config.config import AgentType
from src.training.league.config import LeagueTrainingConfig


@dataclass
class GenerationalComparisonStats:
    """代际适应度对比统计"""

    generation: int
    current_avg_fitness: dict[AgentType, float]  # 本代平均适应度
    previous_avg_fitness: dict[AgentType, float]  # 上一代平均适应度
    improvement: dict[AgentType, float]  # 提升量
    elite_current_avg: dict[AgentType, float]  # 本代精英平均
    elite_previous_avg: dict[AgentType, float]  # 上一代精英平均
    elite_improvement: dict[AgentType, float]  # 精英提升量


class HybridFitnessAggregator:
    """混合竞技场适应度汇总器

    所有竞技场使用相同的Agent集合（当前代+历史代精英），
    适应度使用简单平均汇总。代际对比作为监控指标。
    """

    def __init__(self, config: LeagueTrainingConfig) -> None:
        """初始化

        Args:
            config: 联盟训练配置
        """
        self.config = config

        # 代际对比历史
        self._comparison_history: deque[GenerationalComparisonStats] = deque(
            maxlen=config.generational_comparison_window
        )

        # 每代平均适应度历史（用于收敛判断）
        self._fitness_history: deque[dict[AgentType, float]] = deque(
            maxlen=config.generational_comparison_window
        )
        self._elite_fitness_history: deque[dict[AgentType, float]] = deque(
            maxlen=config.generational_comparison_window
        )

        # 首次收敛的代数
        self._first_convergence_generation: int | None = None

    def aggregate_fitness(
        self,
        arena_fitnesses: dict[int, dict[AgentType, np.ndarray]],
    ) -> dict[AgentType, np.ndarray]:
        """简单平均所有竞技场的适应度

        Args:
            arena_fitnesses: {arena_id: {AgentType: fitness_array}}

        Returns:
            {AgentType: averaged_fitness_array}
        """
        if not arena_fitnesses:
            return {}

        # 收集每种类型的所有适应度
        collected: dict[AgentType, list[np.ndarray]] = {t: [] for t in AgentType}
        for arena_id, fitness_by_type in arena_fitnesses.items():
            for agent_type, fitness_arr in fitness_by_type.items():
                collected[agent_type].append(fitness_arr)

        result: dict[AgentType, np.ndarray] = {}
        for agent_type, arrays in collected.items():
            if arrays:
                result[agent_type] = np.mean(arrays, axis=0).astype(np.float32)

        del collected
        return result

    def _compute_elite_avg(self, fitness_array: np.ndarray) -> float:
        """计算精英平均适应度

        Args:
            fitness_array: 适应度数组

        Returns:
            精英平均适应度
        """
        if len(fitness_array) == 0:
            return 0.0
        n_elite = max(1, int(len(fitness_array) * self.config.elite_ratio))
        sorted_fitness = np.sort(fitness_array)[::-1]  # 降序
        return float(np.mean(sorted_fitness[:n_elite]))

    def compute_generational_comparison(
        self,
        generation: int,
        avg_fitness: dict[AgentType, np.ndarray],
    ) -> GenerationalComparisonStats | None:
        """计算代际适应度对比

        Args:
            generation: 当前代数
            avg_fitness: {AgentType: fitness_array}（当前代平均适应度）

        Returns:
            代际对比统计，第一代时返回 None
        """
        # 计算当前代的种群和精英平均
        current_avg: dict[AgentType, float] = {}
        elite_current: dict[AgentType, float] = {}
        for agent_type in AgentType:
            arr = avg_fitness.get(agent_type)
            if arr is not None and len(arr) > 0:
                current_avg[agent_type] = float(np.mean(arr))
                elite_current[agent_type] = self._compute_elite_avg(arr)
            else:
                current_avg[agent_type] = 0.0
                elite_current[agent_type] = 0.0

        # 记录到历史
        self._fitness_history.append(current_avg.copy())
        self._elite_fitness_history.append(elite_current.copy())

        # 与上一代对比
        if len(self._fitness_history) < 2:
            return None

        previous_avg = self._fitness_history[-2]
        elite_previous = self._elite_fitness_history[-2]

        improvement: dict[AgentType, float] = {}
        elite_improvement: dict[AgentType, float] = {}
        for agent_type in AgentType:
            improvement[agent_type] = (
                current_avg[agent_type] - previous_avg.get(agent_type, 0.0)
            )
            elite_improvement[agent_type] = (
                elite_current[agent_type] - elite_previous.get(agent_type, 0.0)
            )

        stats = GenerationalComparisonStats(
            generation=generation,
            current_avg_fitness=current_avg,
            previous_avg_fitness=previous_avg,
            improvement=improvement,
            elite_current_avg=elite_current,
            elite_previous_avg=elite_previous,
            elite_improvement=elite_improvement,
        )

        self._comparison_history.append(stats)
        return stats

    def check_convergence(
        self,
    ) -> tuple[bool, dict[AgentType, bool]]:
        """检查收敛：最近N代适应度标准差 ≤ 阈值

        Returns:
            (全部收敛, {AgentType: 是否收敛})
        """
        threshold = self.config.convergence_fitness_std_threshold
        required_gens = self.config.convergence_generations

        if len(self._fitness_history) < required_gens:
            return False, {t: False for t in AgentType}

        recent_fitness = list(self._fitness_history)[-required_gens:]
        recent_elite = list(self._elite_fitness_history)[-required_gens:]

        converged_by_type: dict[AgentType, bool] = {}
        for agent_type in AgentType:
            # 种群适应度标准差
            pop_values = [f.get(agent_type, 0.0) for f in recent_fitness]
            pop_std = float(np.std(pop_values))

            # 精英适应度标准差
            elite_values = [f.get(agent_type, 0.0) for f in recent_elite]
            elite_std = float(np.std(elite_values))

            # 双重收敛：种群和精英的std都 ≤ 阈值
            converged_by_type[agent_type] = (
                pop_std <= threshold and elite_std <= threshold
            )

        is_all_converged = all(converged_by_type.values())

        # 记录首次收敛
        if is_all_converged and self._first_convergence_generation is None:
            if self._comparison_history:
                self._first_convergence_generation = (
                    self._comparison_history[-1].generation
                )
            elif self._fitness_history:
                # fallback: 从 fitness_history 长度推算当前代数
                self._first_convergence_generation = len(self._fitness_history) - 1

        return is_all_converged, converged_by_type

    def get_first_convergence_generation(self) -> int | None:
        """获取首次收敛的代数"""
        return self._first_convergence_generation

    def get_comparison_history(self) -> list[GenerationalComparisonStats]:
        """获取代际对比历史"""
        return list(self._comparison_history)

    def clear_history(self) -> None:
        """清空历史记录"""
        self._comparison_history.clear()
        self._fitness_history.clear()
        self._elite_fitness_history.clear()
        self._first_convergence_generation = None
