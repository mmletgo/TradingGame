"""联盟适应度汇总器"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.config.config import AgentType
from src.training.league.config import LeagueTrainingConfig
from src.training.league.arena_allocator import ArenaAllocation


@dataclass
class GeneralizationAdvantageStats:
    """单代泛化优势比统计"""
    generation: int
    advantages: dict[AgentType, float]          # 泛化优势比
    baseline_avg: dict[AgentType, float]        # 基准平均适应度
    generalization_avg: dict[AgentType, float]  # 泛化平均适应度


class LeagueFitnessAggregator:
    """联盟适应度汇总器

    按类型和角色汇总来自不同竞技场的适应度数据。

    Main Agents 的适应度来自基准竞技场和泛化测试竞技场。
    """

    def __init__(self, config: LeagueTrainingConfig) -> None:
        """初始化

        Args:
            config: 联盟训练配置
        """
        self.config = config

        # 泛化优势比历史
        self._advantage_history: deque[GeneralizationAdvantageStats] = deque(
            maxlen=config.generalization_advantage_window
        )

    def aggregate_main_fitness(
        self,
        agent_type: AgentType,
        baseline_fitnesses: list[np.ndarray],
        generalization_fitnesses: list[np.ndarray],
    ) -> np.ndarray:
        """计算 Main Agents 的最终适应度

        使用加权平均汇总基准竞技场和泛化测试竞技场的适应度。

        Args:
            agent_type: Agent 类型
            baseline_fitnesses: 基准竞技场中该类型的适应度数组列表
            generalization_fitnesses: 泛化测试竞技场中该类型的适应度数组列表

        Returns:
            该类型所有个体的最终适应度数组
        """
        if not baseline_fitnesses and not generalization_fitnesses:
            return np.array([])

        # 确定数组大小
        sample_arr = baseline_fitnesses[0] if baseline_fitnesses else generalization_fitnesses[0]
        n_agents = len(sample_arr)

        strategy = self.config.fitness_strategy

        if strategy == 'simple':
            # 简单平均
            all_fitnesses = baseline_fitnesses + generalization_fitnesses
            return np.mean(all_fitnesses, axis=0)

        elif strategy == 'weighted_average':
            # 加权平均
            baseline_weight = self.config.baseline_weight
            generalization_weight = self.config.generalization_weight

            total_fitness = np.zeros(n_agents, dtype=np.float64)
            total_weight = 0.0

            for fitness in baseline_fitnesses:
                total_fitness += fitness * baseline_weight
                total_weight += baseline_weight

            for fitness in generalization_fitnesses:
                total_fitness += fitness * generalization_weight
                total_weight += generalization_weight

            if total_weight > 0:
                return total_fitness / total_weight
            else:
                return np.zeros(n_agents, dtype=np.float64)

        elif strategy == 'min':
            # 取最小值（保守策略）
            all_fitnesses = baseline_fitnesses + generalization_fitnesses
            return np.min(all_fitnesses, axis=0)

        else:
            # 默认使用加权平均
            return self.aggregate_main_fitness(
                agent_type,
                baseline_fitnesses,
                generalization_fitnesses,
            )

    def collect_fitness_by_role(
        self,
        allocation: ArenaAllocation,
        arena_fitnesses: dict[int, dict[AgentType, np.ndarray]],
    ) -> dict[str, dict[AgentType, list[np.ndarray]]]:
        """按角色收集适应度

        根据竞技场分配方案，将各竞技场的适应度分配到对应的角色。

        Args:
            allocation: 竞技场分配方案
            arena_fitnesses: {arena_id: {AgentType: fitness_array}}

        Returns:
            {
                'main_baseline': {AgentType: [fitness_arrays]},
                'main_generalization': {AgentType: [fitness_arrays]},
            }
        """
        result: dict[str, dict[AgentType, list[np.ndarray]]] = {
            'main_baseline': {t: [] for t in AgentType},
            'main_generalization': {t: [] for t in AgentType},
        }

        # 基准竞技场：所有类型都参与 Main 适应度
        for arena_id in allocation.baseline_arena_ids:
            if arena_id not in arena_fitnesses:
                continue
            for agent_type in AgentType:
                if agent_type in arena_fitnesses[arena_id]:
                    result['main_baseline'][agent_type].append(
                        arena_fitnesses[arena_id][agent_type]
                    )

        # 泛化测试竞技场：只有目标类型参与 Main 适应度
        for agent_type, arena_ids in allocation.generalization_arena_ids.items():
            for arena_id in arena_ids:
                if arena_id not in arena_fitnesses:
                    continue
                if agent_type in arena_fitnesses[arena_id]:
                    result['main_generalization'][agent_type].append(
                        arena_fitnesses[arena_id][agent_type]
                    )

        return result

    def compute_all_fitness(
        self,
        allocation: ArenaAllocation,
        arena_fitnesses: dict[int, dict[AgentType, np.ndarray]],
    ) -> dict[str, dict[AgentType, np.ndarray]]:
        """计算所有角色的最终适应度

        Args:
            allocation: 竞技场分配方案
            arena_fitnesses: {arena_id: {AgentType: fitness_array}}

        Returns:
            {
                'main': {AgentType: final_fitness_array},
            }
        """
        # 收集适应度
        collected = self.collect_fitness_by_role(allocation, arena_fitnesses)

        result: dict[str, dict[AgentType, np.ndarray]] = {
            'main': {},
        }

        # 计算 Main Agents 适应度
        for agent_type in AgentType:
            baseline = collected['main_baseline'][agent_type]
            generalization = collected['main_generalization'][agent_type]

            if baseline or generalization:
                result['main'][agent_type] = self.aggregate_main_fitness(
                    agent_type, baseline, generalization
                )

        return result

    def determine_winner(
        self,
        fitness_array: np.ndarray,
        threshold: float = 0.0,
    ) -> bool:
        """判断是否获胜

        基于平均适应度判断是否获胜。

        Args:
            fitness_array: 适应度数组
            threshold: 胜利阈值

        Returns:
            是否获胜
        """
        if len(fitness_array) == 0:
            return False
        return float(np.mean(fitness_array)) > threshold

    def compute_generalization_advantage(
        self,
        generation: int,
        allocation: ArenaAllocation,
        arena_fitnesses: dict[int, dict[AgentType, np.ndarray]],
    ) -> GeneralizationAdvantageStats | None:
        """计算泛化优势比

        Args:
            generation: 当前代数
            allocation: 竞技场分配方案
            arena_fitnesses: {arena_id: {AgentType: fitness_array}}

        Returns:
            泛化优势比统计，无泛化竞技场时返回 None
        """
        # 检查是否有泛化测试竞技场
        has_generalization = any(
            len(ids) > 0 for ids in allocation.generalization_arena_ids.values()
        )
        if not has_generalization:
            return None

        # 使用已有方法分离基准/泛化适应度
        collected = self.collect_fitness_by_role(allocation, arena_fitnesses)

        advantages: dict[AgentType, float] = {}
        baseline_avg: dict[AgentType, float] = {}
        generalization_avg: dict[AgentType, float] = {}

        for agent_type in AgentType:
            # 基准平均
            baseline_arrays = collected['main_baseline'][agent_type]
            baseline_avg[agent_type] = (
                float(np.mean([np.mean(arr) for arr in baseline_arrays]))
                if baseline_arrays else 0.0
            )

            # 泛化平均
            gen_arrays = collected['main_generalization'][agent_type]
            generalization_avg[agent_type] = (
                float(np.mean([np.mean(arr) for arr in gen_arrays]))
                if gen_arrays else 0.0
            )

            # 泛化优势比
            advantages[agent_type] = generalization_avg[agent_type] - baseline_avg[agent_type]

        stats = GeneralizationAdvantageStats(
            generation=generation,
            advantages=advantages,
            baseline_avg=baseline_avg,
            generalization_avg=generalization_avg,
        )

        # 记录到历史
        self._advantage_history.append(stats)

        return stats

    def check_convergence(self) -> tuple[bool, dict[AgentType, bool]]:
        """检查是否收敛

        收敛条件：最近 N 代的泛化优势比绝对值都 <= 阈值

        Returns:
            (全部收敛, {AgentType: 是否收敛})
        """
        threshold = self.config.convergence_threshold
        required_gens = self.config.convergence_generations

        if len(self._advantage_history) < required_gens:
            return False, {t: False for t in AgentType}

        recent = list(self._advantage_history)[-required_gens:]

        converged_by_type: dict[AgentType, bool] = {}
        for agent_type in AgentType:
            converged_by_type[agent_type] = all(
                abs(stats.advantages[agent_type]) <= threshold
                for stats in recent
            )

        return all(converged_by_type.values()), converged_by_type

    def get_advantage_history(self) -> list[GeneralizationAdvantageStats]:
        """获取泛化优势比历史"""
        return list(self._advantage_history)

    def clear_history(self) -> None:
        """清空历史记录"""
        self._advantage_history.clear()
