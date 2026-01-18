"""联盟适应度汇总器"""
from __future__ import annotations

from typing import Literal

import numpy as np

from src.config.config import AgentType
from src.training.league.config import LeagueTrainingConfig
from src.training.league.arena_allocator import ArenaAllocation


class LeagueFitnessAggregator:
    """联盟适应度汇总器

    按类型和角色汇总来自不同竞技场的适应度数据。

    Main Agents 的适应度来自基准竞技场和泛化测试竞技场。
    Exploiter 的适应度仅来自其专用训练竞技场。
    """

    def __init__(self, config: LeagueTrainingConfig) -> None:
        """初始化

        Args:
            config: 联盟训练配置
        """
        self.config = config

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

    def aggregate_exploiter_fitness(
        self,
        agent_type: AgentType,
        exploiter_type: Literal['league_exploiter', 'main_exploiter'],
        exploiter_fitnesses: list[np.ndarray],
    ) -> np.ndarray:
        """计算 Exploiter 的适应度

        Exploiter 的适应度仅来自其专用训练竞技场，使用简单平均。

        Args:
            agent_type: Agent 类型
            exploiter_type: Exploiter 类型
            exploiter_fitnesses: Exploiter 训练竞技场中的适应度数组列表

        Returns:
            Exploiter 的最终适应度数组
        """
        if not exploiter_fitnesses:
            return np.array([])

        # 简单平均
        return np.mean(exploiter_fitnesses, axis=0)

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
                'league_exploiter': {AgentType: [fitness_arrays]},
                'main_exploiter': {AgentType: [fitness_arrays]},
            }
        """
        result: dict[str, dict[AgentType, list[np.ndarray]]] = {
            'main_baseline': {t: [] for t in AgentType},
            'main_generalization': {t: [] for t in AgentType},
            'league_exploiter': {t: [] for t in AgentType},
            'main_exploiter': {t: [] for t in AgentType},
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

        # League Exploiter 训练竞技场
        for agent_type, arena_ids in allocation.league_exploiter_arena_ids.items():
            for arena_id in arena_ids:
                if arena_id not in arena_fitnesses:
                    continue
                # Exploiter 的适应度
                if agent_type in arena_fitnesses[arena_id]:
                    result['league_exploiter'][agent_type].append(
                        arena_fitnesses[arena_id][agent_type]
                    )

        # Main Exploiter 训练竞技场
        for arena_id in allocation.main_exploiter_arena_ids:
            if arena_id not in arena_fitnesses:
                continue
            assignment = allocation.get_assignment(arena_id)
            if assignment is None:
                continue

            # 被攻击的类型使用 Main
            # 攻击者类型使用 Main Exploiter
            target_type = assignment.target_type
            for agent_type in AgentType:
                if agent_type != target_type:
                    # 这是攻击者（Main Exploiter）
                    if agent_type in arena_fitnesses[arena_id]:
                        result['main_exploiter'][agent_type].append(
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
                'league_exploiter': {AgentType: final_fitness_array},
                'main_exploiter': {AgentType: final_fitness_array},
            }
        """
        # 收集适应度
        collected = self.collect_fitness_by_role(allocation, arena_fitnesses)

        result: dict[str, dict[AgentType, np.ndarray]] = {
            'main': {},
            'league_exploiter': {},
            'main_exploiter': {},
        }

        # 计算 Main Agents 适应度
        for agent_type in AgentType:
            baseline = collected['main_baseline'][agent_type]
            generalization = collected['main_generalization'][agent_type]

            if baseline or generalization:
                result['main'][agent_type] = self.aggregate_main_fitness(
                    agent_type, baseline, generalization
                )

        # 计算 League Exploiter 适应度
        for agent_type in AgentType:
            exploiter_fitness = collected['league_exploiter'][agent_type]
            if exploiter_fitness:
                result['league_exploiter'][agent_type] = self.aggregate_exploiter_fitness(
                    agent_type, 'league_exploiter', exploiter_fitness
                )

        # 计算 Main Exploiter 适应度
        for agent_type in AgentType:
            exploiter_fitness = collected['main_exploiter'][agent_type]
            if exploiter_fitness:
                result['main_exploiter'][agent_type] = self.aggregate_exploiter_fitness(
                    agent_type, 'main_exploiter', exploiter_fitness
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
