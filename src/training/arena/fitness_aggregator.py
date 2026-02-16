"""适应度汇总器模块

本模块提供 FitnessAggregator 类，用于汇总多个竞技场、多个 episode 的适应度数据。
"""

import numpy as np

from src.config.config import AgentType


class FitnessAggregator:
    """适应度汇总器

    汇总多个竞技场、多个 episode 的适应度数据。
    使用简单平均策略。
    """

    @staticmethod
    def aggregate_simple_average(
        arena_fitnesses: list[dict[tuple[AgentType, int], np.ndarray]],
        episode_counts: list[int],
    ) -> dict[tuple[AgentType, int], np.ndarray]:
        """简单加权平均

        公式：avg_fitness = sum(arena_fitness * episode_count) / total_episodes

        由于每个竞技场返回的 arena_fitness 已经是累积值（累加了多个 episode），
        实际上 arena_fitness 已经隐含了 episode_count 的权重，
        所以计算公式为：avg_fitness = sum(arena_fitness) / total_episodes

        Args:
            arena_fitnesses: 每个竞技场返回的适应度累积值
                - key: (agent_type, sub_pop_id) 元组
                - value: 累积适应度数组，shape=(pop_size,)
            episode_counts: 每个竞技场运行的 episode 数量

        Returns:
            平均适应度字典
                - key: (agent_type, sub_pop_id)
                - value: 平均适应度数组

        Raises:
            ValueError: 当 arena_fitnesses 和 episode_counts 长度不一致时
            ValueError: 当 arena_fitnesses 为空时
            ValueError: 当总 episode 数为 0 时

        Example:
            >>> arena_fitnesses = [
            ...     {(AgentType.RETAIL_PRO, 0): np.array([10.0, 20.0, 30.0])},  # 累积值
            ...     {(AgentType.RETAIL_PRO, 0): np.array([20.0, 30.0, 40.0])},  # 累积值
            ... ]
            >>> episode_counts = [10, 10]
            >>> # 结果：(10.0 + 20.0) / 20 = 1.5, (20.0 + 30.0) / 20 = 2.5, ...
            >>> result = FitnessAggregator.aggregate_simple_average(
            ...     arena_fitnesses, episode_counts
            ... )
        """
        # 参数校验
        if len(arena_fitnesses) != len(episode_counts):
            raise ValueError(
                f"arena_fitnesses 和 episode_counts 长度不一致: "
                f"{len(arena_fitnesses)} vs {len(episode_counts)}"
            )

        if not arena_fitnesses:
            raise ValueError("arena_fitnesses 不能为空")

        total_episodes = sum(episode_counts)
        if total_episodes == 0:
            raise ValueError("总 episode 数不能为 0")

        # 收集所有出现的 key
        all_keys: set[tuple[AgentType, int]] = set()
        for arena_fitness in arena_fitnesses:
            all_keys.update(arena_fitness.keys())

        # 汇总各个 key 的适应度
        result: dict[tuple[AgentType, int], np.ndarray] = {}

        for key in all_keys:
            # 收集所有竞技场中该 key 的累积适应度
            accumulated_fitness: np.ndarray | None = None

            for arena_fitness in arena_fitnesses:
                if key in arena_fitness:
                    fitness_array = arena_fitness[key]
                    if accumulated_fitness is None:
                        # 第一次遇到，复制一份（避免修改原数据）
                        accumulated_fitness = fitness_array.copy()
                    else:
                        # 累加
                        accumulated_fitness += fitness_array

            # 计算平均值
            if accumulated_fitness is not None:
                result[key] = accumulated_fitness / total_episodes

        return result
