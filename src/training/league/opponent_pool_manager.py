"""多类型对手池管理器"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.config import AgentType
from src.training.league.config import LeagueTrainingConfig
from src.training.league.opponent_entry import OpponentEntry, OpponentMetadata
from src.training.league.opponent_pool import OpponentPool

if TYPE_CHECKING:
    from src.training.league.arena_allocator import HybridSamplingResult


class OpponentPoolManager:
    """多类型对手池管理器

    管理四种 Agent 类型的独立对手池。
    """

    def __init__(self, config: LeagueTrainingConfig) -> None:
        """初始化

        Args:
            config: 联盟训练配置
        """
        self.config = config
        self.pool_dir = Path(config.pool_dir)

        # 四种类型的对手池
        self.pools: dict[AgentType, OpponentPool] = {
            agent_type: OpponentPool(agent_type, self.pool_dir, config)
            for agent_type in AgentType
        }

    def get_pool(self, agent_type: AgentType) -> OpponentPool:
        """获取指定类型的对手池

        Args:
            agent_type: Agent 类型

        Returns:
            对手池实例
        """
        return self.pools[agent_type]

    def add_snapshot(
        self,
        generation: int,
        genome_data_map: dict[AgentType, dict[int, tuple]],
        network_data_map: dict[AgentType, dict[int, tuple]] | None,
        fitness_map: dict[AgentType, float],
        source: str,
        add_reason: str,
        sub_population_counts: dict[AgentType, int] | None = None,
        agent_counts: dict[AgentType, int] | None = None,
        pre_evolution_fitness_map: dict[AgentType, dict[int, np.ndarray]] | None = None,
    ) -> dict[AgentType, str]:
        """保存当前所有类型的快照到各自的对手池

        每个 AgentType 的 pool 写入独立目录，使用多线程并行写入。

        Args:
            generation: 当前代数
            genome_data_map: {AgentType: {sub_pop_id: genome_tuple}}
            network_data_map: {AgentType: {sub_pop_id: network_tuple}} or None
            fitness_map: {AgentType: avg_fitness}
            source: 来源 ('main_agents', 'league_exploiter', 'main_exploiter')
            add_reason: 添加原因
            sub_population_counts: {AgentType: count}
            agent_counts: {AgentType: count}
            pre_evolution_fitness_map: {AgentType: {sub_pop_id: fitness_array}} or None

        Returns:
            {AgentType: entry_id} 映射
        """
        # 需要处理的 AgentType 列表
        types_to_process: list[AgentType] = [
            agent_type for agent_type in AgentType
            if agent_type in genome_data_map
        ]

        if not types_to_process:
            return {}

        def _process_single_type(agent_type: AgentType) -> tuple[AgentType, str]:
            """处理单个 AgentType 的快照保存

            Args:
                agent_type: 要处理的 Agent 类型

            Returns:
                (agent_type, entry_id) 元组
            """
            entry_id = f"gen_{generation:05d}"
            if source != "main_agents":
                entry_id = f"gen_{generation:05d}_{source}"

            # 创建元数据
            metadata = OpponentMetadata(
                entry_id=entry_id,
                agent_type=agent_type,
                source=source,
                source_generation=generation,
                add_reason=add_reason,
                avg_fitness=fitness_map.get(agent_type, 0.0),
                agent_count=agent_counts.get(agent_type, 0) if agent_counts else 0,
                sub_population_count=sub_population_counts.get(agent_type, 1) if sub_population_counts else 1,
            )

            # 创建条目
            entry = OpponentEntry(
                metadata=metadata,
                genome_data=genome_data_map[agent_type],
                network_data=network_data_map.get(agent_type) if network_data_map else None,
                pre_evolution_fitness=pre_evolution_fitness_map.get(agent_type) if pre_evolution_fitness_map else None,
            )

            # 添加到对手池（各类型写入独立目录，线程安全）
            pool = self.pools[agent_type]
            saved_entry_id: str = pool.add_entry(entry)
            return agent_type, saved_entry_id

        # 多线程并行写入各 AgentType 的对手池
        result: dict[AgentType, str] = {}
        with ThreadPoolExecutor(max_workers=len(types_to_process)) as executor:
            futures: dict[AgentType, Any] = {
                agent_type: executor.submit(_process_single_type, agent_type)
                for agent_type in types_to_process
            }
            for agent_type in types_to_process:
                future = futures[agent_type]
                completed_type, entry_id = future.result()
                result[completed_type] = entry_id

        return result

    def sample_opponents_for_arena(
        self,
        target_type: AgentType,
        strategy: str | None = None,
        current_generation: int = 0,
    ) -> dict[AgentType, str | None]:
        """为某个类型的训练采样其他三种类型的对手

        Args:
            target_type: 目标类型（当前要训练的类型）
            strategy: 采样策略，None 使用配置默认值
            current_generation: 当前代数

        Returns:
            {AgentType: entry_id}，包含所有四种类型
            target_type 对应的值为 None（使用当前代）
        """
        if strategy is None:
            strategy = self.config.sampling_strategy

        result: dict[AgentType, str | None] = {}

        for agent_type in AgentType:
            if agent_type == target_type:
                # 目标类型使用当前代 Main
                result[agent_type] = None
            else:
                # 其他类型从对手池采样
                pool = self.pools[agent_type]
                sampled = pool.sample_opponents(1, strategy, target_type, current_generation)
                result[agent_type] = sampled[0] if sampled else None

        return result

    def sample_all_historical(
        self,
        strategy: str | None = None,
        target_type: AgentType | None = None,
        current_generation: int = 0,
    ) -> dict[AgentType, str | None]:
        """采样所有类型的历史对手

        Args:
            strategy: 采样策略
            target_type: 目标类型（用于 PFSP）
            current_generation: 当前代数

        Returns:
            {AgentType: entry_id}
        """
        if strategy is None:
            strategy = self.config.sampling_strategy

        result: dict[AgentType, str | None] = {}

        for agent_type in AgentType:
            pool = self.pools[agent_type]
            sampled = pool.sample_opponents(1, strategy, target_type, current_generation)
            result[agent_type] = sampled[0] if sampled else None

        return result

    def save_all(self) -> None:
        """保存所有对手池索引"""
        for pool in self.pools.values():
            pool.save_index()

    def load_all(self) -> None:
        """加载所有对手池索引"""
        for pool in self.pools.values():
            pool.load_index()

    def cleanup_all(self, current_generation: int) -> dict[AgentType, list[str]]:
        """清理所有对手池

        Args:
            current_generation: 当前代数

        Returns:
            {AgentType: [removed_entry_ids]}
        """
        result: dict[AgentType, list[str]] = {}
        for agent_type, pool in self.pools.items():
            removed = pool.cleanup(current_generation)
            if removed:
                result[agent_type] = removed
        return result

    def get_pool_sizes(self) -> dict[AgentType, int]:
        """获取所有对手池的大小

        Returns:
            {AgentType: size}
        """
        return {
            agent_type: pool.get_pool_size()
            for agent_type, pool in self.pools.items()
        }

    def has_historical_opponents(self, agent_type: AgentType) -> bool:
        """检查指定类型是否有历史对手

        Args:
            agent_type: Agent 类型

        Returns:
            是否有历史对手
        """
        return not self.pools[agent_type].is_empty()

    def has_any_historical_opponents(self) -> bool:
        """检查是否有任何历史对手"""
        return any(not pool.is_empty() for pool in self.pools.values())

    def get_entry(
        self,
        agent_type: AgentType,
        entry_id: str,
        load_networks: bool = False,
    ) -> OpponentEntry | None:
        """获取指定条目

        Args:
            agent_type: Agent 类型
            entry_id: 条目 ID
            load_networks: 是否加载网络参数

        Returns:
            条目对象
        """
        return self.pools[agent_type].get_entry(entry_id, load_networks)

    def sample_opponents_batch_for_type(
        self,
        target_type: AgentType,
        n_arenas: int,
        strategy: str | None = None,
        current_generation: int = 0,
    ) -> list[dict[AgentType, str | None]]:
        """为目标类型的多个泛化竞技场批量采样对手

        保证各竞技场间对手不重复（如果对手池足够大）。

        Args:
            target_type: 目标物种类型
            n_arenas: 竞技场数量
            strategy: 采样策略
            current_generation: 当前代数

        Returns:
            长度为 n_arenas 的列表，每个元素是 {AgentType: entry_id | None}
        """
        if strategy is None:
            strategy = self.config.sampling_strategy

        # 对每种非目标类型批量采样
        batch_by_type: dict[AgentType, list[str]] = {}
        for agent_type in AgentType:
            if agent_type == target_type:
                continue
            pool = self.pools[agent_type]
            sampled = pool.sample_opponents_batch(
                n_arenas, strategy, target_type, current_generation
            )
            batch_by_type[agent_type] = sampled

        # 组装结果
        results: list[dict[AgentType, str | None]] = []
        for i in range(n_arenas):
            arena_opponents: dict[AgentType, str | None] = {}
            for agent_type in AgentType:
                if agent_type == target_type:
                    arena_opponents[agent_type] = None
                else:
                    sampled_list = batch_by_type.get(agent_type, [])
                    if i < len(sampled_list):
                        arena_opponents[agent_type] = sampled_list[i]
                    else:
                        arena_opponents[agent_type] = None
            results.append(arena_opponents)

        return results
