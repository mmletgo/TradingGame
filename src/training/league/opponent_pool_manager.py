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

    管理两种 Agent 类型的独立对手池。
    """

    def __init__(self, config: LeagueTrainingConfig) -> None:
        """初始化

        Args:
            config: 联盟训练配置
        """
        self.config = config
        self.pool_dir = Path(config.pool_dir)

        # 两种类型的对手池
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

