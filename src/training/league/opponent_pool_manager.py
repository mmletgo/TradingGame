"""多类型对手池管理器"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config.config import AgentType
from src.training.league.config import LeagueTrainingConfig
from src.training.league.opponent_entry import OpponentEntry, OpponentMetadata
from src.training.league.opponent_pool import OpponentPool


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
    ) -> dict[AgentType, str]:
        """保存当前所有类型的快照到各自的对手池

        Args:
            generation: 当前代数
            genome_data_map: {AgentType: {sub_pop_id: genome_tuple}}
            network_data_map: {AgentType: {sub_pop_id: network_tuple}} or None
            fitness_map: {AgentType: avg_fitness}
            source: 来源 ('main_agents', 'league_exploiter', 'main_exploiter')
            add_reason: 添加原因
            sub_population_counts: {AgentType: count}
            agent_counts: {AgentType: count}

        Returns:
            {AgentType: entry_id} 映射
        """
        result: dict[AgentType, str] = {}

        for agent_type in AgentType:
            if agent_type not in genome_data_map:
                continue

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
            )

            # 添加到对手池
            pool = self.pools[agent_type]
            entry_id = pool.add_entry(entry)
            result[agent_type] = entry_id

        return result

    def sample_opponents_for_arena(
        self,
        target_type: AgentType,
        strategy: str | None = None,
    ) -> dict[AgentType, str | None]:
        """为某个类型的训练采样其他三种类型的对手

        Args:
            target_type: 目标类型（当前要训练的类型）
            strategy: 采样策略，None 使用配置默认值

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
                sampled = pool.sample_opponents(1, strategy, target_type)
                result[agent_type] = sampled[0] if sampled else None

        return result

    def sample_all_historical(
        self,
        strategy: str | None = None,
        target_type: AgentType | None = None,
    ) -> dict[AgentType, str | None]:
        """采样所有类型的历史对手

        Args:
            strategy: 采样策略
            target_type: 目标类型（用于 PFSP）

        Returns:
            {AgentType: entry_id}
        """
        if strategy is None:
            strategy = self.config.sampling_strategy

        result: dict[AgentType, str | None] = {}

        for agent_type in AgentType:
            pool = self.pools[agent_type]
            sampled = pool.sample_opponents(1, strategy, target_type)
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
