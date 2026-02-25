"""混合竞技场历史对手采样器"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from src.config.config import AgentType
from src.training.league.config import LeagueTrainingConfig
from src.training.league.opponent_pool import OpponentPool
from src.training.league.opponent_pool_manager import OpponentPoolManager


@dataclass
class HybridSamplingResult:
    """一轮训练的历史对手采样结果

    每种 AgentType 独立采样多代历史 entries，提取精英网络。
    """

    # 采样到的 entry IDs
    sampled_entries: dict[AgentType, list[str]]  # {AgentType: [entry_id, ...]}

    # 每个 entry 的精英网络参数: {AgentType: {entry_id: (elite_count, packed_params)}}
    elite_networks: dict[
        AgentType, dict[str, tuple[int, tuple[np.ndarray, ...]]]
    ] = field(default_factory=dict)

    # 每种类型的历史精英总数
    total_elite_counts: dict[AgentType, int] = field(default_factory=dict)

    def get_total_historical_count(self, agent_type: AgentType) -> int:
        """获取指定类型的历史精英总数"""
        return self.total_elite_counts.get(agent_type, 0)


class HybridArenaAllocator:
    """混合竞技场分配器

    负责从对手池中采样历史对手 entries（带新鲜度约束的 PFSP 采样）。
    精英网络提取由 LeagueTrainer 负责（因为需要加载 entry 数据）。
    """

    def __init__(self, config: LeagueTrainingConfig) -> None:
        """初始化

        Args:
            config: 联盟训练配置
        """
        self.config = config

    def sample_historical(
        self,
        pool_manager: OpponentPoolManager,
        current_generation: int,
    ) -> HybridSamplingResult | None:
        """带新鲜度约束的 PFSP 采样

        对每种 AgentType 独立采样 num_historical_generations 个历史 entries。

        采样策略（以 num_historical_generations=6, freshness_ratio=0.5 为例）：
        1. 将对手池按代数排序
        2. 分为"最近 1/3"和"全池"
        3. 前 3 个 entry: 从最近 1/3 中按策略采样
        4. 后 3 个 entry: 从全池中按策略采样（排除已选的）

        Args:
            pool_manager: 对手池管理器
            current_generation: 当前代数

        Returns:
            采样结果，所有类型的对手池都为空时返回 None
        """
        sampled_entries: dict[AgentType, list[str]] = {}
        any_sampled: bool = False

        for agent_type in AgentType:
            pool: OpponentPool = pool_manager.get_pool(agent_type)

            if pool.is_empty():
                sampled_entries[agent_type] = []
                continue

            entry_ids: list[str] = self._sample_with_freshness(
                pool=pool,
                n_total=self.config.num_historical_generations,
                freshness_ratio=self.config.historical_freshness_ratio,
                current_generation=current_generation,
            )

            sampled_entries[agent_type] = entry_ids
            if entry_ids:
                any_sampled = True

        if not any_sampled:
            return None

        return HybridSamplingResult(
            sampled_entries=sampled_entries,
            elite_networks={},
            total_elite_counts={},
        )

    def _sample_with_freshness(
        self,
        pool: OpponentPool,
        n_total: int,
        freshness_ratio: float,
        current_generation: int,
    ) -> list[str]:
        """带新鲜度约束的采样

        将对手池按代数排序，分为"最近 1/3"和"全池"两部分。
        先从最近 1/3 中采样 n_recent 个，再从全池中采样 n_rest 个（排除已选的）。

        Args:
            pool: 对手池
            n_total: 总采样数
            freshness_ratio: 最近历史的最低占比
            current_generation: 当前代数

        Returns:
            采样的 entry_id 列表
        """
        # 获取所有 entry 及其代数
        entries_with_gen: list[tuple[str, int]] = []
        for entry_id, entry in pool.entries.items():
            gen: int = entry.metadata.source_generation
            entries_with_gen.append((entry_id, gen))

        # 如果 entries 字典为空，尝试从索引获取
        if not entries_with_gen:
            index_entries = pool.list_entries()
            for index_entry in index_entries:
                entry_id_str: str = index_entry["entry_id"]
                gen_val: int = index_entry.get("source_generation", 0)
                entries_with_gen.append((entry_id_str, gen_val))

        if not entries_with_gen:
            return []

        # 按代数降序排序（最新的在前）
        entries_with_gen.sort(key=lambda x: x[1], reverse=True)

        # 计算采样数量
        n_total = min(n_total, len(entries_with_gen))
        n_recent: int = max(1, int(n_total * freshness_ratio))
        n_rest: int = n_total - n_recent

        # 划分"最近 1/3"池
        recent_cutoff: int = max(1, len(entries_with_gen) // 3)
        recent_pool: list[str] = [
            eid for eid, _ in entries_with_gen[:recent_cutoff]
        ]

        # 从最近池采样 n_recent 个
        if len(recent_pool) <= n_recent:
            recent_sampled: list[str] = recent_pool[:]
        else:
            recent_sampled = random.sample(recent_pool, n_recent)

        # 从全池采样 n_rest 个（排除已选）
        selected_set: set[str] = set(recent_sampled)
        remaining_pool: list[str] = [
            eid for eid, _ in entries_with_gen if eid not in selected_set
        ]

        if len(remaining_pool) <= n_rest:
            rest_sampled: list[str] = remaining_pool[:]
        else:
            rest_sampled = random.sample(remaining_pool, n_rest)

        return recent_sampled + rest_sampled
