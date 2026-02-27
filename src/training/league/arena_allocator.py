"""混合竞技场历史对手采样器"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config.config import AgentType
from src.training.league.config import LeagueTrainingConfig
from src.training.league.opponent_pool import OpponentPool
from src.training.league.opponent_pool_manager import OpponentPoolManager, LEAGUE_AGENT_TYPES


@dataclass
class HybridSamplingResult:
    """一轮训练的历史对手采样结果

    每种 AgentType 独立采样多代历史 entries，提取精英网络。

    注意：elite_networks 和 total_elite_counts 由 LeagueTrainer（调用方）
    在 _prepare_historical_agents() 中填充，allocator 仅填充 sampled_entries。
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

        for agent_type in LEAGUE_AGENT_TYPES:
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
        """带新鲜度约束的 PFSP 采样

        将对手池按代数排序，分为"最近 1/3"和"全池"两部分。
        先从最近 1/3 中按 PFSP 权重采样 n_recent 个，
        再从全池中按 PFSP 权重采样 n_rest 个（排除已选的）。

        Args:
            pool: 对手池
            n_total: 总采样数
            freshness_ratio: 最近历史的最低占比
            current_generation: 当前代数

        Returns:
            采样的 entry_id 列表
        """
        # 统一从索引获取所有 entry 及其代数（避免双数据源不一致）
        entries_with_gen: list[tuple[str, int]] = []
        index_entries: list[dict[str, Any]] = pool.list_entries()
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
        if freshness_ratio <= 0:
            n_recent = 0
        else:
            n_recent: int = max(1, int(n_total * freshness_ratio))
        n_rest: int = n_total - n_recent

        # 划分"最近 1/3"池（向上取整）
        recent_cutoff: int = max(1, (len(entries_with_gen) + 2) // 3)
        recent_ids: list[str] = [eid for eid, _ in entries_with_gen[:recent_cutoff]]
        all_ids: list[str] = [eid for eid, _ in entries_with_gen]

        # 获取全池 PFSP 权重（L8: 复用 index_entries，避免重复调用 list_entries()）
        pool_entries: list[dict[str, Any]] = index_entries
        strategy: str = self.config.sampling_strategy
        weights: np.ndarray = pool.compute_weights(
            pool_entries, strategy, pool.agent_type, current_generation
        )

        # 建立 entry_id -> 权重索引映射
        entry_id_to_weight: dict[str, float] = {}
        for i, entry_meta in enumerate(pool_entries):
            assert i < len(weights), f"权重数组长度({len(weights)})与条目数({len(pool_entries)})不一致"
            entry_id_to_weight[entry_meta["entry_id"]] = float(weights[i])

        # 从最近池按 PFSP 权重采样 n_recent 个
        actual_n_recent: int = min(n_recent, len(recent_ids))
        if actual_n_recent >= len(recent_ids):
            recent_sampled: list[str] = recent_ids[:]
        else:
            recent_weights: np.ndarray = np.array(
                [entry_id_to_weight.get(eid, 1.0) for eid in recent_ids],
                dtype=np.float64,
            )
            total_w: float = float(recent_weights.sum())
            if total_w > 0:
                recent_probs: np.ndarray = recent_weights / total_w
                nonzero_count: int = int(np.count_nonzero(recent_probs))
                if nonzero_count >= actual_n_recent:
                    sampled_indices: np.ndarray = np.random.choice(
                        len(recent_ids),
                        size=actual_n_recent,
                        replace=False,
                        p=recent_probs,
                    )
                    recent_sampled = [recent_ids[i] for i in sampled_indices]
                else:
                    # 非零项不足：先取所有非零项，再从零权重项中随机补齐
                    nonzero_indices: np.ndarray = np.nonzero(recent_probs)[0]
                    recent_sampled = [recent_ids[i] for i in nonzero_indices]
                    zero_indices: np.ndarray = np.where(recent_probs == 0)[0]
                    n_fill: int = actual_n_recent - nonzero_count
                    if n_fill > 0 and len(zero_indices) > 0:
                        fill_indices: np.ndarray = np.random.choice(
                            zero_indices, size=min(n_fill, len(zero_indices)), replace=False
                        )
                        recent_sampled.extend(recent_ids[i] for i in fill_indices)
            else:
                recent_sampled = recent_ids[:actual_n_recent]

        # 从全池采样 n_rest 个（排除已选）
        selected_set: set[str] = set(recent_sampled)
        # S2: 采样不足补偿 — 最近池不足时将余量补到全池采样
        n_rest = n_total - len(recent_sampled)
        remaining_ids: list[str] = [eid for eid in all_ids if eid not in selected_set]

        if n_rest <= 0 or not remaining_ids:
            return recent_sampled

        actual_n_rest: int = min(n_rest, len(remaining_ids))
        if actual_n_rest >= len(remaining_ids):
            rest_sampled: list[str] = remaining_ids[:]
        else:
            rest_weights: np.ndarray = np.array(
                [entry_id_to_weight.get(eid, 1.0) for eid in remaining_ids],
                dtype=np.float64,
            )
            total_w_rest: float = float(rest_weights.sum())
            if total_w_rest > 0:
                rest_probs: np.ndarray = rest_weights / total_w_rest
                nonzero_count_rest: int = int(np.count_nonzero(rest_probs))
                if nonzero_count_rest >= actual_n_rest:
                    sampled_rest_indices: np.ndarray = np.random.choice(
                        len(remaining_ids),
                        size=actual_n_rest,
                        replace=False,
                        p=rest_probs,
                    )
                    rest_sampled = [remaining_ids[i] for i in sampled_rest_indices]
                else:
                    # 非零项不足：先取所有非零项，再从零权重项中随机补齐
                    nonzero_rest_indices: np.ndarray = np.nonzero(rest_probs)[0]
                    rest_sampled = [remaining_ids[i] for i in nonzero_rest_indices]
                    zero_rest_indices: np.ndarray = np.where(rest_probs == 0)[0]
                    n_fill_rest: int = actual_n_rest - nonzero_count_rest
                    if n_fill_rest > 0 and len(zero_rest_indices) > 0:
                        fill_rest_indices: np.ndarray = np.random.choice(
                            zero_rest_indices, size=min(n_fill_rest, len(zero_rest_indices)), replace=False
                        )
                        rest_sampled.extend(remaining_ids[i] for i in fill_rest_indices)
            else:
                rest_sampled = remaining_ids[:actual_n_rest]

        return recent_sampled + rest_sampled
