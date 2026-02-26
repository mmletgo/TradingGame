"""单类型对手池管理器"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config.config import AgentType
from src.training.league.config import LeagueTrainingConfig
from src.training.league.opponent_entry import OpponentEntry, OpponentMetadata


class OpponentPool:
    """单个 Agent 类型的对手池管理器

    管理某一 Agent 类型的历史版本，支持添加、删除、采样等操作。
    """

    def __init__(
        self,
        agent_type: AgentType,
        pool_dir: Path,
        config: LeagueTrainingConfig,
    ) -> None:
        """初始化对手池

        Args:
            agent_type: Agent 类型
            pool_dir: 对手池根目录
            config: 联盟训练配置
        """
        self.agent_type = agent_type
        self.pool_dir = pool_dir / agent_type.value
        self.config = config
        self.entries: dict[str, OpponentEntry] = {}
        self._index: dict[str, Any] = self._create_empty_index()
        # Recency 采样权重缓存
        self._prob_cache: np.ndarray | None = None
        self._entry_ids_cache: list[str] | None = None
        self._cache_version: int = 0
        self._last_cached_generation: int = -1

    def _invalidate_cache(self) -> None:
        """失效缓存"""
        self._prob_cache = None
        self._entry_ids_cache = None

    def _rebuild_cache_if_needed(self, current_generation: int) -> None:
        """按需重建 recency 权重缓存

        使用指数衰减公式：weight = exp(-lambda * (current_gen - source_gen) / milestone_interval)

        Args:
            current_generation: 当前代数，用于计算与 source_generation 的差值
        """
        entries = self._index.get("entries", [])
        current_version = len(entries)

        if (
            self._prob_cache is None
            or self._cache_version != current_version
            or self._last_cached_generation != current_generation
        ):
            self._entry_ids_cache = [e["entry_id"] for e in entries]
            decay_lambda: float = self.config.recency_decay_lambda
            milestone_interval: int = self.config.milestone_interval
            weights = np.array([
                math.exp(
                    -decay_lambda
                    * max(0, current_generation - e.get("source_generation", 0))
                    / milestone_interval
                )
                for e in entries
            ], dtype=np.float64)
            total = weights.sum()
            self._prob_cache = weights / total if total > 0 else None
            self._cache_version = current_version
            self._last_cached_generation = current_generation

    def _create_empty_index(self) -> dict[str, Any]:
        """创建空索引"""
        return {
            "version": 1,
            "agent_type": self.agent_type.value,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "entries": [],
            "config": {
                "max_pool_size": self.config.max_pool_size_per_type,
                "milestone_interval": self.config.milestone_interval,
            }
        }

    def save_index(self) -> None:
        """保存索引文件（原子写入）"""
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.pool_dir / "pool_index.json"
        tmp_path = self.pool_dir / "pool_index.json.tmp"
        self._index["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)
        os.rename(tmp_path, index_path)

    def load_index(self) -> None:
        """加载索引文件"""
        index_path = self.pool_dir / "pool_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self._index = json.loads(content)
                    else:
                        # 文件为空，创建新索引
                        self._index = self._create_empty_index()
            except json.JSONDecodeError:
                # 文件损坏，创建新索引
                self._index = self._create_empty_index()
        else:
            self._index = self._create_empty_index()

    def add_entry(self, entry: OpponentEntry) -> str:
        """添加条目到对手池

        Args:
            entry: 对手条目

        Returns:
            条目 ID
        """
        entry_id = entry.entry_id

        # 保存到磁盘
        entry_dir = self.pool_dir / entry_id
        entry.save(entry_dir)

        # 【内存泄漏修复】保存后清理大数据字段，只在内存中保留元数据
        # genome_data 和 network_data 已保存到磁盘，不需要在内存中保留
        entry.genome_data = None
        entry.network_data = None

        # 添加到内存（只保留元数据）
        self.entries[entry_id] = entry

        # 更新索引
        entry_meta = {
            "entry_id": entry_id,
            "source": entry.metadata.source,
            "source_generation": entry.metadata.source_generation,
            "add_reason": entry.metadata.add_reason,
            "avg_fitness": entry.metadata.avg_fitness,
            "win_rates": entry.metadata.win_rates,
            "match_counts": entry.metadata.match_counts,
            "created_at": entry.metadata.created_at,
        }
        self._index["entries"].append(entry_meta)
        self.save_index()
        self._invalidate_cache()

        return entry_id

    def remove_entry(self, entry_id: str, _skip_save_index: bool = False) -> None:
        """移除条目

        Args:
            entry_id: 条目 ID
            _skip_save_index: 内部参数，跳过保存索引和清除缓存（批量删除时使用）
        """
        # 从磁盘删除
        entry_dir = self.pool_dir / entry_id
        if entry_dir.exists():
            shutil.rmtree(entry_dir)

        # 从内存删除
        if entry_id in self.entries:
            # 【内存泄漏修复】显式清理大数据字段
            entry = self.entries[entry_id]
            entry.genome_data = None
            entry.network_data = None
            del self.entries[entry_id]

        # 更新索引
        self._index["entries"] = [
            e for e in self._index["entries"] if e["entry_id"] != entry_id
        ]
        if not _skip_save_index:
            self.save_index()
            self._invalidate_cache()

    def get_entry(self, entry_id: str, load_networks: bool = False) -> OpponentEntry | None:
        """获取条目

        Args:
            entry_id: 条目 ID
            load_networks: 是否加载网络参数

        Returns:
            条目对象，不存在返回 None
        """
        # 先检查内存缓存
        if entry_id in self.entries:
            entry = self.entries[entry_id]
            # 如果需要网络但缓存中没有，重新加载
            if load_networks and entry.network_data is None:
                entry_dir = self.pool_dir / entry_id
                loaded = OpponentEntry.load(entry_dir, load_networks=True)
                # 【内存泄漏修复】赋值字段而非替换对象，确保外部引用
                # （如 ensure_cached 中的 entry）指向同一对象
                entry.genome_data = loaded.genome_data
                entry.network_data = loaded.network_data
            return entry

        # 从磁盘加载
        entry_dir = self.pool_dir / entry_id
        if entry_dir.exists():
            entry = OpponentEntry.load(entry_dir, load_networks=load_networks)
            self.entries[entry_id] = entry
            return entry

        return None

    def list_entries(self) -> list[dict[str, Any]]:
        """列出所有条目的元数据

        Returns:
            条目元数据列表
        """
        return self._index.get("entries", [])

    def get_entry_ids(self) -> list[str]:
        """获取所有条目 ID

        Returns:
            条目 ID 列表
        """
        return [e["entry_id"] for e in self._index.get("entries", [])]

    def sample_opponents(
        self,
        n: int,
        strategy: str,
        target_type: AgentType | None = None,
        current_generation: int = 0,
    ) -> list[str]:
        """采样对手

        Args:
            n: 采样数量
            strategy: 采样策略 ('uniform', 'recency', 'diverse', 'pfsp')
            target_type: 目标物种类型（PFSP 策略需要）
            current_generation: 当前代数，用于 recency 衰减计算

        Returns:
            采样的条目 ID 列表
        """
        entries = self._index.get("entries", [])
        if not entries:
            return []

        if n >= len(entries):
            return [e["entry_id"] for e in entries]

        if strategy == 'uniform':
            return self._sample_uniform(entries, n)
        elif strategy == 'recency':
            return self._sample_recency_weighted(entries, n, current_generation)
        elif strategy == 'diverse':
            return self._sample_diverse(entries, n)
        elif strategy == 'pfsp':
            return self._sample_pfsp(entries, n, target_type, current_generation)
        else:
            # 默认使用时间加权
            return self._sample_recency_weighted(entries, n, current_generation)

    def _sample_uniform(self, entries: list[dict[str, Any]], n: int) -> list[str]:
        """均匀随机采样"""
        sampled = random.sample(entries, n)
        return [e["entry_id"] for e in sampled]

    def _sample_recency_weighted(
        self,
        entries: list[dict[str, Any]],
        n: int,
        current_generation: int = 0,
    ) -> list[str]:
        """时间加权采样：优先选择更新的历史对手

        使用指数衰减权重，代数越新权重越高。
        使用缓存避免每次采样重复计算权重。
        """
        self._rebuild_cache_if_needed(current_generation)
        if self._prob_cache is None or self._entry_ids_cache is None or len(self._entry_ids_cache) == 0:
            return []
        sampled = np.random.choice(
            self._entry_ids_cache,
            size=min(n, len(self._entry_ids_cache)),
            replace=False,
            p=self._prob_cache,
        )
        return list(sampled)

    def _sample_diverse(self, entries: list[dict[str, Any]], n: int) -> list[str]:
        """多样性采样：选择适应度分布较均匀的对手"""
        # 按适应度排序
        sorted_entries = sorted(entries, key=lambda e: e.get("avg_fitness", 0))

        # 均匀间隔采样
        step = len(sorted_entries) / n
        sampled: list[str] = []
        for i in range(n):
            idx = int(i * step)
            idx = min(idx, len(sorted_entries) - 1)
            sampled.append(sorted_entries[idx]["entry_id"])

        return sampled

    def update_entry_win_rate(
        self,
        entry_id: str,
        target_type: AgentType,
        outcome: float,
        ema_alpha: float = 0.3,
    ) -> None:
        """更新指定条目对特定目标类型的胜率（EMA 平滑）

        Args:
            entry_id: 条目 ID
            target_type: 目标物种类型（即 Main Agent 的类型）
            outcome: 本次对局结果（0.0~1.0）
            ema_alpha: EMA 平滑因子，越大越重视近期结果
        """
        key: str = f"vs_{target_type.value}"
        for entry_meta in self._index["entries"]:
            if entry_meta["entry_id"] != entry_id:
                continue

            # 更新 win_rates
            win_rates: dict[str, float | None] = entry_meta.get("win_rates", {})
            if win_rates is None:
                win_rates = {}
            old_win_rate = win_rates.get(key)
            if old_win_rate is None:
                win_rates[key] = outcome
            else:
                win_rates[key] = (1 - ema_alpha) * old_win_rate + ema_alpha * outcome
            entry_meta["win_rates"] = win_rates

            # 更新 match_counts
            match_counts: dict[str, int] = entry_meta.get("match_counts", {})
            if match_counts is None:
                match_counts = {}
            match_counts[key] = match_counts.get(key, 0) + 1
            entry_meta["match_counts"] = match_counts

            break

        self._invalidate_cache()

    def compute_weights(
        self,
        entries: list[dict[str, Any]],
        strategy: str,
        target_type: AgentType | None,
        current_generation: int,
    ) -> np.ndarray:
        """统一计算各策略的采样权重

        Args:
            entries: 条目元数据列表
            strategy: 采样策略
            target_type: 目标物种类型（PFSP 需要）
            current_generation: 当前代数

        Returns:
            归一化前的权重数组（长度与 entries 相同）
        """
        n_entries: int = len(entries)
        if n_entries == 0:
            return np.array([], dtype=np.float64)

        if strategy == 'uniform':
            return np.ones(n_entries, dtype=np.float64)

        decay_lambda: float = self.config.recency_decay_lambda
        milestone_interval: int = self.config.milestone_interval

        if strategy == 'recency':
            weights = np.array([
                math.exp(
                    -decay_lambda
                    * max(0, current_generation - e.get("source_generation", 0))
                    / milestone_interval
                )
                for e in entries
            ], dtype=np.float64)
            return weights

        if strategy == 'pfsp':
            if target_type is None:
                # 退化为 recency 权重
                return self.compute_weights(entries, 'recency', None, current_generation)

            pfsp_exponent: float = self.config.pfsp_exponent
            explore_bonus_coeff: float = self.config.pfsp_explore_bonus
            key: str = f"vs_{target_type.value}"

            weights = np.empty(n_entries, dtype=np.float64)
            for i, e in enumerate(entries):
                # Recency factor
                delta_gen: int = max(0, current_generation - e.get("source_generation", 0))
                recency_factor: float = math.exp(
                    -decay_lambda * delta_gen / milestone_interval
                )

                # Win rate factor + exploration bonus
                win_rates: dict[str, float | None] | None = e.get("win_rates")
                win_rate: float | None = None
                if win_rates is not None:
                    win_rate = win_rates.get(key)

                if win_rate is None:
                    f_win: float = 1.0
                    exploration_bonus: float = explore_bonus_coeff
                else:
                    f_win = (1.0 - win_rate) ** pfsp_exponent
                    match_counts: dict[str, int] | None = e.get("match_counts")
                    count: int = 0
                    if match_counts is not None:
                        count = match_counts.get(key, 0)
                    exploration_bonus = max(1.0, explore_bonus_coeff / math.sqrt(count + 1))

                weights[i] = f_win * recency_factor * exploration_bonus

            return weights

        # diverse 策略：返回均匀权重（batch 模式下特殊处理在 sample_opponents_batch 中）
        return np.ones(n_entries, dtype=np.float64)

    def _sample_pfsp(
        self,
        entries: list[dict[str, Any]],
        n: int,
        target_type: AgentType | None,
        current_generation: int,
    ) -> list[str]:
        """PFSP 采样策略：优先选择难以击败且较新的对手

        公式：p(opponent) ∝ f(win_rate) × recency_factor × exploration_bonus
          - f(win_rate) = (1 - win_rate)^p
          - recency_factor = exp(-λ × (current_gen - source_gen) / milestone_interval)
          - exploration_bonus = max(1.0, bonus / sqrt(match_count + 1))

        Args:
            entries: 条目元数据列表
            n: 采样数量
            target_type: 目标物种类型
            current_generation: 当前代数

        Returns:
            采样的条目 ID 列表
        """
        if target_type is None:
            return self._sample_recency_weighted(entries, n, current_generation)

        weights = self.compute_weights(entries, 'pfsp', target_type, current_generation)
        total: float = float(weights.sum())

        entry_ids: list[str] = [e["entry_id"] for e in entries]

        if total == 0:
            # 所有权重为 0，退化为均匀采样
            return self._sample_uniform(entries, n)

        sample_size: int = min(n, len(entry_ids))
        non_zero_count: int = int(np.count_nonzero(weights))
        if non_zero_count < sample_size:
            weights = np.ones(len(entry_ids), dtype=np.float64)
            total = float(weights.sum())

        probs: np.ndarray = weights / total
        sampled = np.random.choice(
            entry_ids,
            size=sample_size,
            replace=False,
            p=probs,
        )
        return list(sampled)

    def sample_opponents_batch(
        self,
        n: int,
        strategy: str,
        target_type: AgentType | None = None,
        current_generation: int = 0,
    ) -> list[str]:
        """批量采样 n 个不重复的对手（用于 n 个泛化竞技场）

        保证多样性：如果对手池 >= n，无放回采样；
        如果对手池 < n，每个对手至少出现 floor(n/pool_size) 次，
        剩余按策略权重无放回采样，最后随机打乱。

        Args:
            n: 需要的对手数量
            strategy: 采样策略
            target_type: 目标物种类型（PFSP 需要）
            current_generation: 当前代数

        Returns:
            长度为 n 的 entry_id 列表
        """
        entries: list[dict[str, Any]] = self._index.get("entries", [])
        if not entries:
            return []

        pool_size: int = len(entries)
        entry_ids: list[str] = [e["entry_id"] for e in entries]

        # diverse 策略特殊处理：按适应度排序后均匀间隔选取
        if strategy == 'diverse':
            sorted_entries = sorted(entries, key=lambda e: e.get("avg_fitness", 0))
            sorted_ids: list[str] = [e["entry_id"] for e in sorted_entries]
            if pool_size >= n:
                step: float = pool_size / n
                result: list[str] = []
                for i in range(n):
                    idx: int = min(int(i * step), pool_size - 1)
                    result.append(sorted_ids[idx])
                return result
            else:
                # pool_size < n: 每个至少出现 n // pool_size 次
                base_count: int = n // pool_size
                remainder: int = n % pool_size
                result = sorted_ids * base_count
                # 剩余用均匀间隔选取
                step = pool_size / remainder if remainder > 0 else 1.0
                for i in range(remainder):
                    idx = min(int(i * step), pool_size - 1)
                    result.append(sorted_ids[idx])
                random.shuffle(result)
                return result

        # 其他策略：使用统一权重计算
        weights: np.ndarray = self.compute_weights(
            entries, strategy, target_type, current_generation
        )
        total: float = float(weights.sum())

        if total == 0:
            # 所有权重为 0，使用均匀权重
            weights = np.ones(pool_size, dtype=np.float64)
            total = float(weights.sum())

        # 非零权重不足以无放回采样时，回退到均匀权重
        sample_needed: int = n if pool_size >= n else (n % pool_size)
        non_zero_count: int = int(np.count_nonzero(weights))
        if non_zero_count < sample_needed:
            weights = np.ones(pool_size, dtype=np.float64)
            total = float(weights.sum())

        probs: np.ndarray = weights / total

        if pool_size >= n:
            sampled = np.random.choice(
                entry_ids,
                size=n,
                replace=False,
                p=probs,
            )
            return list(sampled)
        else:
            # pool_size < n: 每个对手至少出现 n // pool_size 次
            base_count = n // pool_size
            remainder = n % pool_size
            result = entry_ids * base_count
            if remainder > 0:
                # 剩余按权重无放回采样
                extra = np.random.choice(
                    entry_ids,
                    size=remainder,
                    replace=False,
                    p=probs,
                )
                result.extend(list(extra))
            random.shuffle(result)
            return result

    def cleanup(self, current_generation: int) -> list[str]:
        """清理对手池，保持在最大大小限制内

        清理策略：保留里程碑和最近的条目，删除最旧的非里程碑条目。

        Args:
            current_generation: 当前代数

        Returns:
            被删除的条目 ID 列表
        """
        entries = self._index.get("entries", [])
        max_size = self.config.max_pool_size_per_type

        if len(entries) <= max_size:
            return []

        # 分类条目
        milestones: list[dict[str, Any]] = []
        others: list[dict[str, Any]] = []

        for entry in entries:
            if entry.get("add_reason") == "milestone":
                milestones.append(entry)
            else:
                others.append(entry)

        # 计算需要删除的数量
        to_remove_count = len(entries) - max_size
        removed: list[str] = []

        # 优先删除非里程碑的旧条目
        # 按创建时间排序（旧的在前）
        others.sort(key=lambda e: e.get("created_at", ""))

        for entry in others:
            if len(removed) >= to_remove_count:
                break
            self.remove_entry(entry["entry_id"], _skip_save_index=True)
            removed.append(entry["entry_id"])

        # 非里程碑不够删时，从里程碑中删除最旧的
        if len(removed) < to_remove_count:
            milestones.sort(key=lambda e: e.get("created_at", ""))
            for entry in milestones:
                if len(removed) >= to_remove_count:
                    break
                self.remove_entry(entry["entry_id"], _skip_save_index=True)
                removed.append(entry["entry_id"])

        # 批量删除后统一保存索引
        if removed:
            self.save_index()
            self._invalidate_cache()

        return removed

    def get_pool_size(self) -> int:
        """获取对手池大小"""
        return len(self._index.get("entries", []))

    def is_empty(self) -> bool:
        """检查对手池是否为空"""
        return self.get_pool_size() == 0

    def clear_memory_cache(self) -> None:
        """清理内存缓存中的大数据

        【内存泄漏修复】清理所有 entries 中的 genome_data 和 network_data，
        只保留元数据。这些大数据已保存到磁盘，需要时可重新加载。
        """
        for entry in self.entries.values():
            entry.genome_data = None
            entry.network_data = None
