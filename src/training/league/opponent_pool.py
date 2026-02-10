"""单类型对手池管理器"""
from __future__ import annotations

import json
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

    def _invalidate_cache(self) -> None:
        """失效缓存"""
        self._prob_cache = None
        self._entry_ids_cache = None

    def _rebuild_cache_if_needed(self) -> None:
        """按需重建缓存"""
        entries = self._index.get("entries", [])
        current_version = len(entries)

        if self._prob_cache is None or self._cache_version != current_version:
            self._entry_ids_cache = [e["entry_id"] for e in entries]
            weights = np.array([
                float(max(e.get("source_generation", 1), 1))
                for e in entries
            ], dtype=np.float64)
            self._prob_cache = weights / weights.sum() if len(weights) > 0 else None
            self._cache_version = current_version

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
        """保存索引文件"""
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.pool_dir / "pool_index.json"
        self._index["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)

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

    def remove_entry(self, entry_id: str) -> None:
        """移除条目

        Args:
            entry_id: 条目 ID
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
        target_type: AgentType | None = None,  # 不再使用，保持接口兼容
    ) -> list[str]:
        """采样对手

        Args:
            n: 采样数量
            strategy: 采样策略 ('uniform', 'recency', 'diverse')
            target_type: 不再使用，保持接口兼容

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
            return self._sample_recency_weighted(entries, n)
        elif strategy == 'diverse':
            return self._sample_diverse(entries, n)
        else:
            # 默认使用时间加权
            return self._sample_recency_weighted(entries, n)

    def _sample_uniform(self, entries: list[dict[str, Any]], n: int) -> list[str]:
        """均匀随机采样"""
        sampled = random.sample(entries, n)
        return [e["entry_id"] for e in sampled]

    def _sample_recency_weighted(
        self,
        entries: list[dict[str, Any]],
        n: int,
    ) -> list[str]:
        """时间加权采样：优先选择更新的历史对手

        使用 source_generation 作为权重，代数越新权重越高。
        使用缓存避免每次采样重复计算权重。
        """
        self._rebuild_cache_if_needed()
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
            self.remove_entry(entry["entry_id"])
            removed.append(entry["entry_id"])

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
