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
            with open(index_path, 'r', encoding='utf-8') as f:
                self._index = json.load(f)
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

        # 添加到内存
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
            del self.entries[entry_id]

        # 更新索引
        self._index["entries"] = [
            e for e in self._index["entries"] if e["entry_id"] != entry_id
        ]
        self.save_index()

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
                entry = OpponentEntry.load(entry_dir, load_networks=True)
                self.entries[entry_id] = entry
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
    ) -> list[str]:
        """采样对手

        Args:
            n: 采样数量
            strategy: 采样策略 ('uniform', 'pfsp', 'diverse')
            target_type: 目标类型（当前要训练的类型，用于 PFSP）

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
        elif strategy == 'pfsp':
            return self._sample_pfsp(entries, n, target_type)
        elif strategy == 'diverse':
            return self._sample_diverse(entries, n)
        else:
            return self._sample_uniform(entries, n)

    def _sample_uniform(self, entries: list[dict[str, Any]], n: int) -> list[str]:
        """均匀随机采样"""
        sampled = random.sample(entries, n)
        return [e["entry_id"] for e in sampled]

    def _sample_pfsp(
        self,
        entries: list[dict[str, Any]],
        n: int,
        target_type: AgentType | None,
    ) -> list[str]:
        """PFSP 采样：优先选择当前 Agent 更难战胜的历史对手

        使用 (1 - win_rate)^2 作为采样权重，胜率越低权重越高。
        """
        if target_type is None:
            return self._sample_uniform(entries, n)

        weights: list[float] = []
        for entry in entries:
            win_rates = entry.get("win_rates", {})
            # 获取目标类型对该历史对手的胜率
            win_rate_key = f"vs_{target_type.value}"
            win_rate = win_rates.get(win_rate_key, 0.5)
            # 胜率越低（越难战胜），采样权重越高
            weight = (1 - win_rate) ** 2
            weights.append(max(weight, 0.01))  # 避免权重为0

        # 归一化
        total = sum(weights)
        probs = [w / total for w in weights]

        # 按概率采样
        entry_ids = [e["entry_id"] for e in entries]
        sampled_ids = np.random.choice(
            entry_ids,
            size=min(n, len(entry_ids)),
            replace=False,
            p=probs,
        )
        return list(sampled_ids)

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

    def update_win_rate(
        self,
        entry_id: str,
        opponent_type: AgentType,
        won: bool,
    ) -> None:
        """更新胜率统计

        Args:
            entry_id: 条目 ID
            opponent_type: 对手类型
            won: 是否获胜
        """
        win_rate_key = f"vs_{opponent_type.value}"

        # 更新索引中的统计
        for entry in self._index["entries"]:
            if entry["entry_id"] == entry_id:
                if "win_rates" not in entry:
                    entry["win_rates"] = {}
                if "match_counts" not in entry:
                    entry["match_counts"] = {}

                # 获取当前统计
                current_win_rate = entry["win_rates"].get(win_rate_key, 0.5)
                current_count = entry["match_counts"].get(win_rate_key, 0)

                # 增量更新胜率
                new_count = current_count + 1
                win_value = 1.0 if won else 0.0
                new_win_rate = (current_win_rate * current_count + win_value) / new_count

                entry["win_rates"][win_rate_key] = new_win_rate
                entry["match_counts"][win_rate_key] = new_count

                # 更新内存中的条目
                if entry_id in self.entries:
                    cached_entry = self.entries[entry_id]
                    cached_entry.metadata.win_rates[win_rate_key] = new_win_rate
                    cached_entry.metadata.match_counts[win_rate_key] = new_count
                break

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
