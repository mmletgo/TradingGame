# -*- coding: utf-8 -*-
"""共享检查点管理模块

每个竞技场使用独立的 checkpoint 文件，避免锁竞争。
目录结构：
    checkpoints/multi_arena/
        arena_0/
            checkpoint.pkl
        arena_1/
            checkpoint.pkl
        ...

写入时只写自己的文件（原子写入），读取其他竞技场时直接读取。
"""

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any

from src.core.log_engine.logger import get_logger


@dataclass
class ArenaCheckpointData:
    """单个竞技场的检查点数据

    Attributes:
        arena_id: 竞技场唯一标识
        episode: 当前 episode 编号
        best_genomes: 各 Agent 类型的最佳基因组
            格式: {agent_type.value: [(genome_data, fitness), ...]}
        populations: 完整种群数据用于恢复
        updated_at: 最后更新时间戳
    """

    arena_id: int
    episode: int = 0
    best_genomes: dict[str, list[tuple[bytes, float]]] = field(default_factory=dict)
    populations: dict[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0


@dataclass
class SharedCheckpointData:
    """共享检查点数据（用于恢复时的聚合视图）

    Attributes:
        version: 检查点版本号
        arenas: 各竞技场的检查点数据
            格式: {arena_id: ArenaCheckpointData}
        config: 训练配置
    """

    version: int = 0
    arenas: dict[int, ArenaCheckpointData] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


class SharedCheckpointManager:
    """共享检查点管理器（独立文件模式）

    每个竞技场使用独立的 checkpoint 文件，完全避免锁竞争。
    - 写入：只写自己的文件，使用原子写入
    - 读取：遍历其他竞技场目录，读取最新 checkpoint

    Attributes:
        checkpoint_dir: 检查点根目录
        num_arenas: 竞技场数量（用于遍历）
    """

    checkpoint_dir: str
    num_arenas: int
    _logger: Any

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/multi_arena",
    ) -> None:
        """初始化检查点管理器

        Args:
            checkpoint_path: 检查点目录路径
        """
        # 兼容旧接口：如果是 .pkl 文件路径，转换为目录
        if checkpoint_path.endswith(".pkl"):
            base_dir = os.path.dirname(checkpoint_path)
            self.checkpoint_dir = os.path.join(base_dir, "multi_arena")
        else:
            self.checkpoint_dir = checkpoint_path

        self.num_arenas = 0
        self._logger = get_logger(__name__)

    def _get_arena_dir(self, arena_id: int) -> str:
        """获取竞技场的 checkpoint 目录"""
        return os.path.join(self.checkpoint_dir, f"arena_{arena_id}")

    def _get_arena_checkpoint_path(self, arena_id: int) -> str:
        """获取竞技场的 checkpoint 文件路径"""
        return os.path.join(self._get_arena_dir(arena_id), "checkpoint.pkl")

    def initialize(self, config: dict[str, Any], num_arenas: int) -> None:
        """初始化检查点目录结构

        为每个竞技场创建独立目录。

        Args:
            config: 训练配置
            num_arenas: 竞技场数量
        """
        self.num_arenas = num_arenas

        # 创建根目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 为每个竞技场创建目录
        for arena_id in range(num_arenas):
            arena_dir = self._get_arena_dir(arena_id)
            os.makedirs(arena_dir, exist_ok=True)

        # 保存配置到根目录
        config_path = os.path.join(self.checkpoint_dir, "config.pkl")
        if not os.path.exists(config_path):
            with open(config_path, "wb") as f:
                pickle.dump({"config": config, "num_arenas": num_arenas}, f)

        self._logger.info(
            f"初始化检查点目录: {self.checkpoint_dir}, 竞技场数量: {num_arenas}"
        )

    def update_arena(
        self,
        arena_id: int,
        episode: int,
        populations: dict[str, Any],
        best_genomes: dict[str, list[tuple[bytes, float]]],
    ) -> None:
        """更新单个竞技场的 checkpoint（原子写入，无锁）

        Args:
            arena_id: 竞技场 ID
            episode: 当前 episode 编号
            populations: 完整种群数据
            best_genomes: 各 Agent 类型的最佳基因组
        """
        arena_data = ArenaCheckpointData(
            arena_id=arena_id,
            episode=episode,
            best_genomes=best_genomes,
            populations=populations,
            updated_at=time.time(),
        )

        checkpoint_path = self._get_arena_checkpoint_path(arena_id)
        self._atomic_write(checkpoint_path, arena_data)

        self._logger.debug(
            f"竞技场 {arena_id} 保存检查点: episode={episode}"
        )

    def get_migration_candidates(
        self,
        requesting_arena_id: int,
        count_per_arena: int = 5,
    ) -> dict[str, list[tuple[bytes, float]]]:
        """获取其他竞技场的最佳个体

        遍历其他竞技场的 checkpoint 文件，收集最佳基因组。

        Args:
            requesting_arena_id: 请求迁移的竞技场 ID（排除自身）
            count_per_arena: 从每个竞技场获取的个体数量

        Returns:
            各 Agent 类型的候选基因组
        """
        candidates: dict[str, list[tuple[bytes, float]]] = {}

        # 遍历所有竞技场目录
        if not os.path.exists(self.checkpoint_dir):
            return candidates

        for entry in os.listdir(self.checkpoint_dir):
            if not entry.startswith("arena_"):
                continue

            try:
                arena_id = int(entry.split("_")[1])
            except (IndexError, ValueError):
                continue

            # 跳过请求方自己
            if arena_id == requesting_arena_id:
                continue

            # 读取该竞技场的 checkpoint
            checkpoint_path = self._get_arena_checkpoint_path(arena_id)
            arena_data = self._read_arena_checkpoint(checkpoint_path)

            if arena_data is None or not arena_data.best_genomes:
                continue

            # 收集最佳基因组
            for agent_type, genomes in arena_data.best_genomes.items():
                if agent_type not in candidates:
                    candidates[agent_type] = []
                selected = genomes[:count_per_arena]
                candidates[agent_type].extend(selected)

        # 按适应度排序（降序）
        for agent_type in candidates:
            candidates[agent_type].sort(key=lambda x: x[1], reverse=True)

        self._logger.debug(
            f"竞技场 {requesting_arena_id} 获取迁移候选者: "
            f"{sum(len(v) for v in candidates.values())} 个"
        )

        return candidates

    def get_full_checkpoint(self) -> SharedCheckpointData | None:
        """获取完整检查点（聚合所有竞技场，用于恢复）

        Returns:
            聚合的共享检查点数据
        """
        if not os.path.exists(self.checkpoint_dir):
            return None

        # 读取配置
        config_path = os.path.join(self.checkpoint_dir, "config.pkl")
        config_data: dict[str, Any] = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "rb") as f:
                    config_data = pickle.load(f)
            except (pickle.PickleError, EOFError, OSError):
                pass

        # 聚合所有竞技场数据
        arenas: dict[int, ArenaCheckpointData] = {}
        max_episode = 0

        for entry in os.listdir(self.checkpoint_dir):
            if not entry.startswith("arena_"):
                continue

            try:
                arena_id = int(entry.split("_")[1])
            except (IndexError, ValueError):
                continue

            checkpoint_path = self._get_arena_checkpoint_path(arena_id)
            arena_data = self._read_arena_checkpoint(checkpoint_path)

            if arena_data is not None:
                arenas[arena_id] = arena_data
                max_episode = max(max_episode, arena_data.episode)

        if not arenas:
            return None

        data = SharedCheckpointData(
            version=max_episode,
            arenas=arenas,
            config=config_data.get("config", {}),
        )

        self._logger.info(
            f"读取完整检查点: 竞技场数量={len(arenas)}, 最大 episode={max_episode}"
        )

        return data

    def _read_arena_checkpoint(self, checkpoint_path: str) -> ArenaCheckpointData | None:
        """读取单个竞技场的 checkpoint 文件

        Args:
            checkpoint_path: checkpoint 文件路径

        Returns:
            竞技场检查点数据，如果文件不存在或损坏则返回 None
        """
        if not os.path.exists(checkpoint_path):
            return None

        try:
            with open(checkpoint_path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, OSError) as e:
            self._logger.warning(f"读取检查点文件失败 {checkpoint_path}: {e}")
            return None

    def _atomic_write(self, path: str, data: ArenaCheckpointData) -> None:
        """原子写入检查点文件

        先写入临时文件，再用 os.replace() 原子重命名。

        Args:
            path: 目标文件路径
            data: 要写入的数据
        """
        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        temp_path = f"{path}.tmp"
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(data, f)
            os.replace(temp_path, path)
        except OSError as e:
            self._logger.error(f"写入检查点文件失败 {path}: {e}")
            # 清理临时文件
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise
