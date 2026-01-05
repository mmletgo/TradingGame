"""多竞技场训练模块

提供多竞技场并行训练功能，支持 Agent 在竞技场之间迁移。
"""

from src.training.arena.arena import Arena
from src.training.arena.arena_manager import (
    ArenaManager,
    ArenaProcess,
    ArenaProcessInfo,
    arena_worker,
    arena_worker_autonomous,
)
from src.training.arena.config import ArenaConfig, MigrationStrategy, MultiArenaConfig
from src.training.arena.metrics import ArenaMetrics, EpisodeMetrics, MetricsAggregator
from src.training.arena.migration import MigrationPacket, MigrationSystem
from src.training.arena.shared_checkpoint import (
    ArenaCheckpointData,
    SharedCheckpointData,
    SharedCheckpointManager,
)

__all__ = [
    # 配置
    "ArenaConfig",
    "MultiArenaConfig",
    "MigrationStrategy",
    # 核心类
    "Arena",
    "ArenaManager",
    "ArenaProcess",  # 已弃用，保留用于兼容
    "ArenaProcessInfo",  # 推荐
    "arena_worker",  # 已弃用，保留用于兼容
    "arena_worker_autonomous",  # 推荐
    # 迁移
    "MigrationPacket",
    "MigrationSystem",
    # 指标
    "ArenaMetrics",
    "EpisodeMetrics",
    "MetricsAggregator",
    # 共享检查点
    "ArenaCheckpointData",
    "SharedCheckpointData",
    "SharedCheckpointManager",
]
