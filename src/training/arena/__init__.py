"""多竞技场训练模块

提供多竞技场并行训练功能，支持 Agent 在竞技场之间迁移。
"""

from src.training.arena.arena import Arena
from src.training.arena.arena_manager import ArenaManager, ArenaProcess
from src.training.arena.config import ArenaConfig, MigrationStrategy, MultiArenaConfig
from src.training.arena.metrics import ArenaMetrics, EpisodeMetrics, MetricsAggregator
from src.training.arena.migration import MigrationPacket, MigrationSystem

__all__ = [
    # 配置
    "ArenaConfig",
    "MultiArenaConfig",
    "MigrationStrategy",
    # 核心类
    "Arena",
    "ArenaManager",
    "ArenaProcess",
    # 迁移
    "MigrationPacket",
    "MigrationSystem",
    # 指标
    "ArenaMetrics",
    "EpisodeMetrics",
    "MetricsAggregator",
]
