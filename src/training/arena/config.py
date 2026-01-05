"""多竞技场配置模块"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.config import Config


class MigrationStrategy(Enum):
    """迁移策略枚举"""
    RING = "ring"                    # 环形迁移：Arena[i] -> Arena[(i+1) % N]
    RANDOM = "random"                # 随机迁移
    BEST_TO_WORST = "best_to_worst"  # 最好竞技场迁移到最差竞技场


@dataclass
class ArenaConfig:
    """单个竞技场配置

    Attributes:
        arena_id: 竞技场唯一标识
        config: 训练配置（所有竞技场共用同一配置）
        seed: 随机种子（不同种子产生不同市场特征）
        migration_interval: 竞技场内部迁移间隔（episode 数）
        checkpoint_interval: 检查点保存间隔（episode 数）
        max_episodes: 最大 episode 数
        checkpoint_dir: 检查点目录（各竞技场独立文件模式）
    """
    arena_id: int
    config: "Config"
    seed: int | None = None
    migration_interval: int = 10
    checkpoint_interval: int = 50
    max_episodes: int = 4000
    checkpoint_dir: str = "checkpoints/multi_arena"


@dataclass
class MultiArenaConfig:
    """多竞技场配置

    Attributes:
        num_arenas: 竞技场数量
        base_config: 基础配置（所有竞技场共用）
        migration_interval: 迁移间隔（episode 数）
        migration_count: 每次迁移的 Agent 数量（每种群）
        migration_best_ratio: 迁移最好个体的比例（0.5 表示一半最好一半最差）
        migration_strategy: 迁移策略
        checkpoint_interval: 检查点保存间隔
        seed_offset: 随机种子偏移量
        checkpoint_dir: 检查点目录（各竞技场独立文件模式）
        max_episodes: 最大 episode 数
    """
    num_arenas: int = 10
    base_config: "Config | None" = None
    migration_interval: int = 10
    migration_count: int = 5
    migration_best_ratio: float = 0.5
    migration_strategy: MigrationStrategy = MigrationStrategy.RING
    checkpoint_interval: int = 50
    seed_offset: int = 0
    checkpoint_dir: str = "checkpoints/multi_arena"
    max_episodes: int = 4000
