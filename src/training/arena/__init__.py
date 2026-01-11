"""竞技场模块

本模块提供多竞技场训练相关的组件。
"""

from .arena_state import (
    AgentAccountState,
    AgentStateAdapter,
    ArenaState,
    CatfishAccountState,
)
from .fitness_aggregator import FitnessAggregator
from .parallel_arena_trainer import MultiArenaConfig, ParallelArenaTrainer

__all__ = [
    "AgentAccountState",
    "AgentStateAdapter",
    "ArenaState",
    "CatfishAccountState",
    "FitnessAggregator",
    "MultiArenaConfig",
    "ParallelArenaTrainer",
]
