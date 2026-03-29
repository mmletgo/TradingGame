"""联盟训练模块

参考 AlphaStar 联盟训练思路，实现完整的联盟训练机制。
"""
from src.training.league.arena_allocator import HybridArenaAllocator, HybridSamplingResult, PerArenaAllocation
from src.training.league.config import LeagueTrainingConfig
from src.training.league.league_fitness import HybridFitnessAggregator, GenerationalComparisonStats
from src.training.league.league_trainer import LeagueTrainer, extract_elite_networks
from src.training.league.opponent_entry import OpponentMetadata, OpponentEntry
from src.training.league.opponent_pool import OpponentPool
from src.training.league.opponent_pool_manager import OpponentPoolManager

__all__ = [
    'HybridArenaAllocator',
    'HybridSamplingResult',
    'PerArenaAllocation',
    'HybridFitnessAggregator',
    'GenerationalComparisonStats',
    'LeagueTrainer',
    'extract_elite_networks',
    'LeagueTrainingConfig',
    'OpponentEntry',
    'OpponentMetadata',
    'OpponentPool',
    'OpponentPoolManager',
]
