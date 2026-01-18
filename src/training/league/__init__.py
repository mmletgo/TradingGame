"""联盟训练模块

参考 AlphaStar 联盟训练思路，实现完整的联盟训练机制。
"""
from src.training.league.arena_allocator import ArenaAllocator, ArenaAllocation, ArenaAssignment, AgentSourceConfig
from src.training.league.config import LeagueTrainingConfig
from src.training.league.exploiter_manager import ExploiterManager
from src.training.league.league_fitness import LeagueFitnessAggregator
from src.training.league.league_trainer import LeagueTrainer
from src.training.league.multi_gen_cache import MultiGenerationNetworkCache
from src.training.league.opponent_entry import OpponentMetadata, OpponentEntry
from src.training.league.opponent_pool import OpponentPool
from src.training.league.opponent_pool_manager import OpponentPoolManager

__all__ = [
    'AgentSourceConfig',
    'ArenaAllocator',
    'ArenaAllocation',
    'ArenaAssignment',
    'ExploiterManager',
    'LeagueFitnessAggregator',
    'LeagueTrainer',
    'LeagueTrainingConfig',
    'MultiGenerationNetworkCache',
    'OpponentEntry',
    'OpponentMetadata',
    'OpponentPool',
    'OpponentPoolManager',
]
