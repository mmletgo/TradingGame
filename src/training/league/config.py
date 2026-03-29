"""联盟训练配置模块"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class LeagueTrainingConfig:
    """联盟训练配置

    混合竞技场方案：当前代和历史代Agent同场交易。
    """

    # 对手池配置
    pool_dir: str = "/mnt/work/TradingGame/league_training/opponent_pools"
    checkpoint_dir: str = (
        "/mnt/work/TradingGame/league_training/checkpoints"
    )
    max_pool_size_per_type: int = 100
    milestone_interval: int = 1

    # 竞技场运行配置
    num_arenas: int = 16  # 竞技场数量（对应物理核心数）
    episodes_per_arena: int = 4  # 每竞技场episode数

    # Per-arena 历史对手分配
    num_pure_arenas: int = 4  # 纯竞技场数量（无历史对手）
    num_retail_challenge_arenas: int = 6  # 散户挑战赛数量（当代散户 vs 历史MM全量）
    # num_mm_challenge_arenas 自动计算 = num_arenas - num_pure_arenas - num_retail_challenge_arenas

    # 历史对手配置
    num_historical_generations: int = 6  # 每轮采样历史代数
    historical_elite_ratio: float = 0.05  # 每代取Top 5%精英
    historical_freshness_ratio: float = 0.5  # 采样中最近历史的最低占比

    # 噪声交易者增强
    hybrid_noise_trader_count: int = 300  # 混合竞技场噪声交易者数
    hybrid_noise_trader_quantity_mu: float = 10.0  # 噪声交易者下单量mu

    # 采样策略
    sampling_strategy: Literal["uniform", "recency", "diverse", "pfsp"] = "pfsp"

    # PFSP 采样配置
    recency_decay_lambda: float = 2.0
    pfsp_exponent: float = 2.0
    pfsp_explore_bonus: float = 2.0
    pfsp_win_rate_ema_alpha: float = 0.3

    # 代际对比配置
    generational_comparison_window: int = 20  # 代际对比历史窗口
    convergence_fitness_std_threshold: float = 0.005  # 适应度标准差收敛阈值
    elite_ratio: float = 0.1  # 精英比例

    # 冻结与复评配置
    freeze_on_convergence: bool = True
    freeze_thaw_threshold: float = 0.05
    min_freeze_generation: int = 30
    convergence_generations: int = 10  # 连续满足收敛条件的代数

    def __post_init__(self) -> None:
        """初始化后自动校验配置"""
        self.validate()

    def validate(self) -> None:
        """验证配置有效性"""
        if not self.pool_dir or not self.pool_dir.strip():
            raise ValueError("pool_dir must not be empty")
        if not self.checkpoint_dir or not self.checkpoint_dir.strip():
            raise ValueError("checkpoint_dir must not be empty")
        if self.max_pool_size_per_type < 1:
            raise ValueError("max_pool_size_per_type must be at least 1")
        if self.milestone_interval < 1:
            raise ValueError("milestone_interval must be at least 1")
        if self.num_arenas < 1:
            raise ValueError("num_arenas must be at least 1")
        if self.num_pure_arenas < 0:
            raise ValueError("num_pure_arenas must be >= 0")
        if self.num_retail_challenge_arenas < 0:
            raise ValueError("num_retail_challenge_arenas must be >= 0")
        if self.num_pure_arenas + self.num_retail_challenge_arenas > self.num_arenas:
            raise ValueError(
                "num_pure_arenas + num_retail_challenge_arenas must be <= num_arenas"
            )
        if self.episodes_per_arena < 1:
            raise ValueError("episodes_per_arena must be at least 1")
        if not 0.0 < self.freeze_thaw_threshold < 1.0:
            raise ValueError(
                "freeze_thaw_threshold must be between 0.0 and 1.0 (exclusive)"
            )
        if self.num_historical_generations < 1:
            raise ValueError("num_historical_generations must be at least 1")
        if not 0.0 < self.historical_elite_ratio <= 1.0:
            raise ValueError(
                "historical_elite_ratio must be between 0.0 (exclusive) and 1.0 (inclusive)"
            )
        if not 0.0 <= self.historical_freshness_ratio <= 1.0:
            raise ValueError(
                "historical_freshness_ratio must be between 0.0 and 1.0"
            )
        # 收敛阈值校验
        if self.convergence_fitness_std_threshold <= 0:
            raise ValueError(
                "convergence_fitness_std_threshold must be > 0"
            )
        if self.recency_decay_lambda <= 0:
            raise ValueError("recency_decay_lambda must be > 0")
        if self.pfsp_exponent <= 0:
            raise ValueError("pfsp_exponent must be > 0")
        if not 0.0 < self.pfsp_win_rate_ema_alpha <= 1.0:
            raise ValueError(
                "pfsp_win_rate_ema_alpha must be between 0.0 (exclusive) and 1.0 (inclusive)"
            )
        if not 0.0 < self.elite_ratio <= 1.0:
            raise ValueError(
                "elite_ratio must be between 0.0 (exclusive) and 1.0 (inclusive)"
            )
        if self.min_freeze_generation < 0:
            raise ValueError("min_freeze_generation must be >= 0")
        if self.convergence_generations < 1:
            raise ValueError("convergence_generations must be >= 1")
        if self.generational_comparison_window < 1:
            raise ValueError("generational_comparison_window must be >= 1")
        if self.pfsp_explore_bonus < 0:
            raise ValueError("pfsp_explore_bonus must be >= 0")
        if self.hybrid_noise_trader_count < 0:
            raise ValueError("hybrid_noise_trader_count must be >= 0")
        if self.convergence_generations > self.generational_comparison_window:
            raise ValueError(
                "convergence_generations must be <= generational_comparison_window"
            )
