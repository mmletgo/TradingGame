"""联盟训练配置模块"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LeagueTrainingConfig:
    """联盟训练配置

    参考 AlphaStar 联盟训练思路的配置类。
    """

    # 对手池配置
    pool_dir: str = "checkpoints/league_training/opponent_pools"
    max_pool_size_per_type: int = 20  # 每种类型最多保留20个历史版本
    milestone_interval: int = 50  # 每50代保存里程碑

    # 竞技场运行配置
    num_arenas: int = 64  # 同时运行的竞技场数量
    episodes_per_arena: int = 1  # 每个竞技场运行的 episode 数

    # 竞技场分配
    num_baseline_arenas: int = 16  # 基准竞技场数量
    num_generalization_arenas_per_type: int = 12  # 每类型泛化测试竞技场数量

    # 适应度计算
    fitness_strategy: Literal['simple', 'weighted_average', 'min'] = 'weighted_average'
    baseline_weight: float = 1.0  # 基准竞技场权重
    generalization_weight: float = 0.8  # 泛化测试竞技场权重

    # 采样策略
    sampling_strategy: Literal['uniform', 'recency', 'diverse'] = 'recency'

    # 对手池注入条件
    elite_fitness_threshold: float = 0.05  # 平均适应度超过历史最高 5% 时注入
    diversity_threshold: float = 0.3  # 与现有对手差异阈值

    def validate(self) -> None:
        """验证配置有效性"""
        if self.max_pool_size_per_type < 1:
            raise ValueError("max_pool_size_per_type must be at least 1")
        if self.milestone_interval < 1:
            raise ValueError("milestone_interval must be at least 1")
        if self.num_arenas < 1:
            raise ValueError("num_arenas must be at least 1")
        if self.episodes_per_arena < 1:
            raise ValueError("episodes_per_arena must be at least 1")

    def get_arena_allocation(self) -> dict[str, int]:
        """获取竞技场分配详情

        Returns:
            各类型竞技场的数量分配
        """
        total_generalization = self.num_generalization_arenas_per_type * 4
        total = self.num_baseline_arenas + total_generalization

        return {
            'baseline': self.num_baseline_arenas,
            'generalization_total': total_generalization,
            'generalization_per_type': self.num_generalization_arenas_per_type,
            'total': total,
        }
