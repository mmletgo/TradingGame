"""
配置模块

定义系统的各类配置数据类。
"""

from dataclasses import dataclass, field
from enum import Enum


class AgentType(str, Enum):
    RETAIL_PRO = "RETAIL_PRO"
    MARKET_MAKER = "MARKET_MAKER"


@dataclass
class MarketConfig:
    """
    市场配置

    定义交易市场的核心参数。

    Attributes:
        initial_price: 初始价格
        tick_size: 最小变动单位
        lot_size: 最小交易单位
        depth: 盘口深度（买卖各多少档）
        ema_alpha: EMA 平滑系数（0-1，值越小价格变化越平滑）
    """

    initial_price: float
    tick_size: float
    lot_size: float
    depth: int
    ema_alpha: float = 0.5


@dataclass
class AgentConfig:
    """
    Agent 配置

    定义特定类型 Agent 的交易参数。

    Attributes:
        count: 数量
        initial_balance: 初始资产
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率
        maker_fee_rate: 挂单费率
        taker_fee_rate: 吃单费率
    """

    count: int
    initial_balance: float
    leverage: float
    maintenance_margin_rate: float
    maker_fee_rate: float
    taker_fee_rate: float


@dataclass
class TrainingConfig:
    """
    训练配置

    定义 NEAT 进化训练的参数。

    Attributes:
        episode_length: 每个 episode 的 tick 数量（默认 1000）
        checkpoint_interval: 检查点间隔（episode 数）
        neat_config_path: NEAT 配置文件路径
        parallel_workers: 并行工作进程数（默认 16）
        enable_parallel_evolution: 是否启用并行进化（默认 True）
        enable_parallel_decision: 是否启用并行决策（默认 True）
        enable_parallel_creation: 是否启用并行创建（默认 True）
        openmp_threads: OpenMP 并行线程数（默认 16，建议设为物理核心数）
        random_seed: 随机种子（默认 None，表示不固定）
        retail_pro_sub_population_count: 高级散户子种群数量（默认 12）
        evolution_interval: 每多少个 episode 进化一次（默认 10）
        num_arenas: 竞技场数量（默认 2）
        episodes_per_arena: 每个竞技场运行的 episode 数（默认 50）
        mm_fitness_pnl_weight: 做市商复合适应度中 PnL 收益率的权重 α（默认 0.4）
        mm_fitness_spread_weight: 做市商复合适应度中盘口价差质量的权重 β（默认 0.3）
        mm_fitness_volume_weight: 做市商复合适应度中 Maker 成交量的权重 γ（默认 0.2）
        mm_fitness_survival_weight: 做市商复合适应度中存活的权重 δ（默认 0.1）
    """

    episode_length: int
    checkpoint_interval: int
    neat_config_path: str
    # 并行化配置
    parallel_workers: int = 16
    enable_parallel_evolution: bool = True
    enable_parallel_decision: bool = True
    enable_parallel_creation: bool = True
    openmp_threads: int = 16  # 默认 16，建议设为物理核心数
    random_seed: int | None = None
    # 高级散户子种群配置
    retail_pro_sub_population_count: int = 12
    # 进化间隔配置
    evolution_interval: int = 10
    # 多竞技场配置
    num_arenas: int = 2
    episodes_per_arena: int = 50
    # 做市商复合适应度权重（α+β+γ+δ=1.0）
    mm_fitness_pnl_weight: float = 0.4        # α: PnL 收益率
    mm_fitness_spread_weight: float = 0.3      # β: 盘口价差质量
    mm_fitness_volume_weight: float = 0.2      # γ: Maker 成交量
    mm_fitness_survival_weight: float = 0.1    # δ: 存活


@dataclass
class DemoConfig:
    """
    演示配置

    定义 WebUI 演示的参数。

    Attributes:
        host: 服务器地址
        port: 服务器端口
        tick_interval: tick 间隔（毫秒）
    """

    host: str
    port: int
    tick_interval: int


@dataclass
class NoiseTraderConfig:
    """噪声交易者配置"""
    count: int = 100
    action_probability: float = 0.5
    quantity_mu: float = 3.0
    quantity_sigma: float = 1.0


@dataclass
class Config:
    """
    全局配置

    汇总所有配置项。

    Attributes:
        market: 市场配置
        agents: Agent 配置（按类型）
        training: 训练配置
        demo: 演示配置
        noise_trader: 噪声交易者配置
    """

    market: MarketConfig
    agents: dict[AgentType, AgentConfig]
    training: TrainingConfig
    demo: DemoConfig
    noise_trader: NoiseTraderConfig = field(default_factory=NoiseTraderConfig)
