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
        openmp_threads: OpenMP 并行线程数（默认 1，建议设为物理核心数）
        random_seed: 随机种子（默认 None，表示不固定）
        retail_pro_sub_population_count: 高级散户子种群数量（默认 10）
        evolution_interval: 每多少个 episode 进化一次（默认 10）
        num_arenas: 竞技场数量（默认 16）
        episodes_per_arena: 每个竞技场运行的 episode 数（默认 4）
        mm_fitness_pnl_weight: 做市商复合适应度中 PnL 收益率的权重 α（默认 0.7）
        mm_fitness_volume_weight: 做市商复合适应度中 Maker 成交量的权重 γ（默认 0.3）
        position_cost_weight: 散户持仓成本权重（默认 0.02），惩罚持仓不平仓
        mm_position_cost_weight: 做市商持仓成本权重（默认 0.005），做市商需持仓做市故权重更小
    """

    episode_length: int
    checkpoint_interval: int
    neat_config_path: str
    # 并行化配置
    parallel_workers: int = 16
    enable_parallel_evolution: bool = True
    enable_parallel_decision: bool = True
    enable_parallel_creation: bool = True
    openmp_threads: int = 1  # 默认 1，建议设为物理核心数
    random_seed: int | None = None
    # 高级散户子种群配置
    retail_pro_sub_population_count: int = 10
    # 进化间隔配置
    evolution_interval: int = 10
    # 多竞技场配置
    num_arenas: int = 16
    episodes_per_arena: int = 4
    # 做市商复合适应度权重（α+γ=1.0）
    # mm_fitness = α × pnl + γ × volume_score
    mm_fitness_pnl_weight: float = 0.7        # α: PnL 收益率
    mm_fitness_volume_weight: float = 0.3      # γ: Maker 成交量
    # 持仓成本权重（对称的 position size penalty）
    position_cost_weight: float = 0.02       # 散户持仓成本权重
    mm_position_cost_weight: float = 0.005   # 做市商持仓成本权重（做市商需持仓做市，权重更小）
    # CPU 亲和性
    enable_cpu_affinity: bool = True  # 是否将 Arena Worker 进程绑定到独立的物理 CPU 核心


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
    count: int = 200
    action_probability: float = 0.5
    quantity_mu: float = 12.0
    quantity_sigma: float = 1.0
    episode_bias_range: float = 0.15  # Episode 级买入概率偏置范围，buy_prob ∈ [0.5-range, 0.5+range]
    ou_theta: float = 0.035  # OU 过程均值回归速度（每 tick 回归 3.5% 的偏差）
    ou_sigma: float = 0.04   # OU 过程噪声强度


@dataclass
class ASConfig:
    """Avellaneda-Stoikov 最优做市模型配置

    Attributes:
        gamma: 基础风险厌恶系数
        kappa_base: 基础订单到达率
        vol_window: 波动率回看窗口 (ticks)
        min_sigma: 波动率下限
        max_sigma: 波动率上限
        gamma_adj_min: NN gamma 调整下限（乘数）
        gamma_adj_max: NN gamma 调整上限（乘数）
        spread_adj_min: NN spread 调整下限（乘数）
        spread_adj_max: NN spread 调整上限（乘数）
        max_reservation_offset: reservation price 最大偏移比例
    """

    gamma: float = 0.1
    kappa_base: float = 1.5
    vol_window: int = 50
    min_sigma: float = 1e-6
    max_sigma: float = 0.1
    gamma_adj_min: float = 0.1
    gamma_adj_max: float = 10.0
    spread_adj_min: float = 0.5
    spread_adj_max: float = 2.0
    max_reservation_offset: float = 0.05


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
        as_model: AS 最优做市模型配置
    """

    market: MarketConfig
    agents: dict[AgentType, AgentConfig]
    training: TrainingConfig
    demo: DemoConfig
    noise_trader: NoiseTraderConfig = field(default_factory=NoiseTraderConfig)
    as_model: ASConfig = field(default_factory=ASConfig)
