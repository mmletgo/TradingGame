"""
配置模块

定义系统的各类配置数据类。
"""

from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """Agent 类型枚举

    定义系统中五种 AI Agent 的类型。
    """

    RETAIL = "RETAIL"  # 散户
    RETAIL_PRO = "RETAIL_PRO"  # 高级散户
    BULL_WHALE = "BULL_WHALE"  # 多头庄家
    BEAR_WHALE = "BEAR_WHALE"  # 空头庄家
    MARKET_MAKER = "MARKET_MAKER"  # 做市商


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
    ema_alpha: float = 0.1


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
        random_seed: 随机种子（默认 None，表示不固定）
    """

    episode_length: int
    checkpoint_interval: int
    neat_config_path: str
    # 并行化配置
    parallel_workers: int = 16
    enable_parallel_evolution: bool = True
    enable_parallel_decision: bool = True
    enable_parallel_creation: bool = True
    random_seed: int | None = None


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
class Config:
    """
    全局配置

    汇总所有配置项。

    Attributes:
        market: 市场配置
        agents: Agent 配置（按类型）
        training: 训练配置
        demo: 演示配置
    """

    market: MarketConfig
    agents: dict[AgentType, AgentConfig]
    training: TrainingConfig
    demo: DemoConfig
