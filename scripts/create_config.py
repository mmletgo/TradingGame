import importlib
import sys
from pathlib import Path

# 关键：在导入任何项目模块之前，先清除 importlib 缓存
# 这可以解决修改代码后由于缓存导致的运行时问题
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import (
    AgentConfig,
    AgentType,
    CatfishConfig,
    Config,
    DemoConfig,
    MarketConfig,
    TrainingConfig,
)


def create_default_config(
    episode_length: int = 1000,
    checkpoint_interval: int = 10,
    config_dir: str = "config",
    catfish_enabled: bool = False,
    evolution_interval: int = 10,
) -> Config:
    """创建默认配置

    Args:
        episode_length: 每个 episode 的 tick 数量
        checkpoint_interval: 检查点间隔（episode 数）
        config_dir: 配置文件目录（Population 会在此目录下查找对应的 NEAT 配置）
        catfish_enabled: 是否启用鲶鱼（启用后三种行为模式同时运行）
        evolution_interval: 每多少个 episode 进化一次

    Returns:
        默认配置对象
    """
    market = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
        ema_alpha=0.9,
    )
    maker_initial_balance = 10_000_000.0  # 做市商初始资金 10M
    maker_leverage = 1.0

    agents = {
        AgentType.RETAIL: AgentConfig(
            count=10000,
            initial_balance=20000.0,  # 2万
            leverage=1.0,
            maintenance_margin_rate=0.5,  # 50%
            maker_fee_rate=0.0002,  # 万2
            taker_fee_rate=0.0005,  # 万5
        ),
        AgentType.RETAIL_PRO: AgentConfig(
            count=100,
            initial_balance=20000.0,  # 2万
            leverage=1.0,
            maintenance_margin_rate=0.5,  # 50%
            maker_fee_rate=0.0002,  # 万2
            taker_fee_rate=0.0005,  # 万5
        ),
        AgentType.WHALE: AgentConfig(
            count=100,  # 庄家（合并多空）
            initial_balance=3_000_000.0,  # 3M
            leverage=1.0,
            maintenance_margin_rate=0.5,  # 50%
            maker_fee_rate=-0.0001,  # 负万1 (maker rebate)
            taker_fee_rate=0.0001,  # 万1
        ),
        AgentType.MARKET_MAKER: AgentConfig(
            count=400,
            initial_balance=maker_initial_balance,
            leverage=maker_leverage,
            maintenance_margin_rate=0.5 / maker_leverage,
            maker_fee_rate=-0.0001,  # 负万1 (maker rebate)
            taker_fee_rate=0.0001,  # 万1
        ),
    }

    training = TrainingConfig(
        episode_length=episode_length,
        checkpoint_interval=checkpoint_interval,
        neat_config_path=config_dir,  # 配置目录，Population 会自动选择对应的配置文件
        evolution_interval=evolution_interval,
    )

    demo = DemoConfig(
        host="localhost",
        port=8000,
        tick_interval=100,
    )

    # 鲶鱼配置（如果启用）
    catfish: CatfishConfig | None = None
    if catfish_enabled:
        catfish = CatfishConfig(
            enabled=True,
            multi_mode=True,  # 三种行为模式同时运行
        )

    return Config(
        market=market, agents=agents, training=training, demo=demo, catfish=catfish
    )
