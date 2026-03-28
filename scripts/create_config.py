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
    ASConfig,
    Config,
    DemoConfig,
    MarketConfig,
    NoiseTraderConfig,
    TrainingConfig,
)


def create_default_config(
    episode_length: int = 1000,
    checkpoint_interval: int = 10,
    config_dir: str = "config",
    evolution_interval: int = 10,
) -> Config:
    """创建默认配置

    Args:
        episode_length: 每个 episode 的 tick 数量
        checkpoint_interval: 检查点间隔（episode 数）
        config_dir: 配置文件目录（Population 会在此目录下查找对应的 NEAT 配置）
        evolution_interval: 每多少个 episode 进化一次

    Returns:
        默认配置对象
    """
    market = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=5,
        ema_alpha=0.9,
    )
    maker_initial_balance = 10_000_000.0  # 做市商初始资金 10M
    maker_leverage = 10.0

    agents = {
        AgentType.RETAIL_PRO: AgentConfig(
            count=2400,
            initial_balance=20000.0,  # 2万
            leverage=10.0,
            maintenance_margin_rate=0.05,  # 5%
            maker_fee_rate=0.0002,  # 万2
            taker_fee_rate=0.0005,  # 万5
        ),
        AgentType.MARKET_MAKER: AgentConfig(
            count=600,
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
        neat_config_path=config_dir,
        evolution_interval=evolution_interval,
    )

    demo = DemoConfig(
        host="localhost",
        port=8000,
        tick_interval=100,
    )

    return Config(
        market=market,
        agents=agents,
        training=training,
        demo=demo,
        noise_trader=NoiseTraderConfig(),
        as_model=ASConfig(),
    )
