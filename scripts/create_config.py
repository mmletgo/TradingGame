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
    CatfishMode,
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
    catfish_mode: str = "trend_following",
    catfish_fund_multiplier: float = 3.0,
) -> Config:
    """创建默认配置

    Args:
        episode_length: 每个 episode 的 tick 数量
        checkpoint_interval: 检查点间隔（episode 数）
        config_dir: 配置文件目录（Population 会在此目录下查找对应的 NEAT 配置）
        catfish_enabled: 是否启用鲶鱼
        catfish_mode: 鲶鱼行为模式
        catfish_fund_multiplier: 鲶鱼资金倍数

    Returns:
        默认配置对象
    """
    market = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
        ema_alpha=1.0,
    )
    maker_initial_balance = 50_000_000.0  # 做市商初始资金 5000万
    maker_leverage = 10.0

    agents = {
        AgentType.RETAIL: AgentConfig(
            count=10000,
            initial_balance=200000.0,  # 20万
            leverage=10.0,
            maintenance_margin_rate=0.05,  # 10%
            maker_fee_rate=0.0002,  # 万2
            taker_fee_rate=0.0005,  # 万5
        ),
        AgentType.RETAIL_PRO: AgentConfig(
            count=100,
            initial_balance=200000.0,  # 20万
            leverage=10.0,
            maintenance_margin_rate=0.05,  # 10%
            maker_fee_rate=0.0002,  # 万2
            taker_fee_rate=0.0005,  # 万5
        ),
        AgentType.WHALE: AgentConfig(
            count=100,  # 庄家（合并多空）
            initial_balance=10000000.0,  # 1000万
            leverage=10.0,
            maintenance_margin_rate=0.05,  # 10%
            maker_fee_rate=-0.0001,  # 负万1 (maker rebate)
            taker_fee_rate=0.0001,  # 万1
        ),
        AgentType.MARKET_MAKER: AgentConfig(
            count=100,
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
    )

    demo = DemoConfig(
        host="localhost",
        port=8000,
        tick_interval=100,
    )

    # 鲶鱼配置（如果启用）
    catfish: CatfishConfig | None = None
    if catfish_enabled:
        mode_map = {
            "trend_following": CatfishMode.TREND_FOLLOWING,
            "cycle_swing": CatfishMode.CYCLE_SWING,
            "mean_reversion": CatfishMode.MEAN_REVERSION,
        }
        catfish = CatfishConfig(
            enabled=True,
            mode=mode_map.get(catfish_mode, CatfishMode.TREND_FOLLOWING),
            fund_multiplier=catfish_fund_multiplier,
            market_maker_base_fund=maker_initial_balance * maker_leverage,
        )

    return Config(
        market=market, agents=agents, training=training, demo=demo, catfish=catfish
    )
