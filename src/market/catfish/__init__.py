"""
鲶鱼模块

提供鲶鱼（Catfish）的各种实现。鲶鱼是一种特殊的市场参与者，
用于在训练中引入外部扰动，增加市场动态性。

主要类：
- CatfishBase: 鲶鱼抽象基类
- TrendCreatorCatfish: 趋势创造者鲶鱼
- MeanReversionCatfish: 逆势操作型鲶鱼
- RandomTradingCatfish: 随机买卖型鲶鱼

工厂函数：
- create_catfish: 根据配置创建鲶鱼实例
- create_all_catfish: 创建所有三种鲶鱼实例
"""

from src.config.config import CatfishConfig, CatfishMode
from src.market.catfish.catfish_base import CatfishBase
from src.market.catfish.trend_following import TrendCreatorCatfish, TrendFollowingCatfish
from src.market.catfish.mean_reversion import MeanReversionCatfish
from src.market.catfish.random_trading import RandomTradingCatfish


def create_catfish(
    catfish_id: int,
    config: CatfishConfig,
    initial_balance: float = 0.0,
    leverage: float = 10.0,
    maintenance_margin_rate: float = 0.05,
) -> CatfishBase:
    """
    根据配置创建鲶鱼实例

    工厂函数，根据 config.mode 创建对应类型的鲶鱼。

    Args:
        catfish_id: 鲶鱼ID（应为负数）
        config: 鲶鱼配置
        initial_balance: 初始余额
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率

    Returns:
        对应类型的鲶鱼实例

    Raises:
        ValueError: 如果 mode 不是有效的 CatfishMode
    """
    if config.mode in (CatfishMode.TREND_CREATOR, CatfishMode.TREND_FOLLOWING):
        return TrendCreatorCatfish(
            catfish_id, config,
            initial_balance, leverage, maintenance_margin_rate
        )
    elif config.mode == CatfishMode.MEAN_REVERSION:
        return MeanReversionCatfish(
            catfish_id, config,
            initial_balance, leverage, maintenance_margin_rate
        )
    elif config.mode == CatfishMode.RANDOM:
        return RandomTradingCatfish(
            catfish_id, config,
            initial_balance, leverage, maintenance_margin_rate
        )
    else:
        raise ValueError(f"未知的鲶鱼模式: {config.mode}")


def create_all_catfish(
    config: CatfishConfig,
    initial_balance: float,
    leverage: float = 10.0,
    maintenance_margin_rate: float = 0.05,
) -> list[CatfishBase]:
    """
    创建所有三种鲶鱼实例

    三种鲶鱼同时运行，每个 tick 各自独立随机决定是否行动。

    Args:
        config: 鲶鱼配置
        initial_balance: 初始余额
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率

    Returns:
        三种鲶鱼实例的列表
    """
    catfish_list: list[CatfishBase] = [
        TrendCreatorCatfish(
            -1, config,
            initial_balance, leverage, maintenance_margin_rate
        ),
        MeanReversionCatfish(
            -2, config,
            initial_balance, leverage, maintenance_margin_rate
        ),
        RandomTradingCatfish(
            -3, config,
            initial_balance, leverage, maintenance_margin_rate
        ),
    ]

    return catfish_list


__all__ = [
    "CatfishBase",
    "TrendCreatorCatfish",
    "TrendFollowingCatfish",  # 向后兼容别名
    "MeanReversionCatfish",
    "RandomTradingCatfish",
    "create_catfish",
    "create_all_catfish",
]
